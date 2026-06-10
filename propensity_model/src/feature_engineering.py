import logging
import numpy as np
import pandas as pd

try:
    from lifetimes import BetaGeoFitter
    _LIFETIMES_AVAILABLE = True
except ImportError:
    _LIFETIMES_AVAILABLE = False

from config import COLD_START_THRESHOLD_DAYS, ANOMALY_PERIODS
logger = logging.getLogger(__name__)


def _buys(tx):
    return tx[tx["transactionType"] == "Buy"]

def _real_buys(tx):
    return tx[(tx["transactionType"] == "Buy") & (tx["is_synthetic"] == 0)]

def _sells(tx):
    return tx[tx["transactionType"] == "Sell"]

def _within(tx: pd.DataFrame, snapshot_date: pd.Timestamp, days: int):
    cutoff = snapshot_date - pd.Timedelta(days=days)
    return tx[tx["timestamp"] > cutoff]



# Группа 1: Профиль клиента
def compute_profile_features(profile_df: pd.DataFrame):
    cols = ["customerID", "customerType", "riskLevel",
            "investmentCapacity", "is_capacity_missing"]
    out = profile_df[cols].set_index("customerID")

    return out



# Группа 2: RFM
def compute_rfm_features(tx_history: pd.DataFrame, snapshot_date: pd.Timestamp):
    buys_all  = _real_buys(tx_history)
    sells_all = _sells(tx_history)

    # Давность: дни с последней покупки
    last_buy_date = (buys_all.groupby("customerID")["timestamp"].max().rename("last_buy_date"))

    # Частота: число покупок за 30 и 90 дней
    buys_30d = _within(buys_all, snapshot_date, 30)
    buys_90d = _within(buys_all, snapshot_date, 90)
    num_buys_30d = (buys_30d.groupby("customerID").size().rename("num_buys_30d"))
    num_buys_90d = (buys_90d.groupby("customerID").size().rename("num_buys_90d"))

    # Сумма покупок за последние 30 дней
    buy_value_30d  = buys_30d.groupby("customerID")["totalValue"]
    sum_value_30d  = buy_value_30d.sum().rename("sum_value_30d")
    avg_ticket_30d = buy_value_30d.mean().rename("avg_ticket_30d")

    # Количество продаж за 30 дней
    sells_30d = _within(sells_all, snapshot_date, 30)
    num_sells_30d = (sells_30d.groupby("customerID").size().rename("num_sells_30d"))

    rfm = pd.concat([last_buy_date, num_buys_30d, num_buys_90d, sum_value_30d, avg_ticket_30d, num_sells_30d],
                    axis=1)
    rfm["days_since_last_buy"] = ((snapshot_date - rfm["last_buy_date"]).dt.days.fillna(9999).astype(int))
    rfm = rfm.drop(columns=["last_buy_date"])
    rfm[["num_buys_30d", "num_buys_90d", "num_sells_30d"]] = (rfm[["num_buys_30d", "num_buys_90d", "num_sells_30d"]].fillna(0).astype(int))
    rfm[["sum_value_30d", "avg_ticket_30d"]] = (rfm[["sum_value_30d", "avg_ticket_30d"]].fillna(0.0))
    rfm["log1p_avg_ticket_30d"] = np.log1p(rfm["avg_ticket_30d"])

    return rfm



# Группа 3 — Временная динамика
def compute_temporal_features(tx_history: pd.DataFrame, snapshot_date: pd.Timestamp, rfm_df: pd.DataFrame):
    buys_all = _real_buys(tx_history)
    buys_7d  = _within(buys_all, snapshot_date, 7)

    num_buys_7d = (buys_7d.groupby("customerID").size().rename("num_buys_7d").fillna(0))
    temp = pd.DataFrame(index=rfm_df.index)

    # Ускорение активности: покупки за 7д vs 30д (рост/спад)
    temp = temp.join(num_buys_7d, how="left")
    temp["num_buys_7d"] = temp["num_buys_7d"].fillna(0)
    temp["activity_trend"] = (temp["num_buys_7d"] / (rfm_df["num_buys_30d"] + 1)).clip(upper=1.0)
    temp["month"] = snapshot_date.month

    # Флаг cold-start
    real_tx = tx_history[tx_history["is_synthetic"] == 0]
    first_tx_date = (real_tx.groupby("customerID")["timestamp"].min().rename("first_tx_date"))
    temp = temp.join(first_tx_date, how="left")
    tenure = (snapshot_date - temp["first_tx_date"]).dt.days.fillna(0)
    temp["is_new_customer"] = (tenure < COLD_START_THRESHOLD_DAYS).astype(int)
    temp = temp.drop(columns=["first_tx_date", "num_buys_7d"])

    return temp



# Группа 4 — Время с последнего события
def compute_timesince_features(tx_history: pd.DataFrame, snapshot_date: pd.Timestamp):
    last_tx = (tx_history.groupby("customerID")["timestamp"].max().rename("last_tx_date"))
    real_tx = tx_history[tx_history["is_synthetic"] == 0]
    first_tx = real_tx.groupby("customerID")["timestamp"].min().rename("first_tx_date")
    ts = pd.concat([last_tx, first_tx], axis=1)
    ts["days_since_last_tx"] = ((snapshot_date - ts["last_tx_date"]).dt.days.fillna(9999).astype(int))
    ts["customer_tenure_days"] = ((snapshot_date - ts["first_tx_date"]).dt.days.fillna(0).astype(int))

    return ts[["days_since_last_tx", "customer_tenure_days"]]



# Группа 5 — Портфельные признаки
def compute_portfolio_features(tx_history: pd.DataFrame, assets_df: pd.DataFrame):
    buys = _real_buys(tx_history)

    if buys.empty:
        logger.warning("В истории нет покупок — признаки портфеля будут нулевыми")
        empty = pd.DataFrame(columns=["num_unique_assets", "share_stocks", "share_bonds", "share_funds"])
        empty.index.name = "customerID"
        return empty

    buys_with_cat = buys.merge(assets_df[["ISIN", "assetCategory"]], on="ISIN", how="left")
    buys_with_cat["assetCategory"] = (buys_with_cat["assetCategory"].fillna("Unknown"))

    unique_assets = (buys_with_cat.groupby("customerID")["ISIN"].nunique().rename("num_unique_assets"))

    cat_counts = (buys_with_cat.groupby(["customerID", "assetCategory"]).size().unstack(fill_value=0))
    for col in ["Stock", "Bond", "MTF"]:
        if col not in cat_counts.columns:
            cat_counts[col] = 0

    total = cat_counts.sum(axis=1).replace(0, np.nan)
    portfolio = pd.DataFrame(index=cat_counts.index)
    portfolio["share_stocks"] = (cat_counts["Stock"] / total).fillna(0.0)
    portfolio["share_bonds"] = (cat_counts["Bond"] / total).fillna(0.0)
    portfolio["share_funds"] = (cat_counts["MTF"] / total).fillna(0.0)

    portfolio = portfolio.join(unique_assets, how="left")
    portfolio["num_unique_assets"] = (portfolio["num_unique_assets"].fillna(0).astype(int))

    return portfolio[["num_unique_assets", "share_stocks", "share_bonds", "share_funds"]]


# Группа 6 —  Рыночный контекст
def compute_market_features(prices_df: pd.DataFrame, snapshot_date: pd.Timestamp, lookback_days: int = 30):
    cutoff = snapshot_date - pd.Timedelta(days=lookback_days)
    window = prices_df[(prices_df["timestamp"] > cutoff) & (prices_df["timestamp"] <= snapshot_date)].copy()

    if len(window) < 5:
        return {
            "market_return_7d": 0.0,
            "market_return_30d": 0.0,
            "market_volatility_30d": 0.0,
            "market_drawdown": 0.0,
        }

    daily = (
        window.groupby("timestamp")["closePrice"]
        .median()
        .sort_index()
    )
    daily_returns = daily.pct_change().dropna()

    market_return_30d = float((daily.iloc[-1] / daily.iloc[0]) - 1)
    market_volatility_30d = float(daily_returns.std()) if len(daily_returns) > 1 else 0.0

    rolling_max = daily.cummax()
    drawdown = float(((daily - rolling_max) / rolling_max).min())

    cutoff_7d = snapshot_date - pd.Timedelta(days=7)
    recent = daily[daily.index > cutoff_7d]
    market_return_7d = float(
        (recent.iloc[-1] / recent.iloc[0]) - 1
    ) if len(recent) >= 2 else 0.0

    return {
        "market_return_7d": market_return_7d,
        "market_return_30d": market_return_30d,
        "market_volatility_30d": market_volatility_30d,
        "market_drawdown": drawdown,
    }


def compute_cadence_features(
    tx_history: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    min_buys_for_gap: int = 2,
) -> pd.DataFrame:
    buys = tx_history[
        (tx_history["transactionType"] == "Buy") &
        (tx_history["is_synthetic"] == 0)
    ].copy().sort_values(["customerID", "timestamp"])

    all_customers = pd.Index(
        tx_history["customerID"].unique(), name="customerID"
    )

    total_buys = (
        buys.groupby("customerID").size()
        .rename("total_buys_lifetime")
        .reindex(all_customers, fill_value=0)
    )

    last_buy = buys.groupby("customerID")["timestamp"].max()
    current_pause = (
        (snapshot_date - last_buy).dt.days
        .rename("current_pause_days")
        .reindex(all_customers, fill_value=9999)
        .astype(int)
    )

    def _compute_gaps(grp):
        diffs = grp.sort_values().diff().dt.days.dropna()
        return diffs[diffs > 0]

    gaps_raw = (
        buys.groupby("customerID")["timestamp"]
        .apply(_compute_gaps)
    )

    median_gap = gaps_raw.groupby(level=0).median().rename("median_buy_interval")
    std_gap = gaps_raw.groupby(level=0).std().rename("std_buy_interval")
    max_gap = gaps_raw.groupby(level=0).max().rename("max_historical_gap")
    p90_gap = gaps_raw.groupby(level=0).quantile(0.90).rename("p90_historical_gap")
    count_gaps = gaps_raw.groupby(level=0).count().rename("_n_gaps")    

    gap_stats = pd.concat([median_gap, std_gap, max_gap, p90_gap, count_gaps], axis=1)
    gap_stats = gap_stats.reindex(all_customers)

    gap_stats["has_cadence_data"] = (
        gap_stats["_n_gaps"].fillna(0) >= 2
    ).astype(int)


    def compute_mad_interval(grp):
        diffs = grp.sort_values().diff().dt.days.dropna()
        if len(diffs) < 2:
            return np.nan
        med = diffs.median()
        return (diffs - med).abs().median()

    mad_gap = (
        buys.groupby("customerID")["timestamp"]
        .apply(compute_mad_interval)
        .rename("mad_buy_interval")
    )

    pause_with_stats = pd.concat([current_pause, gap_stats, mad_gap], axis=1)

    pause_with_stats["pause_zscore"] = np.where(
        pause_with_stats["has_cadence_data"] == 1,
        (
            (pause_with_stats["current_pause_days"] - pause_with_stats["median_buy_interval"]) /
            (1.4826 * pause_with_stats["mad_buy_interval"].fillna(
                pause_with_stats["mad_buy_interval"].median()
            ) + 1e-6)
        ).clip(-3, 3),
        np.nan
    )


    pause_with_stats["survived_similar_pause"] = np.where(
        pause_with_stats["has_cadence_data"] == 1,
        (
            pause_with_stats["current_pause_days"] <=
            pause_with_stats["p90_historical_gap"]
        ).astype(int),
        np.nan
    )

    pause_with_stats["pause_near_personal_max"] = np.where(
        pause_with_stats["has_cadence_data"] == 1,
        (
            pause_with_stats["current_pause_days"] /
            (pause_with_stats["p90_historical_gap"].fillna(1) + 1)
        ).clip(0, 2),
        np.nan
    )

    buys["year_before"] = (
        buys["timestamp"] > (snapshot_date - pd.Timedelta(days=365))
    )
    buys_last_year = (
        buys[buys["year_before"]]
        .groupby("customerID").size()
        .rename("buys_last_year")
        .reindex(all_customers, fill_value=0)
    )

    first_buy = buys.groupby("customerID")["timestamp"].min()
    tenure_buy = (
        (snapshot_date - first_buy).dt.days
        .rename("tenure_buy_days")
        .reindex(all_customers, fill_value=0)
    )

    def has_returned_after_long_gap(grp, threshold=90):
        dates = grp.sort_values()
        diffs = dates.diff().dt.days.dropna()
        return int((diffs > threshold).any())

    returned_after_gap = (
        buys.groupby("customerID")["timestamp"]
        .apply(has_returned_after_long_gap)
        .rename("has_returned_after_90d_gap")
        .reindex(all_customers, fill_value=0)
    )

    def compute_activity_trend(grp, snapshot):
        dates = grp.sort_values()
        if len(dates) < 2:
            return np.nan
        midpoint = dates.iloc[0] + (dates.iloc[-1] - dates.iloc[0]) / 2
        n_first  = (dates <= midpoint).sum()
        n_second = (dates > midpoint).sum()
        span_days = max((snapshot - dates.iloc[0]).days, 1)
        return (n_second - n_first) / (span_days / 180)

    long_term_trend = (
        buys.groupby("customerID")["timestamp"]
        .apply(compute_activity_trend, snapshot_date)
        .rename("long_term_activity_trend")
        .reindex(all_customers, fill_value=np.nan)
    )

    out = pd.concat([
        gap_stats[["median_buy_interval", "std_buy_interval",
                   "max_historical_gap", "has_cadence_data"]],
        total_buys,
        current_pause,
        buys_last_year,
        tenure_buy,
        returned_after_gap,
        long_term_trend,
        pause_with_stats[["pause_zscore", "survived_similar_pause",
                           "pause_near_personal_max"]],
    ], axis=1)

    out["total_buys_lifetime"]      = out["total_buys_lifetime"].fillna(0)
    out["has_cadence_data"]         = out["has_cadence_data"].fillna(0).astype(int)
    out["has_returned_after_90d_gap"] = out["has_returned_after_90d_gap"].fillna(0)
    out["buys_last_year"]           = out["buys_last_year"].fillna(0).astype(int)
    out["tenure_buy_days"]          = out["tenure_buy_days"].fillna(0).astype(int)
    out["long_term_activity_trend"]  = out["long_term_activity_trend"].fillna(0.0)

    return out


# Группа 8 - Сезонность
def compute_seasonal_features(tx_history: pd.DataFrame, snapshot_date: pd.Timestamp):
    buys = _real_buys(tx_history).copy()
    buys["month"] = buys["timestamp"].dt.month
    buys["year"] = buys["timestamp"].dt.year
    buys["quarter"] = buys["timestamp"].dt.quarter

    snap_month = snapshot_date.month
    snap_quarter = snapshot_date.quarter
    last_year = snapshot_date.year - 1

    bought_same_month = (
        buys[(buys["year"] == last_year) & (buys["month"] == snap_month)]
        .groupby("customerID").size()
        .gt(0).astype(int)
        .rename("bought_same_month_last_year")
    )

    bought_same_quarter = (
        buys[(buys["year"] == last_year) & (buys["quarter"] == snap_quarter)]
        .groupby("customerID").size()
        .gt(0).astype(int)
        .rename("bought_same_quarter_last_year")
    )

    num_same_month = (
        buys[buys["month"] == snap_month]
        .groupby("customerID").size()
        .rename("num_buys_same_month_hist")
    )

    return pd.concat(
        [bought_same_month, bought_same_quarter, num_same_month],
        axis=1
    ).fillna(0).astype(int)



# Группа 9а — Историческая реакция клиента на рыночные события
def compute_market_response_features(
    tx_history: pd.DataFrame,
    prices_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    lookback_window: int = 7,
) -> pd.DataFrame:
    buys = _real_buys(tx_history).copy()

    if buys.empty or prices_df.empty:
        empty = pd.DataFrame(
            columns=["buys_on_correction_rate"]
        )
        empty.index.name = "customerID"
        return empty

    unique_dates = buys["timestamp"].dt.normalize().unique()

    date_market = {}
    for dt in unique_dates:
        dt = pd.Timestamp(dt)
        cutoff = dt - pd.Timedelta(days=lookback_window)
        window = prices_df[
            (prices_df["timestamp"] > cutoff) &
            (prices_df["timestamp"] <= dt)
        ]
        if len(window) < 5:
            date_market[dt] = {"is_correction": 0, "is_high_vol": 0}
            continue

        daily = (
            window.groupby("timestamp")["closePrice"]
            .median()
            .sort_index()
        )
        daily_returns = daily.pct_change().dropna()

        ret_30d = float((daily.iloc[-1] / daily.iloc[0]) - 1)
        is_correction = int(ret_30d < -0.05)

        date_market[dt] = {
            "is_correction": is_correction,
        }

    buys["date_norm"] = buys["timestamp"].dt.normalize()
    buys["is_correction"] = buys["date_norm"].map(
        lambda d: date_market.get(d, {}).get("is_correction", 0)
    )

    agg = buys.groupby("customerID").agg(
        total_buys_for_rate=("is_correction", "count"),
        correction_buys=("is_correction", "sum"),
    )

    result = pd.DataFrame(index=agg.index)
    result["buys_on_correction_rate"] = (
        agg["correction_buys"] / agg["total_buys_for_rate"]
    ).fillna(0.0)

    return result


def compute_anomaly_features(
    tx_history: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:

    buys = tx_history[
        (tx_history["transactionType"] == "Buy") &
        (tx_history["is_synthetic"] == 0)
    ]

    all_customers = pd.Index(
        tx_history["customerID"].unique(), name="customerID"
    )

    last_buy  = buys.groupby("customerID")["timestamp"].max().reindex(all_customers)
    first_buy = buys.groupby("customerID")["timestamp"].min().reindex(all_customers)

    snap_in_anomaly    = pd.Series(0, index=all_customers, name="snapshot_in_anomaly")
    pause_in_anomaly   = pd.Series(0, index=all_customers, name="pause_includes_anomaly")
    history_in_anomaly = pd.Series(0, index=all_customers, name="history_includes_anomaly")

    for period_start, period_end in ANOMALY_PERIODS:
        p_start = pd.Timestamp(period_start)
        p_end   = pd.Timestamp(period_end)

        if p_start <= snapshot_date <= p_end:
            snap_in_anomaly[:] = 1

        if snapshot_date >= p_start:
            pause_flag = (last_buy <= p_end).fillna(0).astype(int)
            pause_in_anomaly = (pause_in_anomaly | pause_flag).astype(int)

        history_flag = (
            (first_buy <= p_end) & (last_buy >= p_start)
        ).fillna(0).astype(int)
        history_in_anomaly = (history_in_anomaly | history_flag).astype(int)

    return pd.DataFrame({
        "snapshot_in_anomaly":    snap_in_anomaly.fillna(0).astype(int),
        "pause_includes_anomaly": pause_in_anomaly.fillna(0).astype(int),
        "history_includes_anomaly": history_in_anomaly.fillna(0).astype(int),
    }, index=all_customers)


def compute_p_alive_features(
    tx_history: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    min_frequency: int = 1,
    penalizer_coef: float = 0.01,
) -> pd.DataFrame:
    empty = pd.DataFrame(
        {"p_alive": pd.Series(dtype=float)},
        index=pd.Index([], name="customerID"),
    )

    if not _LIFETIMES_AVAILABLE:
        logger.warning("lifetimes не установлен — p_alive не вычисляется. pip install lifetimes")
        return empty

    buys = tx_history[
        (tx_history["transactionType"] == "Buy") &
        (tx_history["is_synthetic"] == 0)
    ].copy()

    if buys.empty:
        return empty

    first_buy = buys.groupby("customerID")["timestamp"].min()
    last_buy  = buys.groupby("customerID")["timestamp"].max()
    frequency = buys.groupby("customerID").size() - 1

    T_days       = (snapshot_date - first_buy).dt.days.clip(lower=1)
    recency_days = (last_buy - first_buy).dt.days.clip(lower=0)

    rfm = pd.DataFrame({
        "frequency": frequency,
        "recency":   recency_days,
        "T":         T_days,
    }).dropna()

    rfm = rfm[rfm["frequency"] >= min_frequency]

    if len(rfm) < 10:
        logger.warning(f"  BG/NBD: слишком мало клиентов ({len(rfm)}) — p_alive пропускается")
        return empty

    try:
        bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"], verbose=False)

        rfm["p_alive"] = bgf.conditional_probability_alive(
            rfm["frequency"], rfm["recency"], rfm["T"]
        )
        result = rfm[["p_alive"]].copy()
        result.index.name = "customerID"
        return result

    except Exception as e:
        logger.warning(f"  BG/NBD fit failed: {e} — p_alive пропускается")
        return empty



# Группа 9 - Взаимодействия (только для reactivation)
def compute_interaction_features(features_df: pd.DataFrame, market_ctx: dict):
    f = pd.DataFrame(index=features_df.index)

    if "market_volatility_30d" in features_df.columns:
        median_vol = features_df["market_volatility_30d"].median()
    else:
        median_vol = 0.02
    is_high_vol = float(market_ctx["market_volatility_30d"] > median_vol)

    f["value_buyer_x_correction"] = (
        features_df["buys_on_correction_rate"].fillna(0.0) * 
        abs(market_ctx["market_drawdown"])  # степень коррекции, не флаг
    )    

    return f


def compute_all_features(
    tx_history, profile_df, assets_df, prices_df, snapshot_date,
    mode: str = "timing", market_ctx=None
):
    g1 = compute_profile_features(profile_df)
    g2 = compute_rfm_features(tx_history, snapshot_date)
    g3 = compute_temporal_features(tx_history, snapshot_date, g2)
    g4 = compute_timesince_features(tx_history, snapshot_date)
    g5 = compute_portfolio_features(tx_history, assets_df)
    g7 = compute_seasonal_features(tx_history, snapshot_date)
    g10 = compute_anomaly_features(tx_history, snapshot_date)
    
    if market_ctx is None:
        market_ctx = compute_market_features(prices_df, snapshot_date)

    if mode == "reactivation":
        g6  = compute_cadence_features(tx_history, snapshot_date)
        g9a = compute_market_response_features(tx_history, prices_df, snapshot_date)
        g9b = compute_p_alive_features(tx_history, snapshot_date)

    base = pd.DataFrame(index=pd.Index(tx_history["customerID"].unique(), name="customerID"))
    features = (
        base
        .join(g1, how="left")
        .join(g2, how="left")
        .join(g3, how="left")
        .join(g4, how="left")
        .join(g5, how="left")
        .join(g7, how="left")
        .join(g10, how="left")
    )

    if mode == "reactivation":
        features = (
            features
            .join(g6, how="left")
            .join(g9a, how="left")
            .join(g9b, how="left")
        )

    count_cols = ["num_buys_30d", "num_buys_90d", "num_sells_30d", "num_unique_assets"]
    features[count_cols] = features[count_cols].fillna(0).astype(int)
    value_cols = ["sum_value_30d", "avg_ticket_30d", "log1p_avg_ticket_30d", "share_stocks", "share_bonds", "share_funds", "is_capacity_missing"]
    features[value_cols] = features[value_cols].fillna(0.0)
    recency_cols = ["days_since_last_buy", "days_since_last_tx"]
    features[recency_cols] = features[recency_cols].fillna(9999).astype(int)
    
    features["customer_tenure_days"] = features["customer_tenure_days"].fillna(0).astype(int)
    features["activity_trend"] = features["activity_trend"].fillna(0.0)
    features["is_new_customer"] = features["is_new_customer"].fillna(1).astype(int)
    features["month"] = features["month"].fillna(snapshot_date.month).astype(int)

    features["riskLevel"] = features["riskLevel"].fillna("Not_Available")
    features["investmentCapacity"] = features["investmentCapacity"].fillna("Not_Available")
    features["customerType"] = features["customerType"].fillna("Unknown")

    for col, val in market_ctx.items():
        features[col] = val

    features["aggressive_x_recency"] = (
        (features["riskLevel"] == "Aggressive").astype(int) *
        np.log1p(features["days_since_last_buy"])
    )
    features["stocks_x_market_correction"] = (
        features["share_stocks"] *
        float(market_ctx["market_drawdown"] < -0.05)
    )

    cadence_defaults = {
        "median_buy_interval": 9999,
        "std_buy_interval": 0.0,
        "max_historical_gap": 9999,
        "total_buys_lifetime": 0,
        "current_pause_days": 9999,
        "pause_zscore": 0.0,
        "survived_similar_pause": 0,
        "pause_near_personal_max": 1.0,
        "long_term_activity_trend": 0.0,
    }
    seasonal_defaults = {
        "bought_same_month_last_year": 0,
        "bought_same_quarter_last_year": 0,
        "num_buys_same_month_hist": 0,
    }
    for col, val in {**cadence_defaults, **seasonal_defaults}.items():
        if col in features.columns:
            features[col] = features[col].fillna(val)

    if "buys_on_correction_rate" in features.columns:
        features["buys_on_correction_rate"] = features["buys_on_correction_rate"].fillna(0.0)

    for col in ["snapshot_in_anomaly", "pause_includes_anomaly", "history_includes_anomaly"]:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)

    logger.info(f"  Feature matrix: {len(features):,} customers × {len(features.columns)} features")

    return features
