import logging
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    TIMING_ALL_FEATURES,
    REACTIVATION_ALL_FEATURES,
    ACTIVE_WINDOW_DAYS,
    DORMANCY_THRESHOLD_DAYS,
    HORIZON_DAYS,
    REACTIVATION_HORIZON_DAYS,
    PREDICTION_GAP_DAYS,
    MIN_TX_REACT,
    ID_COLS,
    SNAPSHOT_FREQ,
    TARGET_COL,
    #TRAIN_END,
    #VALID_END,
    DORMANCY_MULTIPLIER, 
    DORMANCY_MIN_DAYS,
)
from config import SNAPSHOT_START
from data_loader import get_customer_profile_at
from feature_engineering import compute_all_features, compute_market_features, compute_interaction_features

logger = logging.getLogger(__name__)



# Генерация временных срезов
def generate_snapshot_dates(transactions: pd.DataFrame, freq: str = SNAPSHOT_FREQ, horizon_days: int = HORIZON_DAYS):
    min_date = transactions["timestamp"].min()
    max_date = transactions["timestamp"].max()

    start = max(min_date + pd.Timedelta(days=ACTIVE_WINDOW_DAYS), pd.Timestamp(SNAPSHOT_START))
    end = max_date - pd.Timedelta(days=horizon_days)
    if start >= end:
        raise ValueError(
            f"Not enough data to generate snapshots: "
            f"effective range {start.date()} – {end.date()}"
        )
    dates = pd.date_range(start=start, end=end, freq=freq).tolist()
    logger.info(
        f"Generated {len(dates)} snapshot dates "
        f"({dates[0].date()} → {dates[-1].date()}, freq={freq})"
    )

    return dates



# Определение активных клиентов
def get_active_customers(tx_history, snapshot_date, active_window_days=ACTIVE_WINDOW_DAYS):
    cutoff = snapshot_date - pd.Timedelta(days=active_window_days)
    tx_recent = tx_history[tx_history["timestamp"] > cutoff]
    active = tx_recent[tx_recent["transactionType"] == "Buy"]["customerID"].unique()

    return active

def get_reactivation_candidates(tx_history, min_tx_react=MIN_TX_REACT):
    buy_counts = tx_history[tx_history["transactionType"] == "Buy"].groupby("customerID").size()
    eligible_ids = buy_counts[buy_counts >= min_tx_react].index

    return eligible_ids



# Таргет
def build_labels(transactions: pd.DataFrame, snapshot_date: pd.Timestamp, customer_ids: np.ndarray,
    horizon_days=HORIZON_DAYS, gap_days=PREDICTION_GAP_DAYS,
):
    window_start = snapshot_date + pd.Timedelta(days=gap_days)
    future_end = snapshot_date + pd.Timedelta(days=horizon_days + gap_days)

    buyers = transactions[
        (transactions["timestamp"] > window_start) &
        (transactions["timestamp"] <= future_end) &
        (transactions["transactionType"] == "Buy") &
        (transactions["customerID"].isin(customer_ids))
    ]["customerID"].unique()

    labels = pd.Series(
        data=np.where(np.isin(customer_ids, buyers), 1, 0),
        index=pd.Index(customer_ids, name="customerID"),
        name=TARGET_COL,
    )

    return labels



# Создание временного среза для Timing
def build_snapshot(snapshot_date: pd.Timestamp, transactions: pd.DataFrame, customers: pd.DataFrame, assets: pd.DataFrame, prices: pd.DataFrame):
    tx_history = transactions[transactions["timestamp"] <= snapshot_date].copy()
    
    active_ids = get_active_customers(tx_history, snapshot_date)
    if len(active_ids) == 0:
        logger.warning(f"Нет активных клиентов для среза {snapshot_date.date()}")
        return pd.DataFrame()
    
    last_buy = (
        tx_history[tx_history["transactionType"] == "Buy"]
        .groupby("customerID")["timestamp"].max()
    )
    days_since = (snapshot_date - last_buy).dt.days.reindex(pd.Index(active_ids, name="customerID"))

    personal_thresholds_warm = _compute_personal_thresholds(
        tx_history, active_ids,
        multiplier=DORMANCY_MULTIPLIER,
        min_days=DORMANCY_MIN_DAYS,
    )
    warm_mask = days_since <= personal_thresholds_warm
    active_ids = warm_mask[warm_mask].index

    if len(active_ids) == 0:
        logger.warning(f"Нет warm клиентов для среза {snapshot_date.date()}")
        return pd.DataFrame()

    profile_at_T = get_customer_profile_at(customers, snapshot_date)
    profile_at_T = profile_at_T[profile_at_T["customerID"].isin(active_ids)] # оставляем только профили активных клиентов
    missing_profiles = set(active_ids) - set(profile_at_T["customerID"])
    if missing_profiles:
        logger.warning(
            f"  {len(missing_profiles):,} активных клиентов без профиля "
            f"на {snapshot_date.date()} — исключены"
        )
    tx_history = tx_history[tx_history["customerID"].isin(active_ids)] #оставляем историю только по активным клиентам

    features = compute_all_features(
        tx_history=tx_history,
        profile_df=profile_at_T,
        assets_df=assets,
        prices_df=prices,
        snapshot_date=snapshot_date,
    )
    labels = build_labels(transactions, snapshot_date, features.index.values)

    snapshot = features.copy()
    snapshot[TARGET_COL] = labels
    snapshot["snapshot_date"] = snapshot_date
    snapshot = snapshot.reset_index()

    pos_rate = labels.mean()
    logger.info(
        f"  Срез {snapshot_date.date()}: "
        f"{len(snapshot):,} клиентов, "
        f"Доля с таргетом=1: {pos_rate:.2%}"
    )

    return snapshot



# Сборка итогового датасета для Timing
def build_dataset(transactions: pd.DataFrame, customers: pd.DataFrame, assets: pd.DataFrame, prices: pd.DataFrame, snapshot_dates: list[pd.Timestamp] | None = None):
    if snapshot_dates is None:
        snapshot_dates = generate_snapshot_dates(transactions)

    snapshots = []
    for i, t in enumerate(snapshot_dates, 1):
        logger.info(f"Сборка среза {i}/{len(snapshot_dates)} — {t.date()}")
        snap = build_snapshot(t, transactions, customers, assets, prices)
        if not snap.empty:
            snapshots.append(snap)

    if not snapshots:
        raise RuntimeError("Срезы не созданы — проверьте диапазон дат")

    dataset = pd.concat(snapshots, ignore_index=True)

    logger.info(
        f"\nСводка:\n"
        f"  Всего строк: {len(dataset):,}\n"
        f"  Уникальных клиентов: {dataset['customerID'].nunique():,}\n"
        f"  Срезов: {dataset['snapshot_date'].nunique()}\n"
        f"  Доля таргета=1: {dataset[TARGET_COL].mean():.2%}\n"
        f"  Диапазон дат: "
        f"{dataset['snapshot_date'].min().date()} – "
        f"{dataset['snapshot_date'].max().date()}"
    )

    return dataset


def _compute_personal_thresholds(tx_history: pd.DataFrame,
                                  candidate_ids,
                                  multiplier: float = DORMANCY_MULTIPLIER,
                                  min_days: int    = DORMANCY_MIN_DAYS,
                                  ) -> pd.Series:
    buys = tx_history[
        (tx_history["transactionType"] == "Buy") &
        (tx_history["is_synthetic"] == 0) &
        (tx_history["customerID"].isin(candidate_ids))
    ].sort_values(["customerID", "timestamp"])

    def _median_interval(grp):
        diffs = grp.diff().dt.days.dropna()
        return diffs.median() if len(diffs) >= 1 else np.nan

    median_intervals = (
        buys.groupby("customerID")["timestamp"]
        .apply(_median_interval)
    )

    thresholds = (median_intervals * multiplier).clip(lower=min_days)
    thresholds = thresholds.reindex(pd.Index(candidate_ids, name="customerID"),
                                    fill_value=min_days)
    return thresholds


# Создание временного среза для Reactivation
def build_reactivation_snapshot(snapshot_date, transactions, customers, assets, prices,
    dormancy_threshold=DORMANCY_THRESHOLD_DAYS, reactivation_horizon=REACTIVATION_HORIZON_DAYS, min_tx=MIN_TX_REACT,
    active_window=ACTIVE_WINDOW_DAYS
):
    tx_history = transactions[transactions["timestamp"] <= snapshot_date].copy()

    # Все клиенты активные за N месяцев
    all_recent = get_active_customers(
        tx_history,
        snapshot_date,
        active_window_days=active_window
    )

    if len(all_recent) == 0:
        logger.warning(f"Нет активных клиентов для среза {snapshot_date.date()}")
        return pd.DataFrame()

    eligible_ids = get_reactivation_candidates(
        tx_history,
        min_tx_react=min_tx
    )

    last_buy = (
        tx_history[tx_history["transactionType"] == "Buy"]
        .groupby("customerID")["timestamp"].max()
    )
    days_since = (snapshot_date - last_buy).dt.days

    # Вычисляем персональные пороги для всех кандидатов
    all_candidates = (
        days_since.index
        .intersection(all_recent)
        .intersection(eligible_ids)
    )
    personal_thresholds = _compute_personal_thresholds(
        tx_history, all_candidates,
        multiplier=DORMANCY_MULTIPLIER,
        min_days=DORMANCY_MIN_DAYS,
    )

    # Клиент dormant если его пауза > его персонального порога
    days_since_candidates = days_since.reindex(all_candidates)
    dormant_mask = days_since_candidates > personal_thresholds
    dormant_ids  = days_since_candidates[dormant_mask].index

    if len(dormant_ids) == 0:
        logger.warning(f"Нет спящих клиентов для среза {snapshot_date.date()}")
        return pd.DataFrame()

    # Сохраняем персональные пороги как признак для модели
    personal_thresh_feature = personal_thresholds[dormant_ids].rename("personal_dormancy_threshold")

    tx_dormant = tx_history[tx_history["customerID"].isin(dormant_ids)]
    profile_at_T = get_customer_profile_at(customers, snapshot_date)
    profile_at_T = profile_at_T[profile_at_T["customerID"].isin(dormant_ids)]

    missing = set(dormant_ids) - set(profile_at_T["customerID"])
    if missing:
        logger.warning(
            f"  {len(missing):,} спящих клиентов без профиля "
            f"на {snapshot_date.date()} — исключены"
        )
        dormant_ids = dormant_ids.difference(missing)
        tx_dormant = tx_dormant[tx_dormant["customerID"].isin(dormant_ids)]

    features = compute_all_features(
        tx_history=tx_dormant,
        profile_df=profile_at_T,
        assets_df=assets,
        prices_df=prices,
        snapshot_date=snapshot_date,
        mode="reactivation"
    )

    # Добавляем взаимодействия — специфичны для reactivation
    market_ctx = compute_market_features(prices, snapshot_date)
    interaction_f = compute_interaction_features(features, market_ctx)
    features = features.join(interaction_f, how="left")

    future_end = snapshot_date + pd.Timedelta(days=reactivation_horizon)
    reactivated = transactions[
        (transactions["timestamp"] > snapshot_date) &
        (transactions["timestamp"] <= future_end) &
        (transactions["transactionType"] == "Buy") &
        (transactions["customerID"].isin(dormant_ids))
    ]["customerID"].unique()

    features[TARGET_COL] = np.where(
        features.index.isin(reactivated), 1, 0
    )
    features["snapshot_date"] = snapshot_date

    # Добавляем персональный порог как признаки
    features = features.join(personal_thresh_feature,  how="left")
    features["personal_dormancy_threshold"] = features["personal_dormancy_threshold"].fillna(DORMANCY_MIN_DAYS)
    
    
    pos_rate = features[TARGET_COL].mean()
    return features.reset_index()



# Сборка итогового датасета для Reactivation
def build_reactivation_dataset(transactions, customers, assets, prices, snapshot_dates=None):
    if snapshot_dates is None:
        snapshot_dates = generate_snapshot_dates(transactions)

    snapshots = []
    for i, t in enumerate(snapshot_dates, 1):
        logger.info(
            f"Reactivation срез {i}/{len(snapshot_dates)} — {t.date()}"
        )
        snap = build_reactivation_snapshot(
            t, transactions, customers, assets, prices
        )
        if not snap.empty:
            snapshots.append(snap)

    if not snapshots:
        raise RuntimeError("Reactivation срезы не созданы")

    dataset = pd.concat(snapshots, ignore_index=True)
    logger.info(
        f"\nReactivation сводка:\n"
        f"  Всего строк: {len(dataset):,}\n"
        f"  Уникальных клиентов: {dataset['customerID'].nunique():,}\n"
        f"  Срезов: {dataset['snapshot_date'].nunique()}\n"
        f"  Доля таргета=1: {dataset[TARGET_COL].mean():.2%}\n"
        f"  Диапазон дат: "
        f"{dataset['snapshot_date'].min().date()} – "
        f"{dataset['snapshot_date'].max().date()}"
    )
    return dataset


def compute_split_dates(transactions: pd.DataFrame) -> tuple[str, str]:
    """Вычисляет TRAIN_END и VALID_END относительно конца данных.

    При текущем датасете (max ≈ конец 2022) воспроизводит:
      VALID_END = 2021-10-01  (TEST_MONTHS=14 от конца)
      TRAIN_END = 2021-01-01  (VALID_MONTHS=9 до VALID_END)
    """
    from config import TEST_MONTHS, VALID_MONTHS
    max_date   = transactions["timestamp"].max()
    valid_end  = max_date - pd.DateOffset(months=TEST_MONTHS)
    train_end  = valid_end - pd.DateOffset(months=VALID_MONTHS)
    # Нормализуем до начала месяца для воспроизводимости
    valid_end  = valid_end.replace(day=1)
    train_end  = train_end.replace(day=1)
    logger.info(
        f"  Динамический сплит: "
        f"TRAIN_END={train_end.date()} | VALID_END={valid_end.date()}"
    )
    return train_end.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")



# Train/validation/test split
def time_split(dataset: pd.DataFrame, transactions: pd.DataFrame | None = None):
    """Разбивка на train/valid/test по snapshot_date.

    Если transactions переданы — вычисляем даты динамически.
    Иначе — берём из config (для обратной совместимости).
    """
    if transactions is not None:
        train_end_str, valid_end_str = compute_split_dates(transactions)
    else:
        from config import TRAIN_END, VALID_END
        train_end_str, valid_end_str = TRAIN_END, VALID_END

    train_end_ts = pd.Timestamp(train_end_str)
    valid_end_ts = pd.Timestamp(valid_end_str)

    train = dataset[dataset["snapshot_date"] <  train_end_ts]
    valid = dataset[(dataset["snapshot_date"] >= train_end_ts) &
                    (dataset["snapshot_date"] <  valid_end_ts)]
    test  = dataset[dataset["snapshot_date"] >= valid_end_ts]

    for name, df in [("Train", train), ("Valid", valid), ("Test", test)]:
        logger.info(
            f"  {name}: {len(df):,} строк | "
            f"таргет=1: {df[TARGET_COL].mean():.2%} | "
            f"диапазон дат: {df['snapshot_date'].min().date()} – "
            f"{df['snapshot_date'].max().date()}"
        )

    return train, valid, test



# X / y
def get_X_y(df, model_type: str = "timing"):
    if model_type == "timing":
        feature_cols = TIMING_ALL_FEATURES
    elif model_type == "reactivation":
        feature_cols = REACTIVATION_ALL_FEATURES
    else:
        raise ValueError(f"Неизвестный model_type: {model_type}")

    # Берём только колонки которые есть в датафрейме
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        logger.warning(f"  Отсутствуют признаки: {missing}")

    X = df[available].copy()
    y = df[TARGET_COL].copy()
    return X, y
