import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from config import (
    TIMING_ALL_FEATURES,
    REACTIVATION_ALL_FEATURES,
    TIMING_CAT_FEATURES,
    REACTIVATION_CAT_FEATURES,
    COLD_START_THRESHOLD_DAYS,
    DORMANCY_THRESHOLD_DAYS,
    MIN_TX_REACT,
    ACTIVE_WINDOW_DAYS,
    MODEL_DIR,
    OUTPUT_DIR,
    TARGET_COL,
    DORMANCY_MULTIPLIER,
    DORMANCY_MIN_DAYS,
    HORIZON_DAYS,
)
from data_loader import (get_customer_profile_at, load_all)
from dataset_builder import (
    get_active_customers, 
    get_reactivation_candidates, 
    generate_snapshot_dates,
    _compute_personal_thresholds
    )
from feature_engineering import (compute_all_features, compute_market_features, compute_interaction_features)

logger = logging.getLogger(__name__)

PREDICT_DIR = OUTPUT_DIR / "predictions"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(name: str):
    cbm_path = MODEL_DIR / f"{name}.cbm"
    pkl_path = MODEL_DIR / f"{name}.pkl"

    if cbm_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(cbm_path))
        logger.info(f"Загружена модель CatBoost из {cbm_path}")
        return model
    elif pkl_path.exists():
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Загружена модель sklearn из {pkl_path}")
        return model
    else:
        raise FileNotFoundError(
            f"Модель не найдена для '{name}' в {MODEL_DIR}.\n"
            f"Сначала запустите main.py, чтобы обучить и сохранить модели"
        )

def load_calibrator(name: str):
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Калибратор не найден: {path}. "
            f"Запустите полный пайплайн, чтобы обучить калибраторы."
        )
    with open(path, "rb") as f:
        return pickle.load(f)



def score_customers_unified(
    timing_model,
    reactivation_model,
    snapshot_date: pd.Timestamp,
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    assets: pd.DataFrame,
    prices: pd.DataFrame,
    dormancy_threshold: int = DORMANCY_THRESHOLD_DAYS,
    reactivation_weight: float = 1.5,
    calibrator_timing=None,
    calibrator_reactivation=None,
    reactivation_business_weight: float = 1.5,
    calibration_method: str = "isotonic",
    min_tx=MIN_TX_REACT,
    active_window=ACTIVE_WINDOW_DAYS,
) -> pd.DataFrame:
    logger.info(f"Скоринг клиентов на дату: {snapshot_date.date()}")
    
    max_data_date = transactions["timestamp"].max()
    latest_safe = max_data_date - pd.Timedelta(days=HORIZON_DAYS)
    if snapshot_date > latest_safe:
        logger.warning(
            f"  snapshot_date {snapshot_date.date()} слишком поздняя — "
            f"горизонт предсказания выходит за пределы данных "
            f"(max safe: {latest_safe.date()}). Результаты могут быть некорректными."
        )

    use_calibration = (calibrator_timing is not None and calibrator_reactivation is not None)
    if use_calibration:
        logger.info("  Режим: откалиброванный скоринг")
    else:
        logger.warning("  Режим: сырой скоринг (калибраторы не переданы)")

    tx_history = transactions[transactions["timestamp"] <= snapshot_date].copy()
    
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

    all_candidates = days_since.index.intersection(all_recent)
    personal_thresholds_split = _compute_personal_thresholds(
        tx_history, all_candidates,
        multiplier=DORMANCY_MULTIPLIER,
        min_days=DORMANCY_MIN_DAYS,
    )
    days_since_all = days_since.reindex(all_candidates)
    dormant_mask = days_since_all > personal_thresholds_split.reindex(all_candidates)

    warm_ids    = all_candidates[~dormant_mask]
    dormant_ids = all_candidates[dormant_mask].intersection(eligible_ids)

    if len(dormant_ids) == 0:
        logger.warning(
            f"  Нет dormant клиентов для среза {snapshot_date.date()} — "
            f"скорим только warm сегмент"
        )
    logger.info(f"  Warm: {len(warm_ids):,}  |  Dormant: {len(dormant_ids):,}")

    results = []

    market_ctx = compute_market_features(prices, snapshot_date)

    if len(warm_ids) > 0:
        tx_w      = tx_history[tx_history["customerID"].isin(warm_ids)]
        profile_w = get_customer_profile_at(customers, snapshot_date)
        profile_w = profile_w[profile_w["customerID"].isin(warm_ids)]
        feat_w = compute_all_features(tx_w, profile_w, assets, prices, snapshot_date, mode="timing", market_ctx=market_ctx)

        avail      = [c for c in TIMING_ALL_FEATURES if c in feat_w.columns]
        raw_scores = timing_model.predict_proba(feat_w[avail])[:, 1]

        if use_calibration:
            if calibration_method == "isotonic":
                final_scores = calibrator_timing.transform(raw_scores)
            else:
                final_scores = calibrator_timing.predict_proba(raw_scores.reshape(-1,1))[:,1]
        else:
            final_scores = raw_scores

        results.append(pd.DataFrame({
            "customerID":       feat_w.index,
            "propensity_score": final_scores,
            "segment":          "warm",
            "days_since_last_buy": (
                feat_w["days_since_last_buy"]
                if "days_since_last_buy" in feat_w.columns
                else pd.Series(0, index=feat_w.index)
            ),
        }))

    if len(dormant_ids) > 0:
        tx_d      = tx_history[tx_history["customerID"].isin(dormant_ids)]
        profile_d = get_customer_profile_at(customers, snapshot_date)
        profile_d = profile_d[profile_d["customerID"].isin(dormant_ids)]
        feat_d = compute_all_features(tx_d, profile_d, assets, prices, snapshot_date, mode="reactivation", market_ctx=market_ctx)

        interaction = compute_interaction_features(feat_d, market_ctx)
        feat_d      = feat_d.join(interaction, how="left")

        personal_thresholds = _compute_personal_thresholds(
            tx_d, dormant_ids,
            multiplier=DORMANCY_MULTIPLIER,
            min_days=DORMANCY_MIN_DAYS,
        )
        feat_d["personal_dormancy_threshold"] = (
            personal_thresholds.reindex(feat_d.index).fillna(DORMANCY_MIN_DAYS)
        )

        avail      = [c for c in REACTIVATION_ALL_FEATURES if c in feat_d.columns]
        raw_scores = reactivation_model.predict_proba(feat_d[avail])[:, 1]
        logger.info(f"  Dormant raw scores: min={raw_scores.min():.4f}  max={raw_scores.max():.4f}  mean={raw_scores.mean():.4f}")

        if use_calibration:
            if calibration_method == "isotonic":
                cal_scores = calibrator_reactivation.transform(raw_scores)
            else:
                cal_scores = calibrator_reactivation.predict_proba(raw_scores.reshape(-1,1))[:,1]
            final_scores = np.clip(cal_scores * reactivation_business_weight, 0, 1)
        else:
            final_scores = raw_scores * reactivation_weight

        results.append(pd.DataFrame({
            "customerID":       feat_d.index,
            "propensity_score": final_scores,
            "segment":          "dormant",
            "days_since_last_buy": (
                feat_d["current_pause_days"]
                if "current_pause_days" in feat_d.columns
                else pd.Series(9999, index=feat_d.index)
            ),
        }))

    if not results:
        logger.warning("Нет клиентов для скоринга — проверьте дату среза и данные")
        return pd.DataFrame()

    result = pd.concat(results, ignore_index=True)
    result["snapshot_date"] = snapshot_date
    result["rank"] = result["propensity_score"].rank(ascending=False, method="first").astype(int)
    result = result.sort_values("rank").reset_index(drop=True)

    top20 = result.nsmallest(max(1, int(len(result) * 0.2)), "rank")
    dormant_in_top20 = top20["segment"].eq("dormant").mean()
    logger.info(f"  Доля dormant в топ-20%: {dormant_in_top20:.1%}")
    logger.info(f"  Скоры: min={result['propensity_score'].min():.4f}  max={result['propensity_score'].max():.4f}  mean={result['propensity_score'].mean():.4f}")

    return result[["customerID", "propensity_score", "segment", "days_since_last_buy", "rank", "snapshot_date"]]



def select_hot_customers(
    scored_df: pd.DataFrame,
    top_k_frac: float = 0.20,
    min_score: float = 0.05,
) -> pd.DataFrame:
    n_total = len(scored_df)
    k_cutoff = max(1, int(n_total * top_k_frac))

    hot = (
        scored_df
        .sort_values("rank")
        .head(k_cutoff)
        .loc[lambda df: df["propensity_score"] >= min_score]
        .reset_index(drop=True)
    )

    logger.info(
        f"Отобрано клиентов для контакта: {len(hot):,} "
        f"(top {top_k_frac:.0%} среди {n_total:,}, min_score={min_score})\n"
        f"  Warm    : {(hot['segment'] == 'warm').sum():,}\n"
        f"  Dormant : {(hot['segment'] == 'dormant').sum():,}"
    )

    return hot



def score_stability_report(scored_t0: pd.DataFrame, scored_t1: pd.DataFrame):
    t0 = scored_t0[["customerID", "propensity_score", "rank"]].rename(
        columns={"propensity_score": "score_t0", "rank": "rank_t0"}
    )
    t1 = scored_t1[["customerID", "propensity_score", "rank"]].rename(
        columns={"propensity_score": "score_t1", "rank": "rank_t1"}
    )
    merged = t0.merge(t1, on="customerID", how="inner")
    merged["score_delta"] = merged["score_t1"] - merged["score_t0"]
    merged["rank_delta"] = merged["rank_t1"]  - merged["rank_t0"]

    logger.info("Стабильность скоринга (t0 → t1):")
    logger.info(f"  Клиенты в обоих срезах: {len(merged):,}")
    logger.info(f"  Mean |score_delta|: {merged['score_delta'].abs().mean():.4f}")
    logger.info(f"  Mean |rank_delta|: {merged['rank_delta'].abs().mean():.1f}")

    return merged.sort_values("rank_delta", ascending=False).reset_index(drop=True)



def run_inference(
    timing_model_name: str = "catboost_timing",
    reactivation_model_name: str = "catboost_reactivation",
    snapshot_date: pd.Timestamp | None = None,
    top_k_frac: float = 0.20,
    min_score: float = 0.05,
    reactivation_weight: float = 1.5,
    reactivation_business_weight: float = 1.5,
    save: bool = True,
):
    timing_model = load_model(timing_model_name)
    reactivation_model = load_model(reactivation_model_name)

    try:
        cal_timing = load_calibrator("calibrator_timing")
        cal_react  = load_calibrator("calibrator_reactivation")
        logger.info("Калибраторы загружены")
    except FileNotFoundError as e:
        logger.warning(f"{e} — скоринг без калибровки")
        cal_timing = cal_react = None

    logger.info("Загрузка данных...")
    data = load_all()

    if snapshot_date is None:
        snapshot_date = data["transactions"]["timestamp"].max()
        logger.info(
            f"Дата среза не указана — используем последнюю: {snapshot_date.date()}"
        )
    snapshot_date = pd.Timestamp(snapshot_date)

    scored = score_customers_unified(
        timing_model=timing_model,
        reactivation_model=reactivation_model,
        snapshot_date=snapshot_date,
        transactions=data["transactions"],
        customers=data["customers"],
        assets=data["assets"],
        prices=data["prices"],
        reactivation_weight=reactivation_weight,
        calibrator_timing=cal_timing,
        calibrator_reactivation=cal_react,
        reactivation_business_weight=reactivation_business_weight,

    )

    if scored.empty:
        logger.warning("Клиенты не проскорены — проверьте данные и дату среза")
        return pd.DataFrame()

    hot = select_hot_customers(scored, top_k_frac=top_k_frac, min_score=min_score)

    if save:
        date_str = snapshot_date.strftime("%Y%m%d")

        hot_path = PREDICT_DIR / f"hot_customers_{date_str}.csv"
        hot.to_csv(hot_path, index=False)
        logger.info(f"Список клиентов для контакта сохранен: {hot_path}")

        full_path = PREDICT_DIR / f"all_scores_{date_str}.csv"
        scored.to_csv(full_path, index=False)
        logger.info(f"Полный список со скорами сохранен: {full_path}")

    return hot