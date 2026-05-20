import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    TEST_MONTHS, 
    VALID_MONTHS,
    INTERACTION_WINDOW_DAYS,
    MIN_ITEM_SUPPORT,
    OUTPUT_DIR,
    ALS_BEST_CONFIDENCE_ALPHA,
    ALS_BEST_DECAY_LAMBDA,
    ALS_BEST_FUSION_ALPHA
)
from data_loader import (
    load_transactions,
    load_customers,
    load_assets,
    build_item_index,
    get_buy_history,
    get_hot_customers,
)
from matrix_builder import build_interaction_matrix
from train import (
    run_training,
    load_als_model,
    run_ease_training,
    load_ease_model,
    build_loo_pairs,
    build_tx_without_loo_pairs,
    train_als_best,
)
from evaluate import run_evaluation, plot_grid_search_results
from predict import run_prediction


# Логирование
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)



# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Recommender Model — FAR-Trans")

    # Дата среза — теперь опциональная, как в propensity
    parser.add_argument(
        "--snapshot-date", type=str, default=None,
        help="Дата среза YYYY-MM-DD. По умолчанию — последняя дата в транзакциях."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train", action="store_true", default=False,
        help="Обучить EASE модель (быстро, без grid search ALS).",
    )
    mode.add_argument(
        "--predict-only", action="store_true", default=False,
        help="Только inference — загрузить модель и сделать рекомендации.",
    )
    mode.add_argument(
        "--eval-only", action="store_true", default=False,
        help="Оценка: обучить ALS baseline + загрузить EASE, сравнить метрики.",
    )
    mode.add_argument(
        "--tune", action="store_true", default=False,
        help="Grid search: ALS (135 комб.) + EASE (5 комб.). Результаты → config.py.",
    )

    parser.add_argument(
        "--with-baseline", action="store_true", default=False,
        help="При --train: обучить ALS baseline в дополнение к EASE.",
    )
    parser.add_argument(
        "--eval-stage", type=str, choices=["validation", "test"], default="validation",
    )
    parser.add_argument("--no-save", action="store_true", default=False)
    return parser.parse_args()



def _weights_file(snapshot_date: pd.Timestamp) -> Path:
    return (Path(__file__).resolve().parent / "models" /
            f"rec_weights_{snapshot_date.strftime('%Y%m%d')}.json")


def _build_imat(
    tx, assets, snapshot_date,
    confidence_alpha=ALS_BEST_CONFIDENCE_ALPHA,
    decay_lambda=ALS_BEST_DECAY_LAMBDA,
):
    tx_window = get_buy_history(tx, snapshot_date, INTERACTION_WINDOW_DAYS)
    item_meta = build_item_index(tx_window, assets, MIN_ITEM_SUPPORT)
    imat = build_interaction_matrix(
        tx_window        = tx_window,
        item_meta        = item_meta,
        snapshot_date    = snapshot_date,
        confidence_alpha = confidence_alpha,
        decay_lambda     = decay_lambda,
    )
    return imat, tx_window


def _load_model_with_imat(tx, assets, snapshot_date):
    model, maps, weights = load_als_model(snapshot_date)
    imat, _ = _build_imat(
        tx, assets, snapshot_date,
        confidence_alpha = weights["confidence_alpha"],
        decay_lambda     = weights["decay_lambda"],
    )
    imat.user_id_to_idx = maps["user_id_to_idx"]
    imat.item_id_to_idx = maps["item_id_to_idx"]
    imat.idx_to_user_id = maps["idx_to_user_id"]
    imat.idx_to_item_id = maps["idx_to_item_id"]
    return model, imat, weights


def _get_val_tx(tx, snapshot_date):
    max_date  = tx["timestamp"].max()
    valid_end = max_date - pd.DateOffset(months=TEST_MONTHS)
    train_end = valid_end - pd.DateOffset(months=VALID_MONTHS)
    valid_end = valid_end.replace(day=1)
    train_end = train_end.replace(day=1)

    if snapshot_date <= valid_end:
        return tx[
            (tx["timestamp"] >= train_end) &
            (tx["timestamp"] <  valid_end)
        ].copy()
    else:
        val_start = snapshot_date - pd.Timedelta(days=90)
        logger.info(
            f"  Validation окно (скользящее): "
            f"[{val_start.date()}, {snapshot_date.date()})"
        )
        return tx[
            (tx["timestamp"] >= val_start) &
            (tx["timestamp"] <  snapshot_date)
        ].copy()



def _get_test_tx(tx):
    max_date  = tx["timestamp"].max()
    valid_end = max_date - pd.DateOffset(months=TEST_MONTHS)
    valid_end = valid_end.replace(day=1)
    return tx[tx["timestamp"] >= valid_end].copy()


def main():
    args = parse_args()
    save = not args.no_save

    tx        = load_transactions()
    customers = load_customers()
    assets    = load_assets()

    if args.snapshot_date is None:
        snapshot_date = tx["timestamp"].max()
        logger.info(f"  snapshot_date не задан, используем последнюю дату: {snapshot_date.date()}")
    else:
        snapshot_date = pd.Timestamp(args.snapshot_date)

    logger.info("═" * 60)
    logger.info("  RECOMMENDER PIPELINE")
    logger.info(f"  snapshot : {snapshot_date.date()}")
    logger.info("═" * 60)

    # --tune: grid search ALS + EASE
    if args.tune:
    
        logger.info("  Режим: --tune (grid search ALS + EASE)")
        tune_snapshot = snapshot_date - pd.Timedelta(days=90)
        imat_init, tx_window = _build_imat(tx, assets, tune_snapshot)
        tx_val = _get_val_tx(tx, snapshot_date)
        _, imat_final, weights = run_training(imat_init, tx_window, tx_val, tune_snapshot, assets)
        plot_grid_search_results(weights)
        run_ease_training(imat_final, tx_window, tx_val, tune_snapshot, assets)
        logger.info("  Зафиксируй лучшие параметры в config.py")
        return

    # --eval-only: ALS baseline + EASE, сравнение метрик
    if args.eval_only:
        logger.info("  Режим: --eval-only (ALS baseline + EASE)")

        eval_snapshot = snapshot_date - pd.Timedelta(days=90)
        imat_init, tx_window = _build_imat(tx, assets, eval_snapshot,
                                        confidence_alpha=ALS_BEST_CONFIDENCE_ALPHA,
                                        decay_lambda=ALS_BEST_DECAY_LAMBDA)

        tx_val  = _get_val_tx(tx, snapshot_date)
        tx_eval = _get_test_tx(tx) if args.eval_stage == "test" else tx_val

        # ALS baseline — без grid search, с лучшими параметрами из config
        als_model = train_als_best(imat_init)
        weights = {
            "confidence_alpha": ALS_BEST_CONFIDENCE_ALPHA,
            "decay_lambda":     ALS_BEST_DECAY_LAMBDA,
            "fusion_alpha":     ALS_BEST_FUSION_ALPHA,
        }

        try:
            ease_model = load_ease_model(eval_snapshot)
        except FileNotFoundError:
            logger.info("  EASE не найдена — обучаем с лучшими параметрами из config")
            ease_model, _, _ = run_ease_training(
                imat_init, tx_window, tx_val, snapshot_date, assets
            )

        loo_pairs = build_loo_pairs(imat_init, tx_val)
        tx_clean  = build_tx_without_loo_pairs(tx_window, loo_pairs)
        item_meta_clean = build_item_index(tx_clean, assets)
        imat_clean = build_interaction_matrix(
            tx_window        = tx_clean,
            item_meta        = item_meta_clean,
            snapshot_date    = eval_snapshot,
            confidence_alpha = ALS_BEST_CONFIDENCE_ALPHA,
            decay_lambda     = ALS_BEST_DECAY_LAMBDA,
        )

        try:
            ease_model = load_ease_model(eval_snapshot)
        except FileNotFoundError:
            logger.info("  EASE не найдена — обучаем с лучшими параметрами из config")
            ease_model, _, _ = run_ease_training(
                imat_init, tx_window, tx_val, eval_snapshot, assets
            )

        hot_customers   = get_hot_customers(snapshot_date)
    
        recommendations = run_prediction(
            hot_customers, als_model, imat_clean, weights,  # ← imat_clean
            tx, customers, snapshot_date, save=False, ease_model=ease_model,
        )
        run_evaluation(
            als_model, imat_init, weights, tx_eval,
            recommendations, stage=args.eval_stage, ease_model=ease_model,
            imat_ease=imat_clean
        )
        return

    # --train: обучить EASE (+ опционально ALS)
    if args.train:
        logger.info("  Режим: --train (EASE)")
    
        train_snapshot = snapshot_date - pd.Timedelta(days=90)
        imat_init, tx_window = _build_imat(tx, assets, train_snapshot,
                                            confidence_alpha=ALS_BEST_CONFIDENCE_ALPHA,
                                            decay_lambda=ALS_BEST_DECAY_LAMBDA)
        tx_val = _get_val_tx(tx, snapshot_date)
        run_ease_training(imat_init, tx_window, tx_val, train_snapshot, assets)
        
        if args.with_baseline:
            logger.info("  --with-baseline: обучаем ALS")
            train_als_best(imat_init)
        return


    # Default / --predict-only: inference
    logger.info("  Режим: inference (загружаем EASE модель)")
    train_snapshot = snapshot_date - pd.Timedelta(days=90)
    try:
        ease_model = load_ease_model(train_snapshot)
    except FileNotFoundError:
        raise RuntimeError(
            f"EASE модель для {train_snapshot.date()} не найдена. "
            f"Запустите: python main.py --snapshot-date {snapshot_date.date()} --train"
        )

    imat, _ = _build_imat(
        tx, assets, train_snapshot,
        confidence_alpha=ALS_BEST_CONFIDENCE_ALPHA,
        decay_lambda=ALS_BEST_DECAY_LAMBDA,
    )

    # ALS — только как fallback, грузим если есть
    try:
        als_model, maps, weights = load_als_model(train_snapshot)
        imat.user_id_to_idx = maps["user_id_to_idx"]
        imat.item_id_to_idx = maps["item_id_to_idx"]
        imat.idx_to_user_id = maps["idx_to_user_id"]
        imat.idx_to_item_id = maps["idx_to_item_id"]
    except FileNotFoundError:
        als_model = None
        weights   = {
            "confidence_alpha": ALS_BEST_CONFIDENCE_ALPHA,
            "decay_lambda":     ALS_BEST_DECAY_LAMBDA,
            "fusion_alpha":     ALS_BEST_FUSION_ALPHA,
        }
        logger.info("  ALS модель не найдена — inference только через EASE")

    hot_customers   = get_hot_customers(snapshot_date)
    recommendations = run_prediction(
        hot_customers, als_model, imat, weights,
        tx, customers, snapshot_date, save=save, ease_model=ease_model,
    )


if __name__ == "__main__":
    main()
