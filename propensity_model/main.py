import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Импорт проектов
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import OUTPUT_DIR#, TRAIN_END, VALID_END
from data_loader import load_all
from dataset_builder import (
    build_dataset,
    build_reactivation_dataset,
    time_split,
    get_X_y,
)
from train import run_training, load_model, tune_timing, tune_reactivation, tune_logreg, tune_logreg_reactivation
from evaluate import run_evaluation
from predict import (
    score_customers_unified,
    select_hot_customers,
)


# Настройка логирования
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="a"),
        ],
    )

logger = logging.getLogger(__name__)


DATASET_CACHE = OUTPUT_DIR / "dataset.parquet"
REACT_DATASET_CACHE = OUTPUT_DIR / "reactivation_dataset.parquet"

def save_dataset(df):
    df.to_parquet(DATASET_CACHE, index=False)
    logger.info(f"Датасет кэширован: {DATASET_CACHE}")

def load_dataset_cache():
    logger.info(f"Загрузка кэшированного датасета из {DATASET_CACHE}")
    return pd.read_parquet(DATASET_CACHE)

def save_react_dataset(df):
    df.to_parquet(REACT_DATASET_CACHE, index=False)
    logger.info(f"Reactivation датасет кэширован: {REACT_DATASET_CACHE}")

def load_react_dataset_cache():
    logger.info(f"Загрузка reactivation датасета из {REACT_DATASET_CACHE}")
    return pd.read_parquet(REACT_DATASET_CACHE)



# Запуск этапов
def stage_load_data():
    logger.info("━" * 60)
    logger.info("  ЭТАП 1 — Загрузка данных")
    logger.info("━" * 60)
    t0 = time.time()
    data = load_all()
    logger.info(f"  Выполнено за {time.time() - t0:.1f}с")
    return data


def stage_build_dataset(data, skip_build):
    logger.info("━" * 60)
    logger.info("  ЭТАП 2 — Сборка датасетов (timing + reactivation)")
    logger.info("━" * 60)
    t0 = time.time()

    # Timing датасет
    if skip_build and DATASET_CACHE.exists():
        dataset = load_dataset_cache()
        logger.info(f"  Timing: загружено из кэша: {len(dataset):,} строк")
    else:
        dataset = build_dataset(
            transactions=data["transactions"],
            customers=data["customers"],
            assets=data["assets"],
            prices=data["prices"],
        )
        save_dataset(dataset)

    _log_dataset_summary(dataset, name="Timing")

    # Reactivation датасет
    if skip_build and REACT_DATASET_CACHE.exists():
        react_dataset = load_react_dataset_cache()
        logger.info(
            f"  Reactivation: загружено из кэша: {len(react_dataset):,} строк"
        )
    else:
        react_dataset = build_reactivation_dataset(
            transactions=data["transactions"],
            customers=data["customers"],
            assets=data["assets"],
            prices=data["prices"],
        )
        save_react_dataset(react_dataset)

    _log_dataset_summary(react_dataset, name="Reactivation")

    logger.info(f"  Выполнено за {time.time() - t0:.1f}с")
    return dataset, react_dataset


def stage_split(dataset, react_dataset, data):
    logger.info("━" * 60)
    logger.info("  ЭТАП 3 — Train / Valid / Test Split (timing + reactivation)")
    logger.info("━" * 60)
    #train_df, valid_df, test_df = time_split(dataset)
    #train_react_df, valid_react_df, test_react_df = time_split(react_dataset)

    train_df, valid_df, test_df = time_split(dataset, transactions=data["transactions"])
    train_react_df, valid_react_df, test_react_df = time_split(
        react_dataset, transactions=data["transactions"]
    )
    logger.info(
        f"  Reactivation — Train: {len(train_react_df):,}  "
        f"Valid: {len(valid_react_df):,}  "
        f"Test: {len(test_react_df):,}"
    )

    return train_df, valid_df, test_df, train_react_df, valid_react_df, test_react_df


def stage_train(train_df, valid_df, train_react_df, valid_react_df, with_baseline=False):
    logger.info("━" * 60)
    logger.info("  ЭТАП 4 — Обучение моделей (timing + reactivation)")
    logger.info("━" * 60)
    t0 = time.time()
    models = run_training(
        train_df, valid_df,
        train_react_df, valid_react_df,
        with_baseline=with_baseline,
    )
    logger.info(f"  Выполнено за {time.time() - t0:.1f}с")
    return models


def stage_evaluate(models, test_df, test_react_df):
    logger.info("━" * 60)
    logger.info("  ЭТАП 5 — Оценка моделей (timing + reactivation)")
    logger.info("━" * 60)
    t0 = time.time()
    metrics = run_evaluation(
        models, test_df,
        test_react_df=test_react_df,
    )
    logger.info(f"  Выполнено за {time.time() - t0:.1f}с")
    return metrics


def stage_predict(models, data, snapshot_date, top_k_frac=0.20):
    logger.info("━" * 60)
    logger.info("  ЭТАП 6 — Инференс (unified scoring)")
    logger.info("━" * 60)
    t0 = time.time()

    if snapshot_date is None:
        snapshot_date = data["transactions"]["timestamp"].max()
        logger.info(f"  Используется последняя дата среза: {snapshot_date.date()}")
    
    from predict import load_calibrator
    cal_timing = models.get("cal_timing")
    cal_react  = models.get("cal_react")
    if cal_timing is None:
        try:
            cal_timing = load_calibrator("calibrator_timing")
            cal_react  = load_calibrator("calibrator_reactivation")
        except FileNotFoundError:
            logger.warning("Калибраторы не найдены — скоринг без калибровки")

    scored = score_customers_unified(
        timing_model=models["timing"],
        reactivation_model=models["reactivation"],
        snapshot_date=snapshot_date,
        transactions=data["transactions"],
        customers=data["customers"],
        assets=data["assets"],
        prices=data["prices"],
        calibrator_timing=cal_timing,
        calibrator_reactivation=cal_react,
        reactivation_business_weight=1.5,
    )

    hot = select_hot_customers(scored, top_k_frac=top_k_frac)

    date_str = snapshot_date.strftime("%Y%m%d")
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    hot_path = pred_dir / f"hot_customers_{date_str}.csv"
    full_path = pred_dir / f"all_scores_{date_str}.csv"
    hot.to_csv(hot_path, index=False)
    scored.to_csv(full_path, index=False)

    logger.info(f"  Клиенты для контакта: {hot_path}")
    logger.info(f"  Полный список скоров: {full_path}")
    logger.info(f"  Выполнено за {time.time() - t0:.1f}с")

    return hot



def _log_dataset_summary(dataset, name=""):
    from config import TARGET_COL
    label = f" [{name}]" if name else ""
    logger.info(
        f"\n  Саммари по датасету{label}:\n"
        f"    Строк : {len(dataset):,}\n"
        f"    Уникальных клиентов : {dataset['customerID'].nunique():,}\n"
        f"    Срезов : {dataset['snapshot_date'].nunique()}\n"
        f"    Доля Label=1 : {dataset[TARGET_COL].mean():.2%}\n"
        f"    Диапазон дат : {dataset['snapshot_date'].min().date()} → "
        f"{dataset['snapshot_date'].max().date()}"
    )


def _print_final_summary(metrics, hot):
    logger.info("\n" + "═" * 60)
    logger.info("  ПАЙПЛАЙН ПОЛНОСТЬЮ ВЫПОЛНЕН")
    logger.info("═" * 60)

    # Timing метрики
    if "timing" in metrics:
        m = metrics["timing"]
        logger.info(f"\n  Timing модель (CatBoost) на тесте:")
        logger.info(f"    ROC-AUC         : {m['roc_auc']:.4f}")
        logger.info(f"    PR-AUC          : {m['pr_auc']:.4f}")
        logger.info(f"    Recall@top20%   : {m.get('recall@top20pct', '—')}")
        logger.info(f"    Lift@top20%     : {m.get('lift@top20pct', '—'):.2f}x")

    # Reactivation метрики
    if "reactivation" in metrics:
        m = metrics["reactivation"]
        logger.info(f"\n  Reactivation модель на тесте:")
        logger.info(f"    ROC-AUC         : {m['roc_auc']:.4f}")
        logger.info(f"    PR-AUC          : {m['pr_auc']:.4f}")
        logger.info(f"    Recall@top20%   : {m.get('recall@top20pct', '—')}")
        logger.info(f"    Lift@top20%     : {m.get('lift@top20pct', '—'):.2f}x")

    if not hot.empty:
        dormant_in_top = (hot["segment"] == "dormant").sum()
        dormant_share = dormant_in_top / len(hot)
        logger.info(
            f"\n  Список клиентов для контакта:\n"
            f"    Всего    : {len(hot):,}\n"
            f"    Warm     : {(hot['segment'] == 'warm').sum():,}\n"
            f"    Dormant  : {dormant_in_top:,} ({dormant_share:.1%})\n"
            f"    Средний скор : {hot['propensity_score'].mean():.4f}"
        )
        # Ключевая бизнес-метрика — доля спящих в топе
        if dormant_share < 0.15:
            logger.warning(
                f"  ⚠ Dormant доля в топе низкая ({dormant_share:.1%}) — "
                f"reactivation модель может требовать настройки reactivation_weight"
            )

    logger.info(f"\n  Все результаты сохранены в: {OUTPUT_DIR.resolve()}")
    logger.info("═" * 60)



# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Propensity Model pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train",
        action="store_true",
        help="Обучить модели (полный пайплайн: build → train → eval).",
    )
    mode.add_argument(
        "--eval-only",
        action="store_true",
        help="Load saved models and re-run evaluation only.",
    )
    mode.add_argument(
        "--predict-only",
        action="store_true",
        help="Load saved model and run inference only.",
    )
    mode.add_argument(
    "--tune",
    action="store_true",
    help=(
        "Run grid search for CatBoost hyperparameters (timing + reactivation). "
        "Results saved to outputs/grid_search_*.json. "
        "After running: manually update CATBOOST_PARAMS in config.py."
        ),
    )

    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dataset building if dataset.parquet already exists.",
    )
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        default=False,
        help="Обучить LogReg baseline в дополнение к CatBoost (для --eval-only).",
    )
    parser.add_argument(
        "--eval-stage", type=str,
        choices=["validation", "test"], default="test",
        help="Датасет для --eval-only (default: test).",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default=None,
        help="Snapshot date for inference (YYYY-MM-DD). Defaults to last date in data.",
    )
    parser.add_argument(
        "--top-k",
        type=float,
        default=0.20,
        help="Fraction of customers to select as 'hot' (default: 0.20).",
    )
    parser.add_argument(
        "--timing-model",
        type=str,
        default="catboost_timing",
        help="Timing model name (default: catboost_timing).",
    )
    parser.add_argument(
        "--reactivation-model",
        type=str,
        default="catboost_reactivation",
        help="Reactivation model name (default: catboost_reactivation).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )

    return parser.parse_args()


def stage_tune(train_df, valid_df, train_react_df, valid_react_df):
    logger.info("━" * 60)
    logger.info("  ЭТАП — Grid Search (timing + reactivation CatBoost)")
    logger.info("  Logreg-бейзлайны не тюнятся — используют фиксированные LOGREG_*_PARAMS")
    logger.info("━" * 60)
    t0 = time.time()

    best_timing      = tune_timing(train_df, valid_df)
    best_reactivation = tune_reactivation(train_react_df, valid_react_df)
    best_logreg       = tune_logreg(train_df, valid_df)
    best_logreg_react = tune_logreg_reactivation(train_react_df, valid_react_df)


    logger.info("\n" + "═" * 60)
    logger.info("  Зафиксируй в config.py (с пометкой '# подобрано grid search'):")
    logger.info(f"  CATBOOST_PARAMS:              depth={best_timing['depth']}, "
                f"lr={best_timing['lr']}, "
                f"best_iter≈{best_timing['best_iter']}")
    logger.info(f"  REACTIVATION_CATBOOST_PARAMS: depth={best_reactivation['depth']}, "
                f"lr={best_reactivation['lr']}, "
                f"best_iter≈{best_reactivation['best_iter']}")
    logger.info(f"  LOGREG_PARAMS:                C={best_logreg['C']}")
    logger.info(f"  LOGREG_REACT_PARAMS:          C={best_logreg_react['C']}")
    logger.info("═" * 60)
    logger.info(f"  Grid search выполнен за {time.time() - t0:.1f}с")

    return best_timing, best_reactivation


# Main
def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    snapshot_date = pd.Timestamp(args.snapshot_date) if args.snapshot_date else None
    total_t0 = time.time()

    # --train: полное обучение
    if args.train:
        logger.info("Mode: TRAIN")

        data = stage_load_data()
        dataset, react_dataset = stage_build_dataset(data, skip_build=args.skip_build)
        train_df, valid_df, test_df, \
        train_react_df, valid_react_df, test_react_df = stage_split(
            dataset, react_dataset, data
        )
        models = stage_train(
            train_df, valid_df, train_react_df, valid_react_df,
            with_baseline=args.with_baseline,
        )
        metrics = stage_evaluate(models, test_df, test_react_df)
        _print_final_summary(metrics, pd.DataFrame())
        logger.info(f"\nTotal time: {time.time() - total_t0:.1f}s")
        return

    # --eval-only: только оценка
    if args.eval_only:
        logger.info("Mode: EVALUATE ONLY")

        data = stage_load_data()
        dataset, react_dataset = stage_build_dataset(data, skip_build=True)
        _, valid_df, test_df, \
        _, valid_react_df, test_react_df = stage_split(dataset, react_dataset, data)

        eval_df       = valid_df       if args.eval_stage == "validation" else test_df
        eval_react_df = valid_react_df if args.eval_stage == "validation" else test_react_df

        timing = load_model(args.timing_model)
        react  = load_model(args.reactivation_model)
        models = {"timing": timing, "reactivation": react}

        if args.with_baseline:
            models["logreg"]       = load_model("logreg_baseline")
            models["logreg_react"] = load_model("logreg_reactivation_baseline")

        metrics = stage_evaluate(models, eval_df, eval_react_df)
        logger.info(f"\nTotal time: {time.time() - total_t0:.1f}s")
        return

    # --tune: grid search
    if args.tune:
        logger.info("Mode: TUNE (grid search)")

        data = stage_load_data()
        dataset, react_dataset = stage_build_dataset(data, skip_build=True)
        train_df, valid_df, _, train_react_df, valid_react_df, _ = stage_split(
            dataset, react_dataset, data
        )
        stage_tune(train_df, valid_df, train_react_df, valid_react_df)
        logger.info(f"\nTotal time: {time.time() - total_t0:.1f}s")
        return

    # Default / --predict-only: inference
    logger.info("Mode: PREDICT ONLY")

    data = stage_load_data()
    timing = load_model(args.timing_model)
    react  = load_model(args.reactivation_model)
    models = {"timing": timing, "reactivation": react}

    hot = stage_predict(
        models=models,
        data=data,
        snapshot_date=snapshot_date,
        top_k_frac=args.top_k,
    )
    logger.info(f"\nTotal time: {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
