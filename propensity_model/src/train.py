import json
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

from config import (
    TIMING_CAT_FEATURES,
    TIMING_NUMERIC_FEATURES,
    REACTIVATION_CAT_FEATURES,
    REACTIVATION_NUMERIC_FEATURES,
    CATBOOST_PARAMS,
    REACTIVATION_CATBOOST_PARAMS,
    LOGREG_PARAMS,
    LOGREG_REACT_PARAMS,
    MODEL_DIR,
    OUTPUT_DIR,
    TARGET_COL,
)
from dataset_builder import get_X_y

logger = logging.getLogger(__name__)



def tune_logreg(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    """
    Grid search по C для LogReg Timing baseline.
    Запускается один раз через python main.py --tune.
    Результат фиксируется в LOGREG_PARAMS в config.py.
    """
    logger.info("═" * 60)
    logger.info("  GRID SEARCH — LogReg Timing Baseline")
    logger.info("═" * 60)

    X_train, y_train = get_X_y(train_df, model_type="timing")
    X_valid, y_valid = get_X_y(valid_df, model_type="timing")

    results = []
    for C in [0.01, 0.1, 1, 10]:
        model = build_logreg_pipeline(C=C)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
        results.append({"C": C, "val_auc": round(auc, 5)})
        logger.info(f"  C={C:<6} → AUC={auc:.4f}")

    best = max(results, key=lambda x: x["val_auc"])
    logger.info(f"\n  Лучший C (timing logreg): {best}")

    out_path = OUTPUT_DIR / "grid_search_logreg_timing.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    logger.info(f"  Результаты сохранены: {out_path}")

    return best


def tune_logreg_reactivation(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    """
    Grid search по C для LogReg Reactivation baseline.
    """
    logger.info("═" * 60)
    logger.info("  GRID SEARCH — LogReg Reactivation Baseline")
    logger.info("═" * 60)

    X_train, y_train = get_X_y(train_df, model_type="reactivation")
    X_valid, y_valid = get_X_y(valid_df, model_type="reactivation")

    results = []
    for C in [0.01, 0.1, 1, 10]:
        model = build_logreg_react_pipeline(C=C)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
        results.append({"C": C, "val_auc": round(auc, 5)})
        logger.info(f"  C={C:<6} → AUC={auc:.4f}")

    best = max(results, key=lambda x: x["val_auc"])
    logger.info(f"\n  Лучший C (reactivation logreg): {best}")

    out_path = OUTPUT_DIR / "grid_search_logreg_reactivation.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    logger.info(f"  Результаты сохранены: {out_path}")

    return best


# Logistic Regression (бейзлайн)
def build_logreg_pipeline(C: float = LOGREG_PARAMS["C"]):
 
    preprocessor = ColumnTransformer(
        transformers=[
                    ("num", StandardScaler(), TIMING_NUMERIC_FEATURES),
                    ("cat", OneHotEncoder(handle_unknown="ignore",
                                        sparse_output=False), TIMING_CAT_FEATURES),
                    ],remainder="drop"
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**{**LOGREG_PARAMS, "C": C})),
            ]
    )
    return pipeline


def train_logreg(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    logger.info("─────────────── Обучение логистической регрессии ───────────────")
    X_train, y_train = get_X_y(train_df, model_type="timing")
    X_valid, y_valid = get_X_y(valid_df, model_type="timing")

    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"найдены NaN в признаках перед обучением LogReg: {nan_cols}")
    
    model = build_logreg_pipeline()
    model.fit(X_train, y_train)
    
    val_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    logger.info(f"  Validation ROC-AUC: {val_auc:.4f}")

    return model


def build_logreg_react_pipeline(C: float = LOGREG_REACT_PARAMS["C"]):
    numeric_steps = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_steps,                                        REACTIVATION_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), REACTIVATION_CAT_FEATURES),
        ], remainder="drop"
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**{**LOGREG_REACT_PARAMS, "C": C})),
        ]
    )
    return pipeline


def train_logreg_reactivation(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    logger.info("─────────────── Обучение LogReg Reactivation (baseline) ───────────────")
    X_train, y_train = get_X_y(train_df, model_type="reactivation")
    X_valid, y_valid = get_X_y(valid_df, model_type="reactivation")

    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"найдены NaN в признаках перед обучением LogReg Reactivation: {nan_cols}")

    model = build_logreg_react_pipeline()
    model.fit(X_train, y_train)

    val_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    logger.info(f"  Validation ROC-AUC: {val_auc:.4f}")
    logger.info(f"  Positive rate (valid): {y_valid.mean():.2%}")

    return model



def tune_timing(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    logger.info("═" * 60)
    logger.info("  GRID SEARCH — CatBoost Timing")
    logger.info("═" * 60)

    X_train, y_train = get_X_y(train_df, model_type="timing")
    X_valid, y_valid = get_X_y(valid_df, model_type="timing")

    train_pool = Pool(X_train, y_train, cat_features=TIMING_CAT_FEATURES)
    valid_pool = Pool(X_valid, y_valid, cat_features=TIMING_CAT_FEATURES)

    grid = {
        "depth":         [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
    }

    results = []
    total = len(grid["depth"]) * len(grid["learning_rate"])
    idx = 0

    for depth in grid["depth"]:
        for lr in grid["learning_rate"]:
            idx += 1
            params = {
                **CATBOOST_PARAMS,
                "depth": depth,
                "learning_rate": lr,
                "verbose": 0,
            }
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

            result = {
                "depth": depth, "lr": lr,
                "best_iter": model.best_iteration_,
                "val_auc": round(auc, 5),
            }
            results.append(result)
            logger.info(
                f"  [{idx:>2}/{total}] depth={depth} lr={lr}"
                f" → AUC={auc:.4f}  iter={model.best_iteration_}"
            )

    best = max(results, key=lambda x: x["val_auc"])
    logger.info(f"\n  Лучшие параметры timing: {best}")

    out_path = OUTPUT_DIR / "grid_search_timing.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    logger.info(f"  Результаты сохранены: {out_path}")

    return best


def tune_reactivation(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    logger.info("═" * 60)
    logger.info("  GRID SEARCH — CatBoost Reactivation")
    logger.info("═" * 60)

    X_train, y_train = get_X_y(train_df, model_type="reactivation")
    X_valid, y_valid = get_X_y(valid_df, model_type="reactivation")

    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"  NaN в признаках reactivation перед grid search: {nan_cols}")

    train_pool = Pool(X_train, y_train, cat_features=REACTIVATION_CAT_FEATURES)
    valid_pool = Pool(X_valid, y_valid, cat_features=REACTIVATION_CAT_FEATURES)

    grid = {
        "depth":         [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
    }

    results = []
    total = len(grid["depth"]) * len(grid["learning_rate"])
    idx = 0

    for depth in grid["depth"]:
        for lr in grid["learning_rate"]:
            idx += 1
            params = {
                **REACTIVATION_CATBOOST_PARAMS,
                "depth": depth,
                "learning_rate": lr,
                "verbose": 0,
            }
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

            result = {
                "depth": depth, "lr": lr,
                "best_iter": model.best_iteration_,
                "val_auc": round(auc, 5),
            }
            results.append(result)
            logger.info(
                f"  [{idx:>2}/{total}] depth={depth} lr={lr}"
                f" → AUC={auc:.4f}  iter={model.best_iteration_}"
            )

    best = max(results, key=lambda x: x["val_auc"])
    logger.info(f"\n  Лучшие параметры reactivation: {best}")

    out_path = OUTPUT_DIR / "grid_search_reactivation.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    logger.info(f"  Результаты сохранены: {out_path}")

    return best


# CatBoost
def train_timing(train_df, valid_df):
    logger.info("─────────────── Обучение CatBoost Timing ───────────────")
    X_train, y_train = get_X_y(train_df, model_type="timing")
    X_valid, y_valid = get_X_y(valid_df, model_type="timing")

    train_pool = Pool(X_train, y_train, cat_features=TIMING_CAT_FEATURES)
    valid_pool = Pool(X_valid, y_valid, cat_features=TIMING_CAT_FEATURES)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    val_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    logger.info(f"  Best iteration  : {model.best_iteration_}")
    logger.info(f"  Validation ROC-AUC: {val_auc:.4f}")

    return model


def train_reactivation(train_df, valid_df):
    logger.info("─────────────── Обучение CatBoost Reactivation ───────────────")
    X_train, y_train = get_X_y(train_df, model_type="reactivation")
    X_valid, y_valid = get_X_y(valid_df, model_type="reactivation")

    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"NaN в признаках reactivation: {nan_cols}")

    train_pool = Pool(X_train, y_train, cat_features=REACTIVATION_CAT_FEATURES)
    valid_pool = Pool(X_valid, y_valid, cat_features=REACTIVATION_CAT_FEATURES)

    model = CatBoostClassifier(**REACTIVATION_CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    val_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    logger.info(f"  Best iteration  : {model.best_iteration_}")
    logger.info(f"  Validation ROC-AUC: {val_auc:.4f}")

    logger.info(f"  Positive rate (valid): {y_valid.mean():.2%}")

    return model



def save_model(model, name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(model, CatBoostClassifier):
        path = MODEL_DIR / f"{name}.cbm"
        model.save_model(str(path))
    else:
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)

    logger.info(f"  Модель сохранена: {path}")
    return path


def load_model(name: str):
    cbm_path = MODEL_DIR / f"{name}.cbm"
    pkl_path = MODEL_DIR / f"{name}.pkl"

    if cbm_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(cbm_path))
        return model
    elif pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Не найдена сохраненная модель для '{name}' in {MODEL_DIR}")


def save_metrics(metrics_list: list[dict], filename: str = "metrics.json"):
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    logger.info(f"  Метрики сохранены: {path}")



def train_calibrators(
    timing_model,
    reactivation_model,
    valid_df: pd.DataFrame,
    valid_react_df: pd.DataFrame,
    method: str = "isotonic",
):
    logger.info("─────────────── Калибровка скоров ───────────────")
    X_val, y_val = get_X_y(valid_df, model_type="timing")
    X_react, y_react = get_X_y(valid_react_df, model_type="reactivation")

    raw_timing = timing_model.predict_proba(X_val)[:, 1]
    raw_react  = reactivation_model.predict_proba(X_react)[:, 1]

    if method == "isotonic":
        cal_timing = IsotonicRegression(out_of_bounds="clip")
        cal_timing.fit(raw_timing, y_val.values)

        cal_react = IsotonicRegression(out_of_bounds="clip")
        cal_react.fit(raw_react, y_react.values)
    else:
        from sklearn.linear_model import LogisticRegression as LR
        cal_timing = LR(C=1e5)
        cal_timing.fit(raw_timing.reshape(-1, 1), y_val.values)
        cal_react = LR(C=1e5)
        cal_react.fit(raw_react.reshape(-1, 1), y_react.values)

    if method == "isotonic":
        ct = cal_timing.transform(raw_timing)
        cr = cal_react.transform(raw_react)
    else:
        ct = cal_timing.predict_proba(raw_timing.reshape(-1,1))[:,1]
        cr = cal_react.predict_proba(raw_react.reshape(-1,1))[:,1]

    logger.info(f"  Timing:      raw={raw_timing.mean():.4f} → cal={ct.mean():.4f} (true={y_val.mean():.4f})")
    logger.info(f"  Reactivation: raw={raw_react.mean():.4f} → cal={cr.mean():.4f} (true={y_react.mean():.4f})")

    return cal_timing, cal_react


def save_calibrators(cal_timing, cal_react):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in [("calibrator_timing", cal_timing), ("calibrator_reactivation", cal_react)]:
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"  Калибратор сохранён: {path}")


def run_training(
    train_df, valid_df,
    train_react_df, valid_react_df,
    with_baseline: bool = False,
):
    timing       = train_timing(train_df, valid_df)
    reactivation = train_reactivation(train_react_df, valid_react_df)

    save_model(timing,       "catboost_timing")
    save_model(reactivation, "catboost_reactivation")

    cal_timing, cal_react = train_calibrators(timing, reactivation, valid_df, valid_react_df)
    save_calibrators(cal_timing, cal_react)

    result = {
        "timing":       timing,
        "reactivation": reactivation,
        "cal_timing":   cal_timing,
        "cal_react":    cal_react,
    }

    if with_baseline:
        logger.info("─── Обучение LogReg baseline (with_baseline=True) ───")
        logreg       = train_logreg(train_df, valid_df)
        logreg_react = train_logreg_reactivation(train_react_df, valid_react_df)
        save_model(logreg,       "logreg_baseline")
        save_model(logreg_react, "logreg_reactivation_baseline")
        result["logreg"]       = logreg
        result["logreg_react"] = logreg_react

    logger.info("─────────────── Обучение завершено ───────────────")
    return result
