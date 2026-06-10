import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import (
    TIMING_CAT_FEATURES,
    TIMING_NUMERIC_FEATURES,
    REACTIVATION_CAT_FEATURES,
    REACTIVATION_NUMERIC_FEATURES,
    COLD_START_THRESHOLD_DAYS,
    DORMANCY_THRESHOLD_DAYS,
    OUTPUT_DIR,
    TARGET_COL,
)
from dataset_builder import get_X_y

logger = logging.getLogger(__name__)

EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "logreg":   "#2563EB",
    "catboost": "#DC2626",
    "baseline": "#94A3B8",
    "positive": "#2563EB",
    "negative": "#94A3B8",
    "green":    "#16A34A",
    "orange":   "#EA580C",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "font.size":         10,
    "figure.dpi":        130,
})



def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float):
    threshold = np.percentile(y_score, (1 - k) * 100)
    predicted = (y_score >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float):
    threshold = np.percentile(y_score, (1 - k) * 100)
    predicted = (y_score >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    pp = int(predicted.sum())
    return tp / pp if pp > 0 else 0.0


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float):
    base_rate = y_true.mean()
    if base_rate == 0:
        return 0.0
    return precision_at_k(y_true, y_score, k) / base_rate


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, model_name: str,top_k_fracs: list[float] | None = None):
    if top_k_fracs is None:
        top_k_fracs = [0.10, 0.20, 0.30]

    metrics: dict[str, Any] = {
        "model": model_name,
        "n_rows": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "pos_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }

    for k in top_k_fracs:
        tag = f"top{int(k * 100)}pct"
        metrics[f"recall@{tag}"] = float(recall_at_k(y_true, y_score, k))
        metrics[f"precision@{tag}"] = float(precision_at_k(y_true, y_score, k))
        metrics[f"lift@{tag}"] = float(lift_at_k(y_true, y_score, k))

    return metrics


def log_metrics(metrics: dict):
    w = 54
    logger.info("─" * w)
    logger.info(f"  Модель: {metrics['model']}")
    logger.info(f"  Строк: {metrics['n_rows']:,}   "
                f"Положительных меток: {metrics['n_pos']:,}  ({metrics['pos_rate']:.2%})")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}  ← primary (imbalanced data)")
    logger.info(f"  Brier: {metrics['brier']:.4f}  (lower = better calibration)")

    for key, val in metrics.items():
        if key.startswith("recall@"):
            tag = key.split("@")[1]
            prec = metrics.get(f"precision@{tag}", 0)
            lift = metrics.get(f"lift@{tag}", 0)
            logger.info(
                f"  {tag:12s}  Recall={val:.3f}  "
                f"Precision={prec:.3f}  Lift={lift:.2f}x"
            )
    logger.info("─" * w)



def evaluate_by_segment(model, df, model_name, model_type="timing"):
    df = df.copy()
    if model_type == "timing":
        if "customer_tenure_days" not in df.columns:
            logger.warning("'customer_tenure_days' не найден")
            return pd.DataFrame()
        df["segment"] = np.where(
            df["customer_tenure_days"] < COLD_START_THRESHOLD_DAYS,
            "cold", "warm"
        )
        segments = ["warm", "cold", "all"]

    elif model_type == "reactivation":
        if "current_pause_days" not in df.columns:
            logger.warning("'current_pause_days' не найден")
            return pd.DataFrame()
        df["segment"] = pd.cut(
            df["current_pause_days"],
            bins=[0, 90, 180, 365, 99999],
            labels=["60-90d", "90-180d", "180-365d", "365d+"]
        ).astype(str)
        segments = ["60-90d", "90-180d", "180-365d", "365d+", "all"]

    X_all, _ = get_X_y(df, model_type=model_type)
    df["score"] = model.predict_proba(X_all)[:, 1]

    results = []
    for seg in segments:
        sub = df if seg == "all" else df[df["segment"] == seg]
        if len(sub) == 0 or sub[TARGET_COL].nunique() < 2:
            continue
        m = compute_metrics(sub[TARGET_COL].values, sub["score"].values, model_name=f"{model_name} [{seg}]")
        m["segment"] = seg
        results.append(m)
        log_metrics(m)

    return pd.DataFrame(results)



def plot_roc_pr(models_scores: dict[str, tuple[np.ndarray, np.ndarray]], save_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    color_cycle = list(PALETTE.values())

    for ax, curve_type in zip(axes, ["roc", "pr"]):
        for i, (name, (y_true, y_score)) in enumerate(models_scores.items()):
            color = color_cycle[i % len(color_cycle)]

            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_val = roc_auc_score(y_true, y_score)
                ax.plot(fpr, tpr, label=f"{name}  (AUC={auc_val:.3f})",
                        color=color, lw=2)
            else:
                prec, rec, _ = precision_recall_curve(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                ax.plot(rec, prec, label=f"{name}  (AP={ap:.3f})",
                        color=color, lw=2)

        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve", fontweight="bold")
        else:
            base = list(models_scores.values())[0][0].mean()
            ax.axhline(base, ls="--", color=PALETTE["baseline"], lw=1.2,
                       label=f"No-skill  (={base:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve", fontweight="bold")

        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Propensity Model — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Графики ROC/PR-кривых сохранены: {save_path}")
    plt.close(fig)



def plot_score_distributions(models_scores: dict[str, tuple[np.ndarray, np.ndarray]], save_path: Path | None = None):
    n = len(models_scores)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, (y_true, y_score)) in zip(axes, models_scores.items()):
        for label, color, lbl in [
            (0, PALETTE["negative"], "Non-buyer (label=0)"),
            (1, PALETTE["positive"], "Buyer (label=1)"),
        ]:
            mask = y_true == label
            ax.hist(y_score[mask], bins=40, alpha=0.55,
                    color=color, label=lbl, density=True)

        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Score Distribution by True Label", fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Графики распределения скоров сохранены: {save_path}")
    plt.close(fig)



def plot_calibration(models_scores: dict[str, tuple[np.ndarray, np.ndarray]], n_bins: int = 10, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect calibration")

    color_cycle = [PALETTE["logreg"], PALETTE["catboost"], PALETTE["green"]]

    for i, (name, (y_true, y_score)) in enumerate(models_scores.items()):
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, "o-", color=color_cycle[i],
                lw=2, ms=5, label=name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  График калибровочной кривой сохранен: {save_path}")
    plt.close(fig)



def plot_lift_curve(models_scores: dict[str, tuple[np.ndarray, np.ndarray]], save_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    color_cycle = [PALETTE["logreg"], PALETTE["catboost"], PALETTE["green"]]

    for ax_idx, (ax, chart_type) in enumerate(zip(axes, ["gain", "lift"])):
        for i, (name, (y_true, y_score)) in enumerate(models_scores.items()):
            order = np.argsort(y_score)[::-1]
            y_sorted = y_true[order]

            n = len(y_sorted)
            x = np.concatenate([[0], np.arange(1, n + 1) / n])
            cumulative_pos = np.concatenate([[0], np.cumsum(y_sorted) / y_sorted.sum()])

            if chart_type == "gain":
                ax.plot(x, cumulative_pos, color=color_cycle[i], lw=2, label=name)
            else:
                lift = cumulative_pos / x
                ax.plot(x, lift, color=color_cycle[i], lw=2, label=name)

        if chart_type == "gain":
            ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random")
            ax.set_ylabel("Fraction of buyers captured")
            ax.set_title("Cumulative Gain Chart", fontweight="bold")
        else:
            ax.axhline(1.0, ls="--", color=PALETTE["baseline"],
                       lw=1.2, label="Random (lift=1)")
            ax.set_ylabel("Lift vs random")
            ax.set_title("Lift Curve", fontweight="bold")

        ax.set_xlabel("Fraction of customers contacted (by score)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)

        ax.axvline(0.20, ls=":", color=PALETTE["orange"], lw=1.5, alpha=0.8,
                   label="20% threshold")

    fig.suptitle("Cumulative Gain & Lift", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Кривая лифта сохранена {save_path}")
    plt.close(fig)



def plot_feature_importance(model, top_n=15, save_path=None, model_type="timing"):
    try:
        fi = pd.DataFrame({
            "feature": model.feature_names_,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    except AttributeError:
        logger.warning("Модель не предоставляет feature_names_")
        return pd.DataFrame()

    top = fi.head(top_n).iloc[::-1]

    cat_features = (
        TIMING_CAT_FEATURES if model_type == "timing"
        else REACTIVATION_CAT_FEATURES
    )

    def _group_color(feature):
        if feature in cat_features:
            return PALETTE["orange"]
        if any(x in feature for x in ["buy", "sell", "value", "ticket"]):
            return PALETTE["logreg"]
        if any(x in feature for x in ["trend", "month", "new_customer"]):
            return PALETTE["green"]
        if any(x in feature for x in ["since", "tenure"]):
            return PALETTE["catboost"]
        if any(x in feature for x in ["share", "unique"]):
            return "#9333EA"
        if any(x in feature for x in ["interval", "pause", "survived",
                                        "cadence", "gap"]):
            return "#0891B2"
        if any(x in feature for x in ["market", "return", "volatility",
                                        "drawdown"]):
            return "#059669"
        if any(x in feature for x in ["seasonal", "same_month",
                                        "same_quarter"]):
            return "#D97706"
        if "_x_" in feature:
            return "#DC2626"
        return PALETTE["baseline"]

    colors = [_group_color(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["feature"], top["importance"], color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xlabel("Importance (CatBoost default)")

    model_label = "Timing" if model_type == "timing" else "Reactivation"
    ax.set_title(
        f"Top-{top_n} Feature Importances — CatBoost {model_label}",
        fontweight="bold"
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color=PALETTE["orange"],   label="Profile (categorical)"),
        Patch(color=PALETTE["logreg"],   label="RFM"),
        Patch(color=PALETTE["green"],    label="Temporal dynamics"),
        Patch(color=PALETTE["catboost"], label="Time-since"),
        Patch(color="#9333EA",           label="Portfolio"),
        Patch(color="#0891B2",           label="Cadence"),
        Patch(color="#059669",           label="Market context"),
        Patch(color="#D97706",           label="Seasonality"),
        Patch(color="#DC2626",           label="Interactions"),
        Patch(color=PALETTE["baseline"], label="Other"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  График важности признаков сохранен: {save_path}")
    plt.close(fig)

    return fi



def plot_metrics_table(metrics_list: list[dict], save_path: Path | None = None):
    cols_order = [
        "model", "roc_auc", "pr_auc", "brier",
        "recall@top10pct", "precision@top10pct", "lift@top10pct",
        "recall@top20pct", "precision@top20pct", "lift@top20pct",
    ]
    display_names = {
        "model": "Model",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC ★",
        "brier": "Brier ↓",
        "recall@top10pct": "Recall\n@10%",
        "precision@top10pct": "Prec\n@10%",
        "lift@top10pct": "Lift\n@10%",
        "recall@top20pct": "Recall\n@20%",
        "precision@top20pct": "Prec\n@20%",
        "lift@top20pct": "Lift\n@20%",
    }

    rows = []
    for m in metrics_list:
        row = {k: m.get(k, "—") for k in cols_order}
        rows.append(row)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 1.5 + 0.5 * len(rows)))
    ax.axis("off")

    headers  = [display_names.get(c, c) for c in cols_order]
    cell_data = []
    for _, row in df.iterrows():
        cell_row = []
        for c in cols_order:
            v = row[c]
            if isinstance(v, float):
                cell_row.append(f"{v:.4f}" if c in ("roc_auc", "pr_auc", "brier")
                                else f"{v:.3f}")
            else:
                cell_row.append(str(v))
        cell_data.append(cell_row)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2)

    for j in range(len(cols_order)):
        tbl[0, j].set_facecolor("#1E3A5F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        bg = "#F0F4FF" if i % 2 == 0 else "white"
        for j in range(len(cols_order)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title("Model Comparison — Test Set Metrics", fontweight="bold",
                 fontsize=12, pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved metrics table → {save_path}")
    plt.close(fig)



def run_evaluation(
    models: dict,
    test_df: pd.DataFrame,
    test_react_df: pd.DataFrame,
    top_k_fracs=None,
):
    if top_k_fracs is None:
        top_k_fracs = [0.10, 0.20, 0.30]

    logger.info("\n" + "═" * 56)
    logger.info("  ОЦЕНКА — TIMING МОДЕЛИ (тестовая выборка)")
    logger.info("═" * 56)

    timing_models = {
        k: v for k, v in models.items()
        if k in ("logreg", "timing")
    }

    X_test, y_test = get_X_y(test_df, model_type="timing")
    timing_scores = {}
    timing_metrics = {}

    TIMING_DISPLAY_NAMES = {
        "logreg": "LogReg Timing",
        "timing": "CatBoost Timing",
    }

    for name, model in timing_models.items():
        display_name = TIMING_DISPLAY_NAMES.get(name, name)
        y_score = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_score, display_name, top_k_fracs)
        log_metrics(metrics)
        timing_metrics[display_name] = metrics
        timing_scores[display_name] = (y_test.values, y_score)

    logger.info("\n─── Оценка timing по сегментам ───")
    timing_seg_results = []
    for name, model in timing_models.items():
        seg_df = evaluate_by_segment(model, test_df, name, model_type="timing")
        if not seg_df.empty:
            timing_seg_results.append(seg_df)

    logger.info("\n" + "═" * 56)
    logger.info("  ОЦЕНКА — REACTIVATION МОДЕЛИ (тестовая выборка)")
    logger.info("═" * 56)

    react_models = {
        k: v for k, v in models.items()
        if k in ("logreg_react", "reactivation")
    }
    react_metrics = {}
    react_scores = {}

    if not test_react_df.empty:
        X_react, y_react = get_X_y(test_react_df, model_type="reactivation")

        REACT_DISPLAY_NAMES = {
            "logreg_react": "LogReg Reactivation",
            "reactivation": "CatBoost Reactivation",
        }

        for name, model in react_models.items():
            if model is None:
                continue
            display_name = REACT_DISPLAY_NAMES.get(name, name)
            y_score_react = model.predict_proba(X_react)[:, 1]
            metrics_react = compute_metrics(y_react.values, y_score_react, display_name, top_k_fracs)
            log_metrics(metrics_react)
            react_metrics[display_name] = metrics_react
            react_scores[display_name]  = (y_react.values, y_score_react)


        logger.info("\n─── Оценка reactivation по глубине паузы ───")
        if "reactivation" in react_models and react_models["reactivation"] is not None:
            seg_react = evaluate_by_segment(
                react_models["reactivation"], test_react_df, "reactivation",
                model_type="reactivation"
            )
        if not seg_react.empty:
            seg_path = EVAL_DIR / "segment_metrics_reactivation.csv"
            seg_react.to_csv(seg_path, index=False)
            logger.info(f"  Метрики по сегментам reactivation: {seg_path}")

    logger.info("\n─── Создание графиков ───")

    plot_roc_pr(timing_scores, save_path=EVAL_DIR / "roc_pr_timing.png")
    plot_score_distributions(timing_scores,
                             save_path=EVAL_DIR / "score_dist_timing.png")
    plot_calibration(timing_scores,
                     save_path=EVAL_DIR / "calibration_timing.png")
    plot_lift_curve(timing_scores,
                    save_path=EVAL_DIR / "lift_timing.png")

    if react_scores:
        plot_roc_pr(react_scores,
                    save_path=EVAL_DIR / "roc_pr_reactivation.png")
        plot_score_distributions(react_scores,
                                 save_path=EVAL_DIR / "score_dist_reactivation.png")
        plot_calibration(react_scores,
                         save_path=EVAL_DIR / "calibration_reactivation.png")
        plot_lift_curve(react_scores,
                        save_path=EVAL_DIR / "lift_reactivation.png")

    for name, model in timing_models.items():
        if hasattr(model, "feature_names_"):
            fi_df = plot_feature_importance(
                model,
                save_path=EVAL_DIR / f"feature_importance_{name}.png",
                model_type="timing",
            )
            fi_df.to_csv(EVAL_DIR / f"feature_importance_{name}.csv", index=False)

    cb_react = react_models.get("reactivation")
    if cb_react is not None and hasattr(cb_react, "feature_names_"):
        fi_df = plot_feature_importance(
            cb_react,
            save_path=EVAL_DIR / "feature_importance_reactivation.png",
            model_type="reactivation",
        )
        fi_df.to_csv(
            EVAL_DIR / "feature_importance_reactivation.csv", index=False
        )

    all_metrics = {**timing_metrics, **react_metrics}

    plot_metrics_table(
        list(timing_metrics.values()),
        save_path=EVAL_DIR / "metrics_table_timing.png",
    )

    if react_metrics:
        plot_metrics_table(
            list(react_metrics.values()),
            save_path=EVAL_DIR / "metrics_table_reactivation.png",
        )

    json_path = EVAL_DIR / "all_metrics.json"
    with open(json_path, "w") as f:
        json.dump(list(all_metrics.values()), f, indent=2)
    logger.info(f"\n  Все метрики сохранены: {json_path}")

    if timing_seg_results:
        seg_summary = pd.concat(timing_seg_results, ignore_index=True)
        seg_summary.to_csv(EVAL_DIR / "segment_metrics_timing.csv", index=False)

    logger.info("═" * 56 + "\n")
    return all_metrics