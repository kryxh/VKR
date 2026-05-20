import json
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TOP_K, MIN_TX_EVAL
from train import (
    build_loo_pairs,
    compute_fusion_score,
    _ndcg_at_k,
    EASEModel,
    evaluate_loo_ease,
)
from matrix_builder import InteractionMatrix
from implicit.als import AlternatingLeastSquares

logger = logging.getLogger(__name__)

EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)



def _precision_at_k(predicted_ranked: list, relevant_item: str, k: int = TOP_K) -> float:
    return 1.0 if relevant_item in predicted_ranked[:k] else 0.0


def _get_top_k_for_user(
    uid: str,
    hidden_isin: str,
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    fusion_alpha: float,
    k: int = TOP_K,
    exclude_recent: set | None = None,
) -> list[str]:
    u_idx  = imat.user_id_to_idx[uid]
    scores = compute_fusion_score(u_idx, model, imat, fusion_alpha)

    hist_isins = set(imat.user_items(uid)) - {hidden_isin}
    for isin in hist_isins:
        if isin in imat.item_id_to_idx:
            scores[imat.item_id_to_idx[isin]] = -np.inf

    if exclude_recent:
        for isin in exclude_recent:
            if isin in imat.item_id_to_idx:
                scores[imat.item_id_to_idx[isin]] = -np.inf

    top_idx   = np.argsort(scores)[::-1][:k]
    top_isins = [imat.idx_to_item_id[i] for i in top_idx]
    return top_isins



def evaluate_als_segment(
    loo_pairs: list[tuple[str, str]],
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    fusion_alpha: float,
    tx_val: pd.DataFrame,
    k: int = TOP_K,
    stage: Literal["validation", "test"] = "validation",
) -> dict:
    if not loo_pairs:
        logger.warning("Нет LOO-пар для ALS evaluation.")
        return {}

    records = []
    for uid, hidden_isin in loo_pairs:
        top_k = _get_top_k_for_user(uid, hidden_isin, model, imat, fusion_alpha, k)

        ndcg = _ndcg_at_k(top_k, hidden_isin, k)
        prec = _precision_at_k(top_k, hidden_isin, k)

        uid_val = tx_val[
            (tx_val["customerID"] == uid) &
            (tx_val["ISIN"] == hidden_isin)
        ]["timestamp"]
        month = uid_val.iloc[-1].to_period("M") if not uid_val.empty else None

        records.append({"uid": uid, "ndcg3": ndcg, "prec3": prec, "month": month})

    df = pd.DataFrame(records)
    overall = {
        "stage":          stage,
        "segment":        "als",
        "n_users":        len(df),
        f"ndcg@{k}":     float(df["ndcg3"].mean()),
        f"precision@{k}": float(df["prec3"].mean()),
    }

    monthly = (
        df.groupby("month")[["ndcg3", "prec3"]]
        .mean()
        .rename(columns={"ndcg3": f"ndcg@{k}", "prec3": f"precision@{k}"})
    )

    overall["monthly_breakdown"] = {
        str(k): v for k, v in monthly.to_dict().items()
    }

    logger.info(
        f"  [{stage.upper()}] ALS | n={len(df)} | "
        f"NDCG@{k}={overall[f'ndcg@{k}']:.4f} | "
        f"Precision@{k}={overall[f'precision@{k}']:.4f}"
    )

    for m, row in monthly.iterrows():
        logger.info(f"    {m}: NDCG@{k}={row[f'ndcg@{k}']:.4f}")

    ndcg_vals = list(overall["monthly_breakdown"][f"ndcg@{k}"].values())
    if len(ndcg_vals) >= 2:
        first_month_ndcg = ndcg_vals[0]
        rest_mean        = np.mean(ndcg_vals[1:])
        if first_month_ndcg > rest_mean * 1.5:
            logger.warning(
                f"  ⚠ Первый месяц validation аномально высокий: "
                f"{first_month_ndcg:.4f} vs среднее остальных {rest_mean:.4f}. "
                f"Возможный эффект близости к TRAIN_END или сезонность. "
                f"NDCG@{k} без первого месяца: {rest_mean:.4f}"
            )
            overall["ndcg_excl_first_month"] = float(rest_mean)

    return overall


def evaluate_ease_segment(
    loo_pairs: list,
    ease_model: EASEModel,
    imat: InteractionMatrix,
    tx_val: pd.DataFrame,
    k: int = TOP_K,
    stage: str = "validation",
    imat_ease: "InteractionMatrix | None" = None,
) -> dict:

    _imat = imat_ease if imat_ease is not None else imat

    if not loo_pairs or ease_model.W is None:
        return {}

    records = []
    for uid, hidden_isin in loo_pairs:
        u_idx = _imat.user_id_to_idx.get(uid)
        if u_idx is None or u_idx >= _imat.n_users:
            records.append({"uid": uid, "ndcg3": 0.0, "prec3": 0.0, "month": None})
            continue

        if hidden_isin not in _imat.item_id_to_idx:
            records.append({"uid": uid, "ndcg3": 0.0, "prec3": 0.0, "month": None})
            continue

        scores = ease_model.score_user_from_matrix(u_idx, _imat)

        for isin in set(_imat.user_items(uid)) - {hidden_isin}:
            if isin in _imat.item_id_to_idx:
                scores[_imat.item_id_to_idx[isin]] = -np.inf

        top_k = [
            _imat.idx_to_item_id[int(i)]
            for i in np.argsort(scores)[::-1]
            if int(i) in _imat.idx_to_item_id
        ][:k]

        uid_val = tx_val[
            (tx_val["customerID"] == uid) &
            (tx_val["ISIN"] == hidden_isin)
        ]["timestamp"]
        month = uid_val.iloc[-1].to_period("M") if not uid_val.empty else None

        records.append({
            "uid":   uid,
            "ndcg3": _ndcg_at_k(top_k, hidden_isin, k),
            "prec3": _precision_at_k(top_k, hidden_isin, k),
            "month": month,
        })

    if not records:
        return {}

    df = pd.DataFrame(records)
    result = {
        "stage":           stage,
        "segment":         "ease",
        "n_users":         len(df),
        f"ndcg@{k}":      float(df["ndcg3"].mean()),
        f"precision@{k}": float(df["prec3"].mean()),
        "monthly_breakdown": {
            str(k_): v for k_, v in (
                df.groupby("month")[["ndcg3", "prec3"]]
                .mean()
                .rename(columns={"ndcg3": f"ndcg@{k}", "prec3": f"precision@{k}"})
                .to_dict()
                .items()
            )
        },
    }
    logger.info(
        f"  [{stage.upper()}] EASE | n={len(df)} | "
        f"NDCG@{k}={result[f'ndcg@{k}']:.4f} | "
        f"Precision@{k}={result[f'precision@{k}']:.4f}"
    )
    return result


def evaluate_fallback_segment(
    recommendations: pd.DataFrame,
    tx_val: pd.DataFrame,
    stage: Literal["validation", "test"] = "validation",
) -> dict:
 
    fallback = recommendations[recommendations["rec_type"] == "fallback"].copy()
    if fallback.empty:
        return {"stage": stage, "segment": "fallback", "n_users": 0, "hitrate@1": 0.0}

    val_pairs = set(zip(tx_val["customerID"], tx_val["ISIN"]))

    hits = 0
    for _, row in fallback.iterrows():
        uid = row["customerID"]
        top3 = [row["rank_1_isin"], row["rank_2_isin"], row["rank_3_isin"]]
        top3 = [x for x in top3 if pd.notna(x)]
        if any((uid, isin) in val_pairs for isin in top3):
            hits += 1

    hitrate = hits / len(fallback)

    logger.info(
        f"  [{stage.upper()}] Fallback | n={len(fallback)} | "
        f"HitRate@3={hitrate:.4f}"
    )
    return {
        "stage":       stage,
        "segment":     "fallback",
        "n_users":     len(fallback),
        "hitrate@3":   float(hitrate),
    }


def evaluate_popularity_baseline(
    loo_pairs: list,
    imat: InteractionMatrix,
    k: int = TOP_K,
    stage: str = "validation",
) -> dict:

    if not loo_pairs:
        return {}

    meta = imat.item_meta.reset_index()
    top_by_cat = {}
    for cat in meta["assetCategory"].dropna().unique():
        top_isins = (
            meta[meta["assetCategory"] == cat]
            .nlargest(k, "n_buyers")["ISIN"]
            .tolist()
        )
        top_by_cat[cat] = top_isins

    global_top = meta.nlargest(k, "n_buyers")["ISIN"].tolist()

    ndcg_scores = []
    prec_scores = []

    for uid, hidden_isin in loo_pairs:
        if hidden_isin in imat.item_meta.index:
            cat = imat.item_meta.loc[hidden_isin, "assetCategory"]
            top_k = top_by_cat.get(cat, global_top)
        else:
            top_k = global_top

        ndcg_scores.append(_ndcg_at_k(top_k, hidden_isin, k))
        prec_scores.append(_precision_at_k(top_k, hidden_isin, k))

    result = {
        "stage":           stage,
        "segment":         "popularity_baseline",
        "n_users":         len(loo_pairs),
        f"ndcg@{k}":      float(np.mean(ndcg_scores)),
        f"precision@{k}": float(np.mean(prec_scores)),
    }
    logger.info(
        f"  [{stage.upper()}] Popularity baseline | "
        f"NDCG@{k}={result[f'ndcg@{k}']:.4f} | "
        f"Precision@{k}={result[f'precision@{k}']:.4f}"
    )
    logger.info(
        f"  CF lift over popularity: "
        f"NDCG Δ=будет посчитан в run_evaluation"
    )
    return result



def evaluate_coverage_and_consistency(
    recommendations: pd.DataFrame,
    imat: InteractionMatrix,
    stage: Literal["validation", "test"] = "validation",
) -> dict:

    all_rec_isins = []
    for col in ["rank_1_isin", "rank_2_isin", "rank_3_isin"]:
        if col in recommendations.columns:
            all_rec_isins.extend(recommendations[col].dropna().tolist())

    unique_rec = set(all_rec_isins)
    all_items  = set(imat.item_id_to_idx.keys())
    coverage   = len(unique_rec & all_items) / len(all_items) if all_items else 0.0

    outside_cols = [c for c in recommendations.columns if "outside_hist" in c]
    if outside_cols:
        outside_vals = pd.concat(
            [recommendations[c] for c in outside_cols], ignore_index=True
        ).dropna()
        consistency_rate = (outside_vals == False).mean()
    else:
        consistency_rate = float("nan")

    result = {
        "stage":                      stage,
        "unique_isins_recommended":   len(unique_rec),
        "total_als_items":            len(all_items),
        "coverage":                   float(coverage),
        "behavioral_consistency_rate": float(consistency_rate),
    }

    logger.info(
        f"  [{stage.upper()}] Coverage={coverage:.2%} | "
        f"BehavioralConsistency={consistency_rate:.2%}"
    )
    return result


def plot_monthly_ndcg(monthly_data: dict, stage: str, k: int = TOP_K):
    ndcg_col = f"ndcg@{k}"
    if ndcg_col not in monthly_data:
        return

    months = sorted(monthly_data[ndcg_col].keys(), key=str)
    values = [monthly_data[ndcg_col][m] for m in months]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(months)), values, marker="o", color="#2563EB", linewidth=2)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m) for m in months], rotation=30, ha="right")
    ax.set_title(f"NDCG@{k} by Month [{stage}]", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"NDCG@{k}")
    ax.spines[["top", "right"]].set_visible(False)

    if len(values) >= 3:
        z = np.polyfit(range(len(values)), values, 1)
        p = np.poly1d(z)
        ax.plot(range(len(months)), p(range(len(months))),
                "--", color="#DC2626", alpha=0.6, label=f"trend (slope={z[0]:.4f})")
        ax.legend(fontsize=9)
        if z[0] < -0.005:
            logger.warning(
                f"  ⚠ Temporal degradation: NDCG@{k} падает на {z[0]:.4f}/месяц"
            )

    plt.tight_layout()
    path = EVAL_DIR / f"ndcg_monthly_{stage}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  График сохранён: {path}")


def plot_grid_search_results(weights: dict):
    als_results = weights.get("als_grid_results", [])
    if als_results:
        df_als = pd.DataFrame(als_results)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        k_ndcg = df_als.groupby("factors")["ndcg3"].max()
        axes[0].bar(k_ndcg.index.astype(str), k_ndcg.values, color="#2563EB", alpha=0.8)
        axes[0].set_title("Max NDCG@3 by k (ALS factors)", fontweight="bold")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("NDCG@3")

        alpha_ndcg = df_als.groupby("confidence_alpha")["ndcg3"].max()
        axes[1].bar(alpha_ndcg.index.astype(str), alpha_ndcg.values, color="#7C3AED", alpha=0.8)
        axes[1].set_title("Max NDCG@3 by CONFIDENCE_ALPHA", fontweight="bold")
        axes[1].set_xlabel("CONFIDENCE_ALPHA")

        plt.tight_layout()
        path = EVAL_DIR / "als_grid_search.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    fusion_results = weights.get("fusion_grid_results", [])
    if fusion_results:
        df_f = pd.DataFrame(fusion_results)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_f["fusion_alpha"], df_f["ndcg3"],
                marker="o", color="#2563EB", linewidth=2)
        ax.axvline(weights["fusion_alpha"], color="#DC2626",
                   linestyle="--", label=f"best α={weights['fusion_alpha']:.1f}")
        ax.set_title("NDCG@3 vs FUSION_ALPHA", fontweight="bold")
        ax.set_xlabel("FUSION_ALPHA (item weight)")
        ax.set_ylabel("NDCG@3")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = EVAL_DIR / "fusion_alpha_search.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Графики grid search сохранены в {EVAL_DIR}")



def _make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_metrics(metrics: list[dict], stage: str):
    path = EVAL_DIR / f"metrics_{stage}.json"
    with open(path, "w") as f:
        json.dump(_make_serializable(metrics), f, indent=2, ensure_ascii=False)
    logger.info(f"  Метрики сохранены: {path}")


def run_evaluation(
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    weights: dict,
    tx_val: pd.DataFrame,
    recommendations: pd.DataFrame,
    stage: Literal["validation", "test"] = "validation",
    ease_model: "EASEModel | None" = None,
    imat_ease: "InteractionMatrix | None" = None,
) -> dict:

    logger.info(f"═" * 60)
    logger.info(f"  ОЦЕНКА МОДЕЛИ [{stage.upper()}]")
    logger.info(f"═" * 60)

    fusion_alpha = weights["fusion_alpha"]

    loo_pairs = build_loo_pairs(imat, tx_val, min_history_isins=MIN_TX_EVAL)

    als_metrics = evaluate_als_segment(
        loo_pairs, model, imat, fusion_alpha, tx_val, stage=stage
    )

    ease_metrics = {}
    if ease_model is not None:
        ease_metrics = evaluate_ease_segment(
            loo_pairs, ease_model, imat, tx_val, stage=stage, imat_ease=imat_ease,
        )
        als_ndcg  = als_metrics.get(f"ndcg@{TOP_K}", 0)
        ease_ndcg = ease_metrics.get(f"ndcg@{TOP_K}", 0)
        pop_ndcg  = 0
        logger.info(f"  Сравнение ALS vs EASE: "
                    f"ALS={als_ndcg:.4f}, EASE={ease_ndcg:.4f}, "
                    f"Δ={ease_ndcg - als_ndcg:+.4f} "
                    f"({'EASE лучше ✓' if ease_ndcg > als_ndcg else 'ALS лучше'})")

    fallback_metrics = evaluate_fallback_segment(recommendations, tx_val, stage=stage)

    popularity_metrics = evaluate_popularity_baseline(loo_pairs, imat, stage=stage)

    cf_ndcg  = als_metrics.get(f"ndcg@{TOP_K}", 0)
    pop_ndcg = popularity_metrics.get(f"ndcg@{TOP_K}", 0)
    if pop_ndcg > 0:
        lift = (cf_ndcg - pop_ndcg) / pop_ndcg * 100
        logger.info(f"  CF lift над popularity baseline: {lift:+.1f}%")
        if lift < 0:
            logger.warning(
                f"  ⚠ CF ХУЖЕ popularity baseline на {abs(lift):.1f}%! "
                f"Рекомендации не добавляют ценности сверх популярности."
            )

    global_metrics = evaluate_coverage_and_consistency(recommendations, imat, stage=stage)

    if als_metrics.get("monthly_breakdown"):
        plot_monthly_ndcg(als_metrics["monthly_breakdown"], stage=stage)

    all_metrics = [als_metrics, ease_metrics, popularity_metrics, fallback_metrics, global_metrics]
    save_metrics(all_metrics, stage=stage)

    pop_ndcg  = popularity_metrics.get(f"ndcg@{TOP_K}", 0)
    als_ndcg  = als_metrics.get(f"ndcg@{TOP_K}", 0)
    ease_ndcg = ease_metrics.get(f"ndcg@{TOP_K}", 0) if ease_metrics else None

    logger.info("─" * 60)
    logger.info("  ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    logger.info(f"  Popularity baseline : NDCG@{TOP_K}={pop_ndcg:.4f}")
    logger.info(f"  ALS                 : NDCG@{TOP_K}={als_ndcg:.4f}  "
                f"(lift={((als_ndcg-pop_ndcg)/pop_ndcg*100 if pop_ndcg else 0):+.1f}%)")
    if ease_ndcg is not None:
        logger.info(f"  EASE                : NDCG@{TOP_K}={ease_ndcg:.4f}  "
                    f"(lift={((ease_ndcg-pop_ndcg)/pop_ndcg*100 if pop_ndcg else 0):+.1f}%)")
        best = "EASE" if ease_ndcg >= als_ndcg else "ALS"
        logger.info(f"  → Лучшая модель: {best}")
    logger.info("─" * 60)

    return {
        "als":        als_metrics,
        "ease":       ease_metrics,
        "popularity": popularity_metrics,
        "fallback":   fallback_metrics,
        "global":     global_metrics,
    }
