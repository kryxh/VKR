import json
import logging
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

from config import (
    ALS_FACTORS_GRID,
    ALS_REGULARIZATION_GRID,
    CONFIDENCE_ALPHA_GRID,
    DECAY_LAMBDA_GRID,
    FUSION_ALPHA_GRID,
    EASE_REGULARIZATION_GRID,
    ALS_ITERATIONS,
    MIN_TX_EVAL,
    TOP_K,
    MODEL_DIR,
)
from matrix_builder import InteractionMatrix, build_interaction_matrix
from data_loader import build_item_index

logger = logging.getLogger(__name__)



# ALS обучение
def train_als(
    imat: InteractionMatrix,
    factors: int,
    regularization: float,
    iterations: int = ALS_ITERATIONS,
    random_state: int = 42,
) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors        = factors,
        regularization = regularization,
        iterations     = iterations,
        random_state   = random_state,
        use_gpu        = False,
    )
    # implicit принимает (n_items, n_users)
    item_users = imat.matrix.T.tocsr()
    model.fit(item_users, show_progress=False)

    # Гарантируем соответствие размеров
    model.user_factors = model.user_factors[:imat.n_users]
    model.item_factors = model.item_factors[:imat.n_items]
    return model


def train_als_best(imat: InteractionMatrix) -> AlternatingLeastSquares:
    from config import ALS_BEST_FACTORS, ALS_BEST_REGULARIZATION
    logger.info(
        f"  ALS fixed: factors={ALS_BEST_FACTORS}, "
        f"regularization={ALS_BEST_REGULARIZATION}"
    )
    return train_als(imat, factors=ALS_BEST_FACTORS, regularization=ALS_BEST_REGULARIZATION)



# CF-скоры
def _percentile_rank_normalize(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return scores
    if n == 1:
        return np.array([1.0])
    ranks = np.argsort(np.argsort(scores)).astype(float)
    return ranks / (n - 1)


def compute_item_score(
    user_idx: int,
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
) -> np.ndarray:
    row   = imat.matrix.getrow(user_idx)
    h_idx = row.indices
    if len(h_idx) == 0:
        return np.zeros(imat.n_items)
    V        = model.item_factors
    mean_vec = V[h_idx].mean(axis=0)
    return V @ mean_vec


def compute_user_score(
    user_idx: int,
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
) -> np.ndarray:
    if user_idx >= model.user_factors.shape[0]:
        return np.zeros(imat.n_items)
    return model.item_factors @ model.user_factors[user_idx]


def compute_fusion_score(
    user_idx: int,
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    fusion_alpha: float,
) -> np.ndarray:
    s_item = compute_item_score(user_idx, model, imat)
    s_user = compute_user_score(user_idx, model, imat)
    return (
        fusion_alpha       * _percentile_rank_normalize(s_item) +
        (1 - fusion_alpha) * _percentile_rank_normalize(s_user)
    )



# LOO evaluation


def _ndcg_at_k(predicted_ranked: list, relevant_item: str, k: int = TOP_K) -> float:
    top_k = predicted_ranked[:k]
    if relevant_item not in top_k:
        return 0.0
    rank = top_k.index(relevant_item) + 1
    return 1.0 / np.log2(rank + 1)


def build_loo_pairs(
    imat: InteractionMatrix,
    tx_val: pd.DataFrame,
    min_history_isins: int = MIN_TX_EVAL,
) -> list:
    als_item_set = set(imat.item_id_to_idx.keys())
    loo_pairs    = []

    for uid, group in tx_val.groupby("customerID"):
        if uid not in imat.user_id_to_idx:
            continue
        if len(imat.user_items(uid)) < min_history_isins:
            continue
        val_in_space = (
            group.sort_values("timestamp")
                 .loc[group["ISIN"].isin(als_item_set), "ISIN"]
        )
        if val_in_space.empty:
            continue
        loo_pairs.append((uid, val_in_space.iloc[-1]))

    logger.info(f"  LOO pairs: {len(loo_pairs):,} клиентов")
    return loo_pairs


def evaluate_loo(
    loo_pairs: list,
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    fusion_alpha: float,
    k: int = TOP_K,
) -> float:
    if not loo_pairs:
        return 0.0

    n_users_in_model = model.user_factors.shape[0]
    ndcg_scores = []

    for uid, hidden_isin in loo_pairs:
        u_idx = imat.user_id_to_idx.get(uid)
        # Пропускаем если пользователь за пределами обученной модели
        # (может произойти если implicit обрезал user_factors)
        if u_idx is None or u_idx >= n_users_in_model:
            continue

        scores = compute_fusion_score(u_idx, model, imat, fusion_alpha)

        # Исключаем историю кроме hidden_isin
        for isin in set(imat.user_items(uid)) - {hidden_isin}:
            if isin in imat.item_id_to_idx:
                scores[imat.item_id_to_idx[isin]] = -np.inf

        # int() — np.argsort возвращает np.int64, dict ключи — int
        top_isins = [
            imat.idx_to_item_id[int(i)]
            for i in np.argsort(scores)[::-1]
            if int(i) in imat.idx_to_item_id
        ][:k]

        ndcg_scores.append(_ndcg_at_k(top_isins, hidden_isin, k))

    if not ndcg_scores:
        return 0.0
    return float(np.mean(ndcg_scores))



# Grid Search — Этап 1: ALS гиперпараметры
def grid_search_als(
    imat: InteractionMatrix,
    tx_window: pd.DataFrame,
    tx_val: pd.DataFrame,
    assets: pd.DataFrame,
) -> dict:
    loo_pairs = build_loo_pairs(imat, tx_val)
    if not loo_pairs:
        raise ValueError("Нет LOO-пар. Проверьте tx_val и MIN_TX_EVAL.")

    neutral_alpha = 0.5
    total = (
        len(ALS_FACTORS_GRID) * len(ALS_REGULARIZATION_GRID) *
        len(CONFIDENCE_ALPHA_GRID) * len(DECAY_LAMBDA_GRID)
    )
    logger.info(f"  Grid search Этап 1: {total} комбинаций")

    best_ndcg   = -1.0
    best_params = None
    results     = []

    for k, lam, conf_alpha, decay in product(
        ALS_FACTORS_GRID,
        ALS_REGULARIZATION_GRID,
        CONFIDENCE_ALPHA_GRID,
        DECAY_LAMBDA_GRID,
    ):
        item_meta_local = build_item_index(tx_window, assets)
        imat_local = build_interaction_matrix(
            tx_window        = tx_window,
            item_meta        = item_meta_local,
            snapshot_date    = imat.snapshot_date,
            window_days      = imat.window_days,
            confidence_alpha = conf_alpha,
            decay_lambda     = decay,
        )
        try:
            model_local = train_als(imat_local, factors=k, regularization=lam)
            # Строим loo_pairs от imat_local чтобы user_idx совпадали с моделью
            loo_local   = build_loo_pairs(imat_local, tx_val)
            ndcg        = evaluate_loo(loo_local, model_local, imat_local, neutral_alpha)
        except Exception as e:
            logger.debug(f"  k={k} λ={lam} conf={conf_alpha} d={decay}: {e}")
            continue

        results.append({
            "factors": k, "regularization": lam,
            "confidence_alpha": conf_alpha, "decay_lambda": decay,
            "ndcg3": ndcg,
        })

        if ndcg > best_ndcg:
            best_ndcg   = ndcg
            best_params = {
                "best_factors":          k,
                "best_regularization":   lam,
                "best_confidence_alpha": conf_alpha,
                "best_decay_lambda":     decay,
                "best_ndcg_stage1":      ndcg,
            }
            logger.info(
                f"  Лучший: k={k} λ={lam} conf={conf_alpha} "
                f"decay={decay} → NDCG@3={ndcg:.4f}"
            )

    if best_params is None:
        logger.warning("  Все комбинации упали с ошибкой. Используем дефолт.")
        best_params = {
            "best_factors":          ALS_FACTORS_GRID[2],
            "best_regularization":   ALS_REGULARIZATION_GRID[0],
            "best_confidence_alpha": CONFIDENCE_ALPHA_GRID[1],
            "best_decay_lambda":     DECAY_LAMBDA_GRID[1],
            "best_ndcg_stage1":      0.0,
        }

    logger.info(f"  Итог Этап 1: {best_params}")

    if best_params["best_ndcg_stage1"] > 0:
        _run_sensitivity_check(
            best_params, tx_window, imat, assets, tx_val, neutral_alpha
        )

    return {**best_params, "als_grid_results": results}


def _run_sensitivity_check(
    best_params: dict,
    tx_window: pd.DataFrame,
    imat_ref: InteractionMatrix,
    assets: pd.DataFrame,
    tx_val: pd.DataFrame,
    fusion_alpha: float,
):
    best_k     = best_params["best_factors"]
    best_lam   = best_params["best_regularization"]
    best_conf  = best_params["best_confidence_alpha"]
    best_decay = best_params["best_decay_lambda"]
    best_ndcg  = best_params["best_ndcg_stage1"]

    k_idx     = ALS_FACTORS_GRID.index(best_k) if best_k in ALS_FACTORS_GRID else -1
    neighbors = []
    if k_idx > 0:
        neighbors.append(ALS_FACTORS_GRID[k_idx - 1])
    if 0 <= k_idx < len(ALS_FACTORS_GRID) - 1:
        neighbors.append(ALS_FACTORS_GRID[k_idx + 1])

    if not neighbors:
        return

    logger.info(f"  Sensitivity check: k={neighbors} vs k*={best_k}")
    for k_n in neighbors:
        try:
            item_meta_local = build_item_index(tx_window, assets)
            imat_local = build_interaction_matrix(
                tx_window        = tx_window,
                item_meta        = item_meta_local,
                snapshot_date    = imat_ref.snapshot_date,
                window_days      = imat_ref.window_days,
                confidence_alpha = best_conf,
                decay_lambda     = best_decay,
            )
            model_n   = train_als(imat_local, factors=k_n, regularization=best_lam)
            loo_local = build_loo_pairs(imat_local, tx_val)
            ndcg_n    = evaluate_loo(loo_local, model_n, imat_local, fusion_alpha)
            delta     = ndcg_n - best_ndcg
            flag      = " ⚠ ЛУЧШЕ СОСЕДНИЙ" if delta > 0.005 else ""
            logger.info(f"    k={k_n}: NDCG@3={ndcg_n:.4f} (Δ={delta:+.4f}){flag}")
        except Exception as e:
            logger.debug(f"  Sensitivity k={k_n}: {e}")



# Grid Search — Этап 2: FUSION_ALPHA
def grid_search_fusion(
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    loo_pairs: list,
) -> dict:
    logger.info(f"  Grid search Этап 2: {len(FUSION_ALPHA_GRID)} значений fusion_alpha")

    best_ndcg         = -1.0
    best_fusion_alpha = 0.5
    results           = []

    for alpha in FUSION_ALPHA_GRID:
        ndcg = evaluate_loo(loo_pairs, model, imat, alpha)
        results.append({"fusion_alpha": alpha, "ndcg3": ndcg})
        if ndcg > best_ndcg:
            best_ndcg         = ndcg
            best_fusion_alpha = alpha
            logger.info(f"  Лучший: fusion_alpha={alpha:.1f} → NDCG@3={ndcg:.4f}")

    return {
        "best_fusion_alpha":   best_fusion_alpha,
        "best_ndcg_stage2":    best_ndcg,
        "fusion_grid_results": results,
    }



# Сохранение / загрузка
def save_als_model(
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    weights: dict,
    snapshot_date: pd.Timestamp,
) -> Path:
    date_str     = snapshot_date.strftime("%Y%m%d")
    npz_path     = MODEL_DIR / f"als_model_{date_str}.npz"
    map_path     = MODEL_DIR / f"als_maps_{date_str}.pkl"
    weights_path = MODEL_DIR / f"rec_weights_{date_str}.json"

    np.savez_compressed(
        npz_path,
        user_factors = model.user_factors,
        item_factors = model.item_factors,
    )
    with open(map_path, "wb") as f:
        pickle.dump({
            "user_id_to_idx": imat.user_id_to_idx,
            "item_id_to_idx": imat.item_id_to_idx,
            "idx_to_user_id": imat.idx_to_user_id,
            "idx_to_item_id": imat.idx_to_item_id,
        }, f)

    # Конвертируем numpy-типы перед json.dump
    def _to_python(v):
        if isinstance(v, np.integer):  return int(v)
        if isinstance(v, np.floating): return float(v)
        return v

    with open(weights_path, "w") as f:
        json.dump({k: _to_python(v) for k, v in weights.items()}, f, indent=2)

    logger.info(f"  Модель: {npz_path}")
    logger.info(f"  Маппинги: {map_path}")
    logger.info(f"  Веса: {weights_path}")
    return npz_path


def load_als_model(snapshot_date: pd.Timestamp) -> tuple:
    date_str     = snapshot_date.strftime("%Y%m%d")
    npz_path     = MODEL_DIR / f"als_model_{date_str}.npz"
    map_path     = MODEL_DIR / f"als_maps_{date_str}.pkl"
    weights_path = MODEL_DIR / f"rec_weights_{date_str}.json"

    for p in [npz_path, map_path, weights_path]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден: {p}")

    data = np.load(npz_path)
    with open(map_path, "rb") as f:
        maps = pickle.load(f)
    with open(weights_path) as f:
        weights = json.load(f)

    model              = AlternatingLeastSquares(factors=data["user_factors"].shape[1])
    model.user_factors = data["user_factors"]
    model.item_factors = data["item_factors"]

    logger.info(f"  Модель загружена: {npz_path}")
    return model, maps, weights




# EASE Model
class EASEModel:
    def __init__(self, regularization: float = 500.0):
        self.regularization = regularization
        self.W: np.ndarray | None = None

    def fit(self, imat: "InteractionMatrix") -> "EASEModel":
        X = imat.matrix.toarray().astype(np.float64)
        G = X.T @ X
        diag_indices = np.arange(G.shape[0])
        G[diag_indices, diag_indices] += self.regularization  # G + λI

        P = np.linalg.inv(G)
        W = np.eye(P.shape[0]) - P / np.diag(P)
        W[diag_indices, diag_indices] = 0.0 

        self.W = W.astype(np.float32)
        return self

    def score_user_from_matrix(
        self,
        u_idx: int,
        imat: "InteractionMatrix",
    ) -> np.ndarray:
        row    = imat.matrix.getrow(u_idx)
        h_idx  = row.indices
        c_vals = row.data

        if len(h_idx) == 0:
            return np.zeros(imat.n_items)

        # Взвешенная сумма строк W
        scores = c_vals @ self.W[h_idx, :]  # (M,)
        return scores

    def score_user_from_history(
        self,
        h_isins: list,
        imat: "InteractionMatrix",
    ) -> np.ndarray:
        if not h_isins or self.W is None:
            return np.zeros(imat.n_items)

        h_idx = [imat.item_id_to_idx[isin] for isin in h_isins
                 if isin in imat.item_id_to_idx]
        if not h_idx:
            return np.zeros(imat.n_items)

        return self.W[h_idx, :].sum(axis=0)  # (M,)


def train_ease(
    imat: InteractionMatrix,
    regularization: float,
) -> EASEModel:
    model = EASEModel(regularization=regularization)
    model.fit(imat)
    logger.info(
        f"  EASE: λ={regularization} | "
        f"W shape={model.W.shape} | "
        f"W max={model.W.max():.4f} W min={model.W.min():.4f}"
    )
    return model


def evaluate_loo_ease(
    loo_pairs: list,
    ease_model: EASEModel,
    imat: InteractionMatrix,
    k: int = TOP_K,
) -> float:
    if not loo_pairs or ease_model.W is None:
        return 0.0

    ndcg_scores = []
    for uid, hidden_isin in loo_pairs:
        u_idx = imat.user_id_to_idx.get(uid)
        if u_idx is None or u_idx >= imat.n_users:
            ndcg_scores.append(0.0)
            continue

        scores = ease_model.score_user_from_matrix(u_idx, imat)

        # Исключаем историю кроме hidden_isin
        for isin in set(imat.user_items(uid)) - {hidden_isin}:
            if isin in imat.item_id_to_idx:
                scores[imat.item_id_to_idx[isin]] = -np.inf

        top_isins = [
            imat.idx_to_item_id[int(i)]
            for i in np.argsort(scores)[::-1]
            if int(i) in imat.idx_to_item_id
        ][:k]

        ndcg_scores.append(_ndcg_at_k(top_isins, hidden_isin, k))

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def grid_search_ease(
    imat: InteractionMatrix,
    loo_pairs: list,
) -> dict:
    logger.info(
        f"  Grid search EASE: {len(EASE_REGULARIZATION_GRID)} значений λ "
        f"= {EASE_REGULARIZATION_GRID}"
    )

    best_ndcg = -1.0
    best_lam  = EASE_REGULARIZATION_GRID[0]
    results   = []

    for lam in EASE_REGULARIZATION_GRID:
        try:
            ease = train_ease(imat, regularization=lam)
            ndcg = evaluate_loo_ease(loo_pairs, ease, imat)
        except Exception as e:
            logger.debug(f"  EASE λ={lam}: {e}")
            continue

        results.append({"regularization": lam, "ndcg3": ndcg})
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_lam  = lam
            logger.info(f"  EASE лучший: λ={lam} → NDCG@3={ndcg:.4f}")

    logger.info(f"  EASE итог: λ={best_lam} | NDCG@3={best_ndcg:.4f}")
    return {
        "best_ease_regularization": best_lam,
        "best_ease_ndcg3":          best_ndcg,
        "ease_grid_results":        results,
    }


def save_ease_model(
    ease_model: EASEModel,
    snapshot_date: pd.Timestamp,
) -> Path:
    date_str  = snapshot_date.strftime("%Y%m%d")
    ease_path = MODEL_DIR / f"ease_model_{date_str}.npz"
    ease_meta = MODEL_DIR / f"ease_weights_{date_str}.json"

    np.savez_compressed(ease_path, W=ease_model.W)
    with open(ease_meta, "w") as f:
        json.dump({"regularization": ease_model.regularization}, f)

    logger.info(f"  EASE модель: {ease_path}")
    return ease_path


def load_ease_model(snapshot_date: pd.Timestamp) -> EASEModel:
    date_str  = snapshot_date.strftime("%Y%m%d")
    ease_path = MODEL_DIR / f"ease_model_{date_str}.npz"
    ease_meta = MODEL_DIR / f"ease_weights_{date_str}.json"

    for p in [ease_path, ease_meta]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден: {p}")

    data  = np.load(ease_path)
    with open(ease_meta) as f:
        meta = json.load(f)

    model   = EASEModel(regularization=meta["regularization"])
    model.W = data["W"]
    logger.info(f"  EASE модель загружена: {ease_path}")
    return model


def build_tx_without_loo_pairs(
    tx_window: pd.DataFrame,
    loo_pairs: list,
) -> pd.DataFrame:
    if not loo_pairs:
        return tx_window.copy()

    loo_df         = pd.DataFrame(loo_pairs, columns=["customerID", "ISIN"])
    loo_df["_loo"] = True

    merged  = tx_window.reset_index(drop=True).merge(loo_df, on=["customerID", "ISIN"], how="left")
    cleaned = tx_window.reset_index(drop=True)[merged["_loo"].isna()].copy()

    logger.info(
        f"  build_tx_without_loo_pairs: удалено "
        f"{len(tx_window) - len(cleaned)} транзакций "
        f"({len(loo_pairs)} LOO-пар)"
    )
    return cleaned


def run_ease_training(
    imat: InteractionMatrix,
    tx_window: pd.DataFrame,
    tx_val: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    assets: pd.DataFrame,
) -> tuple:
    logger.info("─" * 60)
    logger.info("  ОБУЧЕНИЕ EASE МОДЕЛИ (без LOO-пар в матрице)")
    logger.info("─" * 60)

    # LOO-пары строим от полной imat — те же что для ALS
    loo_pairs = build_loo_pairs(imat, tx_val)

    # Строим чистый tx_window без LOO-пар
    tx_clean = build_tx_without_loo_pairs(tx_window, loo_pairs)

    # Чистая матрица: те же параметры что у ALS, только без LOO-транзакций
    item_meta_clean = build_item_index(tx_clean, assets)
    imat_clean = build_interaction_matrix(
        tx_window        = tx_clean,
        item_meta        = item_meta_clean,
        snapshot_date    = imat.snapshot_date,
        window_days      = imat.window_days,
        confidence_alpha = imat.confidence_alpha,
        decay_lambda     = imat.decay_lambda,
    )

    logger.info(
        f"  imat       : {imat.n_users:,} users × {imat.n_items:,} items "
        f"| {imat.n_interactions:,} пар"
    )
    logger.info(
        f"  imat_clean : {imat_clean.n_users:,} users × {imat_clean.n_items:,} items "
        f"| {imat_clean.n_interactions:,} пар "
        f"(удалено {imat.n_interactions - imat_clean.n_interactions:,} LOO-пар)"
    )

    # Grid search и обучение на чистой матрице, оценка на тех же loo_pairs
    ease_result = grid_search_ease(imat_clean, loo_pairs)
    best_lam    = ease_result["best_ease_regularization"]
    ease_model  = train_ease(imat_clean, regularization=best_lam)
    save_ease_model(ease_model, snapshot_date)

    return ease_model, ease_result, imat_clean




# Точка входа
def run_training(
    imat: InteractionMatrix,
    tx_window: pd.DataFrame,
    tx_val: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    assets: pd.DataFrame,
) -> tuple:
    logger.info("═" * 60)
    logger.info("  ОБУЧЕНИЕ RECOMMENDER МОДЕЛИ")
    logger.info("═" * 60)

    stage1 = grid_search_als(imat, tx_window, tx_val, assets)

    best_k     = stage1["best_factors"]
    best_lam   = stage1["best_regularization"]
    best_conf  = stage1["best_confidence_alpha"]
    best_decay = stage1["best_decay_lambda"]

    item_meta_final = build_item_index(tx_window, assets)
    imat_final = build_interaction_matrix(
        tx_window        = tx_window,
        item_meta        = item_meta_final,
        snapshot_date    = snapshot_date,
        confidence_alpha = best_conf,
        decay_lambda     = best_decay,
    )
    model     = train_als(imat_final, factors=best_k, regularization=best_lam)
    loo_pairs = build_loo_pairs(imat_final, tx_val)
    stage2    = grid_search_fusion(model, imat_final, loo_pairs)

    best_fusion = stage2["best_fusion_alpha"]

    weights = {
        "snapshot_date":       snapshot_date.strftime("%Y-%m-%d"),
        "als_factors":         int(best_k),
        "als_regularization":  float(best_lam),
        "confidence_alpha":    float(best_conf),
        "decay_lambda":        float(best_decay),
        "fusion_alpha":        float(best_fusion),
        "ndcg3_stage1":        float(stage1["best_ndcg_stage1"]),
        "ndcg3_stage2":        float(stage2["best_ndcg_stage2"]),
        "n_users":             int(imat_final.n_users),
        "n_items":             int(imat_final.n_items),
        "matrix_density":      round(imat_final.density, 6),
        "als_grid_results":    stage1.get("als_grid_results", []),
        "fusion_grid_results": stage2.get("fusion_grid_results", []),
    }

    save_als_model(model, imat_final, weights, snapshot_date)

    logger.info("═" * 60)
    logger.info(f"  k={best_k} λ={best_lam} conf={best_conf} "
                f"decay={best_decay} fusion={best_fusion:.1f}")
    logger.info(f"  NDCG@3 validation: {stage2['best_ndcg_stage2']:.4f}")
    logger.info("═" * 60)

    return model, imat_final, weights
