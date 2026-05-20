import logging
from typing import Literal, Optional
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

from config import (
    INTERACTION_WINDOW_DAYS,
    EXTENDED_WINDOW_DAYS,
    MIN_TX_REC,
    EXCLUDE_RECENT_DAYS,
    TOP_K,
    PORTFOLIO_BOOST_GAMMA,
    PORTFOLIO_BOOST_MIN_CATS,
    RISK_FALLBACK_CATEGORY,
    OUTPUT_DIR,
    OUTPUT_COLS,
)
from train import (
    compute_fusion_score,
    _percentile_rank_normalize,
    EASEModel,
)
from matrix_builder import InteractionMatrix

logger = logging.getLogger(__name__)

REC_DIR = OUTPUT_DIR / "recommendations"
REC_DIR.mkdir(parents=True, exist_ok=True)


WindowLabel = Literal["365d", "730d", "full", "none"]


def _get_history_window(
    tx_all: pd.DataFrame,
    uid: str,
    snapshot_date: pd.Timestamp,
    als_item_set: set[str],
    window_days: int,
) -> list[str]:
    cutoff = snapshot_date - pd.Timedelta(days=window_days)
    mask   = (
        (tx_all["customerID"] == uid) &
        (tx_all["timestamp"] >= cutoff) &
        (tx_all["timestamp"] < snapshot_date) &
        (tx_all["ISIN"].isin(als_item_set))
    )
    return tx_all.loc[mask, "ISIN"].unique().tolist()


def build_client_profile(
    tx_all: pd.DataFrame,
    uid: str,
    snapshot_date: pd.Timestamp,
    als_item_set: set[str],
) -> tuple[list[str], WindowLabel]:

    for window_days, label in [
            (INTERACTION_WINDOW_DAYS, "365d"),
            (EXTENDED_WINDOW_DAYS,    "730d"),
            (3650,                    "full"),
    ]:    
        h_u = _get_history_window(tx_all, uid, snapshot_date, als_item_set, window_days)
        if len(h_u) >= MIN_TX_REC:
            return h_u, label

    return [], "none"



# Шаг 4 — ALS scoring
def _compute_user_scores_from_history(
    h_u: list[str],
    u_idx: Optional[int],
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    fusion_alpha: float,
) -> np.ndarray:
    V = model.item_factors  # (n_items, k)
    n_items = imat.n_items

    # Item-based score: mean_{j in H_u} (v_j · v_i)
    h_idx = [imat.item_id_to_idx[isin] for isin in h_u
             if isin in imat.item_id_to_idx]
    if h_idx:
        mean_vec = V[h_idx].mean(axis=0)   # (k,)
        s_item   = V @ mean_vec             # (n_items,)
    else:
        s_item = np.zeros(n_items)

    # User-based score: u_u · v_i
    if model is not None and u_idx is not None and u_idx < model.user_factors.shape[0]:
        U = model.user_factors
        u_vec  = U[u_idx]
        s_user = V @ u_vec
    else:
        # Клиент новый для матрицы: s_user = 0, полагаемся только на item-CF
        s_user = np.zeros(n_items)
        fusion_alpha = 1.0

    # Percentile rank нормализация (Koren et al., 2009)
    s_item_norm  = _percentile_rank_normalize(s_item)
    s_user_norm  = _percentile_rank_normalize(s_user)

    return fusion_alpha * s_item_norm + (1 - fusion_alpha) * s_user_norm


def _exclude_recently_bought(
    scores: np.ndarray,
    uid: str,
    tx_all: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    imat: InteractionMatrix,
) -> np.ndarray:
    cutoff   = snapshot_date - pd.Timedelta(days=EXCLUDE_RECENT_DAYS)
    recent   = tx_all[
        (tx_all["customerID"] == uid) &
        (tx_all["timestamp"] >= cutoff) &
        (tx_all["timestamp"] < snapshot_date)
    ]["ISIN"].unique()

    for isin in recent:
        if isin in imat.item_id_to_idx:
            scores[imat.item_id_to_idx[isin]] = -np.inf
    return scores



# Шаг 5 — Diversity корректировка
def _apply_portfolio_boost(
    scores: np.ndarray,
    h_u_isins: list[str],
    imat: InteractionMatrix,
    gamma: float = PORTFOLIO_BOOST_GAMMA,
) -> np.ndarray:
    cat_counts: dict[str, int] = {}
    for isin in h_u_isins:
        if isin in imat.item_meta.index:
            cat = imat.item_meta.loc[isin, "assetCategory"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    n_cats = len(cat_counts)
    if n_cats < PORTFOLIO_BOOST_MIN_CATS:
        return scores

    # Находим категорию с наименьшей долей
    total    = sum(cat_counts.values())
    cat_share = {c: v / total for c, v in cat_counts.items()}
    min_cat  = min(cat_share, key=cat_share.get)

    # Бустим ISINs из недостающей категории
    boosted = scores.copy()

    for i, isin in imat.idx_to_item_id.items():
        if i >= len(boosted):      # ← ISIN не входит в матрицу EASE
            continue
        if isin in imat.item_meta.index:
            if imat.item_meta.loc[isin, "assetCategory"] == min_cat:
                if boosted[i] > -np.inf:
                    boosted[i] *= gamma

    return boosted


def _select_diverse_top_k(
    scores: np.ndarray,
    imat: InteractionMatrix,
    k: int = TOP_K,
) -> list[int]:
    # Полный ранжированный список доступных кандидатов
    ranked_idx   = np.argsort(scores)[::-1]
    valid_ranked = [i for i in ranked_idx if scores[i] > -np.inf]

    if len(valid_ranked) <= k:
        return valid_ranked

    # Определяем preferred_category (у большинства — одна)
    # Берём из топ-1 кандидата
    top1_isin = imat.idx_to_item_id[valid_ranked[0]]
    pref_cat  = (
        imat.item_meta.loc[top1_isin, "assetCategory"]
        if top1_isin in imat.item_meta.index
        else None
    )

    # Проверяем доступность sector-данных
    meta = imat.item_meta
    has_sector = (
        pref_cat is not None and
        meta["assetCategory"].eq(pref_cat).any() and
        meta.loc[meta["assetCategory"] == pref_cat, "sector"].notna().any()
    )

    if has_sector:
        # Стратегия 1: по секторам — top-1 из каждого из k различных секторов
        chosen   = []
        used_sec = set()

        for idx in valid_ranked:
            if len(chosen) >= k:
                break
            isin = imat.idx_to_item_id[idx]
            if isin not in meta.index:
                chosen.append(idx)
                continue
            sector = meta.loc[isin, "sector"]
            if pd.isna(sector) or sector not in used_sec:
                chosen.append(idx)
                if not pd.isna(sector):
                    used_sec.add(sector)

        # Дополняем если не хватает (rare case)
        if len(chosen) < k:
            for idx in valid_ranked:
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= k:
                    break
        return chosen[:k]

    else:
        # Стратегия 2: weighted random по percentile rank из топ-N кандидатов
        pool_size = min(len(valid_ranked), max(k * 5, 20))
        pool      = valid_ranked[:pool_size]
        weights   = np.array([scores[i] for i in pool])
        weights   = _percentile_rank_normalize(weights) + 1e-9  # избегаем 0
        weights  /= weights.sum()

        chosen_set = set()
        chosen     = []
        # Не полагаемся на replace=False при маленьком pool
        attempts = 0
        while len(chosen) < k and attempts < pool_size * 3:
            idx = np.random.choice(pool, p=weights)
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
            attempts += 1

        if len(chosen) < k:
            for idx in pool:
                if idx not in chosen_set:
                    chosen.append(idx)
                if len(chosen) >= k:
                    break
        return chosen[:k]



# Fallback для холодных клиентов
def _popularity_fallback(
    uid: str,
    tx_all: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    imat: InteractionMatrix,
    profile: pd.Series,
    k: int = TOP_K,
) -> list[tuple[str, str]]:
    # Preferred category
    user_hist = tx_all[
        (tx_all["customerID"] == uid) &
        (tx_all["timestamp"] < snapshot_date)
    ]
    if not user_hist.empty:
        cat_counts = (
            user_hist.merge(imat.item_meta.reset_index()[["ISIN","assetCategory"]],
                            on="ISIN", how="left")
            .groupby("assetCategory").size()
            .sort_values(ascending=False)
        )
        pref_cat = cat_counts.index[0] if not cat_counts.empty else "Stock"
    else:
        risk = profile.get("riskLevel", "Not_Available") if profile is not None else "Not_Available"
        pref_cat = RISK_FALLBACK_CATEGORY.get(risk, "Stock")

    # Пул ISINs нужной категории из ALS item space
    meta_cat = imat.item_meta.reset_index()
    cat_pool = meta_cat[meta_cat["assetCategory"] == pref_cat].copy()
    if cat_pool.empty:
        # Фолбэк на весь item space
        cat_pool = meta_cat.copy()

    # Исключаем недавно купленные
    cutoff     = snapshot_date - pd.Timedelta(days=EXCLUDE_RECENT_DAYS)
    recent_set = set(
        tx_all[
            (tx_all["customerID"] == uid) &
            (tx_all["timestamp"] >= cutoff)
        ]["ISIN"]
    )
    cat_pool = cat_pool[~cat_pool["ISIN"].isin(recent_set)].copy()

    if cat_pool.empty:
        return []

    # log-weight для popular diversity
    cat_pool["log_weight"] = np.log1p(cat_pool["n_buyers"])

    has_sector = cat_pool["sector"].notna().any()
    chosen_isins = []

    if has_sector:
        # Стратегия 1: top-1 из каждого из k секторов
        top_sectors = (
            cat_pool.dropna(subset=["sector"])
            .groupby("sector")["n_buyers"].sum()
            .nlargest(k).index.tolist()
        )
        used_sec = set()
        # Сначала из топ-секторов
        for sector in top_sectors:
            sec_pool = cat_pool[cat_pool["sector"] == sector]
            if sec_pool.empty or sector in used_sec:
                continue
            best_row = sec_pool.nlargest(1, "n_buyers").iloc[0]
            chosen_isins.append((best_row["ISIN"], "popular_sector"))
            used_sec.add(sector)
            if len(chosen_isins) >= k:
                break

        # Дополняем без сектора если нужно
        if len(chosen_isins) < k:
            no_sec_pool = cat_pool[cat_pool["sector"].isna()]
            for _, row in no_sec_pool.nlargest(k, "n_buyers").iterrows():
                if row["ISIN"] not in {x[0] for x in chosen_isins}:
                    chosen_isins.append((row["ISIN"], "popular_category"))
                if len(chosen_isins) >= k:
                    break
    else:
        # Стратегия 2: weighted random по log_weight
        weights = cat_pool["log_weight"].values
        weights /= weights.sum()
        pool_isins = cat_pool["ISIN"].values

        chosen_set = set()
        attempts   = 0
        while len(chosen_isins) < k and attempts < len(pool_isins) * 3:
            idx  = np.random.choice(len(pool_isins), p=weights)
            isin = pool_isins[idx]
            if isin not in chosen_set:
                chosen_isins.append((isin, "popular_category"))
                chosen_set.add(isin)
            attempts += 1

    return chosen_isins[:k]



# Шаг 6 — Формирование одной рекомендации
def _format_recommendation(
    uid: str,
    top_k_idx: list[int],
    scores: np.ndarray,
    imat: InteractionMatrix,
    h_u: list[str],
    fusion_alpha: float,
    u_idx: Optional[int],
    model_type: str = "als"
) -> dict:
    hist_cats = set()
    for isin in h_u:
        if isin in imat.item_meta.index:
            hist_cats.add(imat.item_meta.loc[isin, "assetCategory"])

    rec: dict = {}
    for rank, item_idx in enumerate(top_k_idx[:TOP_K], start=1):
        isin     = imat.idx_to_item_id[item_idx]
        category = (
            imat.item_meta.loc[isin, "assetCategory"]
            if isin in imat.item_meta.index else "Unknown"
        )
        score    = float(scores[item_idx])
        outside  = category not in hist_cats


        if model_type == "ease":
            just = "item_cf"
        elif fusion_alpha >= 0.7:
            just = "item_cf"
        elif fusion_alpha <= 0.3:
            just = "user_cf"
        else:
            just = "item_cf" if u_idx is None else "user_cf"

        rec[f"rank_{rank}_isin"]          = isin
        rec[f"rank_{rank}_category"]      = category
        rec[f"rank_{rank}_score"]         = round(score, 6)
        rec[f"rank_{rank}_justification"] = just
        rec[f"rank_{rank}_outside_hist"]  = outside

    return rec


def _format_fallback_recommendation(
    fallback_list: list[tuple[str, str]],
    imat: InteractionMatrix,
    h_u: list[str],
) -> dict:
    hist_cats = set()
    for isin in h_u:
        if isin in imat.item_meta.index:
            hist_cats.add(imat.item_meta.loc[isin, "assetCategory"])

    rec: dict = {}
    for rank, (isin, just) in enumerate(fallback_list[:TOP_K], start=1):
        category = (
            imat.item_meta.loc[isin, "assetCategory"]
            if isin in imat.item_meta.index else "Unknown"
        )
        outside = category not in hist_cats
        rec[f"rank_{rank}_isin"]          = isin
        rec[f"rank_{rank}_category"]      = category
        rec[f"rank_{rank}_score"]         = float("nan")
        rec[f"rank_{rank}_justification"] = just
        rec[f"rank_{rank}_outside_hist"]  = outside

    # Заполняем оставшиеся ранги если fallback вернул < TOP_K
    for rank in range(len(fallback_list) + 1, TOP_K + 1):
        rec[f"rank_{rank}_isin"]          = None
        rec[f"rank_{rank}_category"]      = None
        rec[f"rank_{rank}_score"]         = float("nan")
        rec[f"rank_{rank}_justification"] = None
        rec[f"rank_{rank}_outside_hist"]  = None

    return rec


def compute_ease_score_for_user(
    h_u: list,
    u_idx: int | None,
    ease_model: "EASEModel",
    imat: InteractionMatrix,
) -> np.ndarray:
    return ease_model.score_user_from_history(h_u, imat)





# Точка входа: inference для всех hot customers
def run_prediction(
    hot_customers: pd.DataFrame,
    model: AlternatingLeastSquares,
    imat: InteractionMatrix,
    weights: dict,
    tx_all: pd.DataFrame,
    customers_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    save: bool = True,
    ease_model: "EASEModel | None" = None,
) -> pd.DataFrame:
    from data_loader import get_customer_profile_at

    fusion_alpha  = weights["fusion_alpha"]
    als_item_set  = set(imat.item_id_to_idx.keys())
    profile_snap  = get_customer_profile_at(customers_df, snapshot_date)

    logger.info(f"  Inference для {len(hot_customers):,} клиентов "
                f"| fusion_alpha={fusion_alpha:.1f}")

    records = []
    n_als, n_fallback = 0, 0
    n_als_with_user_emb = 0
    n_als_item_only = 0

    for _, hot_row in hot_customers.iterrows():
        uid   = hot_row["customerID"]
        seg   = hot_row.get("segment", "unknown")
        score = hot_row.get("propensity_score", float("nan"))

        # Профиль клиента (для risk_profile_verified и fallback mapping)
        user_profile = profile_snap.loc[
            profile_snap["customerID"] == uid
        ].iloc[0] if uid in profile_snap["customerID"].values else None

        risk_verified = (
            not bool(user_profile["is_profile_predicted"])
            if user_profile is not None else False
        )

        base = {
            "customerID":           uid,
            "segment":              seg,
            "propensity_score":     score,
            "snapshot_date":        snapshot_date,
            "risk_profile_verified": risk_verified,
        }

        # Шаг 3: ступенчатое расширение окна
        h_u, window_label = build_client_profile(
            tx_all, uid, snapshot_date, als_item_set
        )
        base["window_used"]      = window_label
        base["n_history_isins"]  = len(h_u)

        # Определяем путь
        if window_label == "none" or not h_u:
            # Fallback path
            fallback_list = _popularity_fallback(
                uid, tx_all, snapshot_date, imat, user_profile
            )
            rec = _format_fallback_recommendation(fallback_list, imat, h_u)
            base["rec_type"] = "fallback"
            n_fallback += 1

        else:
            # ALS path
            u_idx = imat.user_id_to_idx.get(uid)  # None если клиент новый в матрице

            # Шаг 4: вычисление скоров
            # Если EASE доступна — используем её (лучшее item-item качество).
            # Иначе — ALS с percentile rank fusion.
            if ease_model is not None and ease_model.W is not None:
                scores = compute_ease_score_for_user(h_u, u_idx, ease_model, imat)
                rec_type_label = "ease" 
            else:
                scores = _compute_user_scores_from_history(
                    h_u, u_idx, model, imat, fusion_alpha,
                    rec_type_label = "als" 
                )

            # Исключаем недавно купленные
            scores = _exclude_recently_bought(
                scores, uid, tx_all, snapshot_date, imat
            )

            # Шаг 5: portfolio boost (только n_cat >= 2)
            scores = _apply_portfolio_boost(scores, h_u, imat)

            # Diversity: intra-category top-K
            top_k_idx = _select_diverse_top_k(scores, imat, k=TOP_K)

            if not top_k_idx:
                # Все кандидаты заблокированы → fallback
                fallback_list = _popularity_fallback(
                    uid, tx_all, snapshot_date, imat, user_profile
                )
                rec = _format_fallback_recommendation(fallback_list, imat, h_u)
                base["rec_type"] = "fallback"
                n_fallback += 1
            else:
                rec = _format_recommendation(
                    uid, top_k_idx, scores, imat, h_u, fusion_alpha, u_idx,
                    model_type="ease" if (ease_model is not None and ease_model.W is not None) else "als"
                )
                base["rec_type"] = rec_type_label
                n_als += 1
                if model is not None and u_idx is not None and u_idx < model.user_factors.shape[0]:
                    n_als_with_user_emb += 1
                else:
                    n_als_item_only += 1

        records.append({**base, **rec})

    df_out = pd.DataFrame(records)

    # Убеждаемся что все колонки присутствуют
    for col in OUTPUT_COLS:
        if col not in df_out.columns:
            df_out[col] = None

    df_out = df_out[OUTPUT_COLS]

    logger.info(
        f"  Результат: {n_als:,} ALS + {n_fallback:,} fallback "
        f"= {len(df_out):,} рекомендаций"
    )
    logger.info(
        f"  ALS детализация: {n_als_with_user_emb:,} с user embedding "
        f"+ {n_als_item_only:,} item-only (u_idx вне матрицы)"
    )

    if save:
        date_str = snapshot_date.strftime("%Y%m%d")
        path = REC_DIR / f"recommendations_{date_str}.csv"
        df_out.to_csv(path, index=False)
        logger.info(f"  Рекомендации сохранены: {path}")

    return df_out
