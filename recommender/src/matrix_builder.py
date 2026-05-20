import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import (
    INTERACTION_WINDOW_DAYS,
    MIN_ITEM_SUPPORT,
)

logger = logging.getLogger(__name__)


@dataclass
class InteractionMatrix:

    matrix: sp.csr_matrix
    user_id_to_idx: dict[str, int]
    item_id_to_idx: dict[str, int]
    idx_to_user_id: dict[int, str]
    idx_to_item_id: dict[int, str]
    item_meta: pd.DataFrame
    snapshot_date: pd.Timestamp
    window_days: int
    confidence_alpha: float = 20.0
    decay_lambda: float = 1.0
    n_interactions: int = field(init=False)
    density: float = field(init=False)

    def __post_init__(self):
        n_u, n_i   = self.matrix.shape
        self.n_interactions = self.matrix.nnz
        self.density = self.n_interactions / (n_u * n_i) if n_u * n_i > 0 else 0.0

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    def user_items(self, user_id: str) -> list[str]:
        if user_id not in self.user_id_to_idx:
            return []
        u_idx = self.user_id_to_idx[user_id]
        row   = self.matrix.getrow(u_idx)
        item_idxs = row.indices
        return [self.idx_to_item_id[i] for i in item_idxs]


def _compute_confidence(
    pair_stats: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    confidence_alpha: float,
    decay_lambda: float,
) -> pd.Series:

    days_ago = (snapshot_date - pair_stats["last_date"]).dt.days.clip(lower=0)

    if decay_lambda > 0:
        time_weight = np.exp(-decay_lambda * days_ago / 365.0)
    else:
        time_weight = np.ones(len(pair_stats))

    confidence = 1.0 + confidence_alpha * np.log1p(pair_stats["count"]) * time_weight
    return confidence


def build_interaction_matrix(
    tx_window: pd.DataFrame,
    item_meta: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window_days: int = INTERACTION_WINDOW_DAYS,
    confidence_alpha: float = 20.0,
    decay_lambda: float = 1.0,
) -> InteractionMatrix:

    valid_isins = set(item_meta.index)
    tx_filtered = tx_window[tx_window["ISIN"].isin(valid_isins)].copy()

    if tx_filtered.empty:
        raise ValueError(
            "После фильтрации по item_meta нет транзакций. "
            "Проверьте параметры window_days и MIN_ITEM_SUPPORT."
        )

    pair_stats = (
        tx_filtered.groupby(["customerID", "ISIN"])
        .agg(count=("ISIN", "count"), last_date=("timestamp", "max"))
        .reset_index()
    )

    pair_stats["confidence"] = _compute_confidence(
        pair_stats, snapshot_date, confidence_alpha, decay_lambda
    )

    user_ids = sorted(pair_stats["customerID"].unique())
    item_ids = list(item_meta.index)          # порядок из item_meta

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    rows = pair_stats["customerID"].map(user_id_to_idx).values
    cols = pair_stats["ISIN"].map(item_id_to_idx).values
    data = pair_stats["confidence"].values.astype(np.float32)

    matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    ).tocsr()

    imat = InteractionMatrix(
        matrix          = matrix,
        user_id_to_idx  = user_id_to_idx,
        item_id_to_idx  = item_id_to_idx,
        idx_to_user_id  = {int(i): uid for uid, i in user_id_to_idx.items()},
        idx_to_item_id  = {int(i): iid for iid, i in item_id_to_idx.items()},
        item_meta       = item_meta,
        snapshot_date   = snapshot_date,
        window_days     = window_days,
    )

    imat.confidence_alpha = confidence_alpha
    imat.decay_lambda     = decay_lambda

    logger.info(
        f"  InteractionMatrix: {imat.n_users:,} users × {imat.n_items:,} items "
        f"| {imat.n_interactions:,} пар | density={imat.density:.4%} "
        f"| confidence_alpha={confidence_alpha}, decay_lambda={decay_lambda}"
    )
    return imat


def get_user_pair_counts(
    tx_window: pd.DataFrame,
    user_id: str,
) -> dict[str, int]:
    user_tx = tx_window[tx_window["customerID"] == user_id]
    return user_tx.groupby("ISIN").size().to_dict()
