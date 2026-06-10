import importlib.util
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from config import DATA_FILES, MIN_ITEM_SUPPORT

logger = logging.getLogger(__name__)


_PROPENSITY_SRC = (
    Path(__file__).resolve().parents[2] / "propensity_model" / "src"
)
_PROPENSITY_LOADER_PATH = _PROPENSITY_SRC / "data_loader.py"

if not _PROPENSITY_LOADER_PATH.exists():
    raise FileNotFoundError(
        f"Не найден propensity data_loader: {_PROPENSITY_LOADER_PATH}\n"
        f"Убедитесь, что recommender/ лежит рядом с propensity_model/"
    )


_prop_src_str = str(_PROPENSITY_SRC)
_path_added   = _prop_src_str not in sys.path
if _path_added:
    sys.path.insert(0, _prop_src_str)

import sys as _sys
_saved_config = _sys.modules.pop("config", None)

try:
    _spec = importlib.util.spec_from_file_location(
        "propensity_data_loader",
        _PROPENSITY_LOADER_PATH,
    )
    _pmod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pmod)
finally:
    if _saved_config is not None:
        _sys.modules["config"] = _saved_config
    elif "config" in _sys.modules:
        del _sys.modules["config"]
    if _path_added and _prop_src_str in sys.path:
        sys.path.remove(_prop_src_str)
_load_tx = _pmod.load_transactions
_load_cust = _pmod.load_customers
_load_assets = _pmod.load_assets
get_customer_profile_at = _pmod.get_customer_profile_at



def load_transactions() -> pd.DataFrame:
    tx = _load_tx(DATA_FILES["transactions"])
    before = len(tx)

    tx = tx[
        (tx["transactionType"] == "Buy") &
        (tx["transactionID"] >= 0)
    ].copy()

    logger.info(
        f"  CF Buy-транзакции: {len(tx):,} из {before:,} "
        f"(клиентов: {tx['customerID'].nunique():,}, "
        f"ISIN: {tx['ISIN'].nunique():,})"
    )
    return tx.reset_index(drop=True)


def load_customers() -> pd.DataFrame:
    return _load_cust(DATA_FILES["customers"])


def load_assets() -> pd.DataFrame:
    return _load_assets(DATA_FILES["assets"])



def get_buy_history(
    tx: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    cutoff = snapshot_date - pd.Timedelta(days=window_days)
    mask = (tx["timestamp"] >= cutoff) & (tx["timestamp"] < snapshot_date)
    return tx[mask].copy()


def build_item_index(
    tx_window: pd.DataFrame,
    assets: pd.DataFrame,
    min_support: int = MIN_ITEM_SUPPORT,
) -> pd.DataFrame:
    item_support = (
        tx_window.groupby("ISIN")["customerID"]
        .nunique()
        .reset_index(name="n_buyers")
    )
    item_support = item_support[item_support["n_buyers"] >= min_support]

    asset_cols = ["ISIN", "assetCategory"]
    if "sector" in assets.columns:
        asset_cols.append("sector")

    item_meta = item_support.merge(
        assets[asset_cols],
        on="ISIN",
        how="left",
    )

    if "sector" not in item_meta.columns:
        item_meta["sector"] = None

    n_all = tx_window["ISIN"].nunique()
    n_kept = len(item_meta)
    n_dropped = n_all - n_kept
    logger.info(
        f"  Item index: {n_kept} ISINs "
        f"(удалено {n_dropped} с n_buyers < {min_support})"
    )
    return item_meta.set_index("ISIN")


def get_hot_customers(snapshot_date: pd.Timestamp) -> pd.DataFrame:
    from config import PROPENSITY_PREDICT_DIR

    date_str = snapshot_date.strftime("%Y%m%d")
    hot_path = PROPENSITY_PREDICT_DIR / f"hot_customers_{date_str}.csv"

    if not hot_path.exists():
        raise FileNotFoundError(
            f"Файл hot_customers не найден: {hot_path}\n"
            f"Сначала запустите propensity модель:\n"
            f"  cd ../propensity_model\n"
            f"  python main.py --predict-only --snapshot-date "
            f"{snapshot_date.date()}"
        )

    df = pd.read_csv(hot_path)
    required = {"customerID", "propensity_score", "segment"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"В hot_customers отсутствуют колонки: {missing}")

    df["snapshot_date"] = snapshot_date
    logger.info(
        f"  Hot customers загружены: {len(df):,} клиентов ({hot_path.name})"
    )
    return df
