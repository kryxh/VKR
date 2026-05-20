import logging
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np

from config import (DATA_FILES, RISK_LEVEL_CANONICAL, INVESTMENT_CAPACITY_CANONICAL)
logger = logging.getLogger(__name__)


# Транзакции
def load_transactions(path: Path | None = None):
    
    path = path or DATA_FILES["transactions"]
    logger.info(f"Загрузка транзакций из {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Очистка
    n_raw = len(df)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="transactionID", keep="last")
    df["is_synthetic"] = (df["transactionID"] < 0).astype(int)
    df = df[df["totalValue"].notna() & (df["totalValue"] > 0)]
    df = df[df["transactionType"].isin(["Buy", "Sell"])]
    df = df[df["timestamp"].notna()]

    logger.info(f"Транзакции: {n_raw:,} → {len(df):,} очищенные (удалено {n_raw - len(df):,})")
    logger.info(f"Покупка/Продажа: {(df['transactionType'] == 'Buy').sum():,} / {(df['transactionType'] == 'Sell').sum():,}")
    return df.reset_index(drop=True)


# Клиенты
def load_customers(path: Path | None = None):

    path = path or DATA_FILES["customers"]
    logger.info(f"Загрузка данных клиентов из {path}")
    df = pd.read_csv(path, parse_dates=["lastQuestionnaireDate", "timestamp"])

    # Помечаем строки, где профильные поля были предсказаны алгоритмом
    is_predicted_risk = df["riskLevel"].str.startswith("Predicted_", na=False)
    is_predicted_capacity = df["investmentCapacity"].str.startswith("Predicted_", na=False)
    df["is_profile_predicted"] = (is_predicted_risk | is_predicted_capacity).astype(int)

    # Маппинг по riskLevel и investmentCapacity (приведение типов)
    df["riskLevel"] = (df["riskLevel"].fillna("Not_Available").map(RISK_LEVEL_CANONICAL).fillna("Not_Available"))
    df["investmentCapacity"] = (df["investmentCapacity"].fillna("Not_Available").map(INVESTMENT_CAPACITY_CANONICAL).fillna("Not_Available"))
    df["is_capacity_missing"] = (df["investmentCapacity"] == "Not_Available").astype(int)

    df["customerType"] = df["customerType"].fillna("Unknown")
    df = df.sort_values(["customerID", "timestamp"]).reset_index(drop=True)

    logger.info(f"  Клиенты: {len(df):,} rows, {df['customerID'].nunique():,} уникальных ID")
    return df


def get_customer_profile_at(customers_df: pd.DataFrame, snapshot_date: pd.Timestamp):
    
    past = customers_df[customers_df["timestamp"] <= snapshot_date].copy() # оставляем только записи, известные на момент T или раньше
    if past.empty:
        logger.warning(f"Нет записей клиента на или до {snapshot_date}")
        return pd.DataFrame(columns=customers_df.columns)
    past = past.sort_values("timestamp") # для каждого клиента берём последний доступный статус
    profile = past.groupby("customerID", sort=False).last().reset_index()

    return profile


# Активы
def load_assets(path: Path | None = None):

    path = path or DATA_FILES["assets"]
    logger.info(f"Загрузка активов из {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    n_raw = len(df)
    df = df.sort_values("timestamp")
    df = df.groupby("ISIN", sort=False).last().reset_index()

    logger.info(f"Assets: {n_raw:,} → {len(df):,} уникальных ISIN (категории: {df['assetCategory'].value_counts().to_dict()})")
    return df


# Цены закрытия
def load_close_prices(path: Path | None = None):
    
    path = path or DATA_FILES["prices"]
    logger.info(f"Загрузка цен закрытия из {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    
    n_raw = len(df)
    df = df[df["closePrice"].notna() & (df["closePrice"] > 0)]
    df = df.sort_values(["ISIN", "timestamp"]).reset_index(drop=True)

    logger.info(f"Цены: {n_raw:,} → {len(df):,} очищенные, {df['ISIN'].nunique():,} уникальных ISIN, за период {df['timestamp'].min().date()} – {df['timestamp'].max().date()}")
    return df


# Рынки: инфо
def load_markets(path: Path | None = None):

    path = path or DATA_FILES["markets"]
    logger.info(f"Загрузка данных по рынкам из {path}")
    df = pd.read_csv(path)

    logger.info(f"  Рынки: {len(df):,} записей")
    return df


# Загружаем всё сразу
def load_all():
    return {
        "transactions": load_transactions(),
        "customers": load_customers(),
        "assets": load_assets(),
        "prices": load_close_prices(),
        "markets": load_markets(),
    }
