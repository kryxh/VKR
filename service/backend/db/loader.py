import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import random

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

# Добавляем родительскую директорию в path для импорта из backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.models import (
    Advisor,
    AdvisorClient,
    Asset,
    ClosePrice,
    Customer,
    Recommendation,
    ScoringResult,
    Transaction,
)
from db.session import SessionLocal, create_tables


# Пути к данным (монтируются через Docker volumes)
# Базовая директория бэкенда (/app внутри контейнера)
_APP_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = _APP_DIR / "data" / "raw"
PROPENSITY_DIR = _APP_DIR / "data" / "propensity"
RECOMMENDER_DIR = _APP_DIR / "data" / "recommender"


# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



def _get_snapshot_date() -> str:
    env_date = os.environ.get("SNAPSHOT_DATE")
    if env_date:
        logger.info(f"[Loader] snapshot_date из ENV: {env_date}")
        return env_date.replace("-", "")

    # Ищем файлы hot_customers_YYYYMMDD.csv и берём максимальную дату
    files = sorted(PROPENSITY_DIR.glob("hot_customers_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Не найдены файлы hot_customers_*.csv в {PROPENSITY_DIR}. "
            f"Сначала запустите ML пайплайн."
        )
    latest = files[-1].stem.replace("hot_customers_", "")
    logger.info(f"[Loader] snapshot_date из имени файла: {latest}")
    return latest


def _safe_bool(val) -> Optional[bool]:
    """Конвертация строк 'True'/'False' в bool, с учётом NaN."""
    if pd.isna(val):
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def _safe_int(val) -> Optional[int]:
    """Конвертация в int с учётом NaN/None."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> Optional[float]:
    """Конвертация в float с учётом NaN/None."""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _chunk_upsert(db: Session, stmt_builder, rows: list[dict], chunk_size: int = 500) -> int:
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        stmt = stmt_builder(chunk)
        db.execute(stmt)
        db.commit()
        total += len(chunk)
    return total



# 1. Загрузка customers
def load_customers(db: Session) -> int:
    path = RAW_DATA_DIR / "customer_information.csv"
    logger.info(f"[Loader] Загрузка customers из {path}")

    df = pd.read_csv(path, parse_dates=["lastQuestionnaireDate", "timestamp"])

    # Оставляем только самую свежую запись на клиента
    df = df.sort_values("timestamp", ascending=False).drop_duplicates("customerID")

    rows = []
    for _, row in df.iterrows():
        risk = str(row.get("riskLevel", "Not_Available") or "Not_Available")
        capacity = str(row.get("investmentCapacity", "Not_Available") or "Not_Available")

        rows.append({
            "customer_id": str(row["customerID"]),
            "customer_type": str(row.get("customerType", "")) or None,
            "risk_level": risk,
            "investment_capacity": capacity,
            "is_profile_predicted": risk.startswith("Predicted_"),
            "is_capacity_missing": capacity == "Not_Available",
            "profile_date": row.get("lastQuestionnaireDate") if not pd.isna(row.get("lastQuestionnaireDate")) else None,
        })

    def _stmt(chunk):
        return pg_insert(Customer).values(chunk).on_conflict_do_update(
            index_elements=["customer_id"],
            set_={
                "customer_type": pg_insert(Customer).excluded.customer_type,
                "risk_level": pg_insert(Customer).excluded.risk_level,
                "investment_capacity": pg_insert(Customer).excluded.investment_capacity,
                "is_profile_predicted": pg_insert(Customer).excluded.is_profile_predicted,
                "is_capacity_missing": pg_insert(Customer).excluded.is_capacity_missing,
                "profile_date": pg_insert(Customer).excluded.profile_date,
            },
        )

    n = _chunk_upsert(db, _stmt, rows)
    logger.info(f"[Loader] customers: {n:,} записей загружено")
    return n



# 2. Загрузка assets
def load_assets(db: Session) -> int:
    path = RAW_DATA_DIR / "asset_information.csv"
    logger.info(f"[Loader] Загрузка assets из {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp", ascending=False).drop_duplicates("ISIN")

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "isin": str(row["ISIN"]),
            "asset_category": str(row.get("assetCategory", "")) or None,
            "asset_sub_category": str(row.get("assetSubCategory", "")) or None if not pd.isna(row.get("assetSubCategory", float("nan"))) else None,
            "sector": str(row.get("sector", "")) or None if not pd.isna(row.get("sector", float("nan"))) else None,
            "market": str(row.get("marketID", "")) or None,
            "asset_name": str(row.get("assetName", "")) or None,
        })

    def _stmt(chunk):
        return pg_insert(Asset).values(chunk).on_conflict_do_update(
            index_elements=["isin"],
            set_={
                "asset_category": pg_insert(Asset).excluded.asset_category,
                "asset_sub_category": pg_insert(Asset).excluded.asset_sub_category,
                "sector": pg_insert(Asset).excluded.sector,
                "market": pg_insert(Asset).excluded.market,
                "asset_name": pg_insert(Asset).excluded.asset_name,
            },
        )

    n = _chunk_upsert(db, _stmt, rows)
    logger.info(f"[Loader] assets: {n:,} записей загружено")
    return n



# 3. Загрузка transactions
def load_transactions(db: Session) -> int:
    """Загружает transactions.csv → таблица transactions.

    Фильтры (из ML pipeline логики):
    - transactionID >= 0  (исключаем синтетические)
    - transactionType in ['Buy', 'Sell']
    """
    path = RAW_DATA_DIR / "transactions.csv"
    logger.info(f"[Loader] Загрузка transactions из {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Фильтрация
    df = df[df["transactionID"] >= 0]
    df = df[df["transactionType"].isin(["Buy", "Sell"])]

    # Дедупликация по transactionID — в исходном CSV встречаются дубли.
    before = len(df)
    df = df.drop_duplicates(subset=["transactionID", "customerID", "transactionType"], keep="first")
    if len(df) < before:
        logger.info(f"[Loader] transactions: удалено {before - len(df):,} дублей transactionID")

    logger.info(f"[Loader] transactions после фильтрации: {len(df):,}")

    # Получаем множество customer_id уже загруженных в customers —
    # FK constraint не даст вставить транзакцию с неизвестным клиентом.
    known_customers = {r[0] for r in db.execute(
        __import__('sqlalchemy').text("SELECT customer_id FROM customers")
    ).fetchall()}
    unknown = df[~df["customerID"].isin(known_customers)]
    if len(unknown) > 0:
        logger.warning(
            f"[Loader] transactions: пропущено {len(unknown):,} строк "
            f"с неизвестным customerID (нет в таблице customers)"
        )
        df = df[df["customerID"].isin(known_customers)]

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "transaction_id": int(row["transactionID"]),
            "customer_id": str(row["customerID"]),
            "isin": str(row["ISIN"]) if not pd.isna(row.get("ISIN")) else None,
            "transaction_type": str(row["transactionType"]),
            "total_value": _safe_float(row.get("totalValue")),
            "channel": str(row.get("channel", "")) or None,
            "timestamp": row["timestamp"],
            "is_synthetic": False,
        })

    def _stmt(chunk):
        return pg_insert(Transaction).values(chunk).on_conflict_do_update(
            constraint="pk_transactions",
            set_={
                "total_value": pg_insert(Transaction).excluded.total_value,
                "channel": pg_insert(Transaction).excluded.channel,
            },
        )

    n = _chunk_upsert(db, _stmt, rows, chunk_size=500)
    logger.info(f"[Loader] transactions: {n:,} записей загружено")
    return n



# 4. Загрузка close_prices (опционально — большая таблица)
def load_close_prices(db: Session) -> int:
    path = RAW_DATA_DIR / "close_prices.csv"
    if not path.exists():
        logger.info(f"[Loader] close_prices.csv не найден — пропускаем")
        return 0

    logger.info(f"[Loader] Загрузка close_prices из {path} (может занять время...)")

    chunk_size = 5000
    total = 0

    for df_chunk in pd.read_csv(path, parse_dates=["timestamp"], chunksize=10000):
        rows = []
        for _, row in df_chunk.iterrows():
            rows.append({
                "isin": str(row["ISIN"]),
                "timestamp": row["timestamp"],
                "close_price": _safe_float(row.get("closePrice")),
            })

        def _stmt(chunk):
            return pg_insert(ClosePrice).values(chunk).on_conflict_do_update(
                index_elements=["isin", "timestamp"],
                set_={"close_price": pg_insert(ClosePrice).excluded.close_price},
            )

        n = _chunk_upsert(db, _stmt, rows, chunk_size=chunk_size)
        total += n

    logger.info(f"[Loader] close_prices: {total:,} записей загружено")
    return total



# 5. Загрузка scoring_results
def load_scoring_results(db: Session, date_str: str) -> int:
    snapshot_date = datetime.strptime(date_str, "%Y%m%d").date()

    all_scores_path = PROPENSITY_DIR / f"all_scores_{date_str}.csv"
    if not all_scores_path.exists():
        logger.warning(f"[Loader] {all_scores_path} не найден — пропускаем all_scores")
        total_all = 0
    else:
        logger.info(f"[Loader] Загрузка all_scores из {all_scores_path}")
        df_all = pd.read_csv(all_scores_path)
        logger.info(f"[Loader] all_scores: {len(df_all):,} строк, колонки: {list(df_all.columns)}")

        rows_all = _build_scoring_rows(df_all, snapshot_date, is_hot=False)

        def _stmt_all(chunk):
            return pg_insert(ScoringResult).values(chunk).on_conflict_do_update(
                constraint="idx_scoring_customer_date",
                set_={
                    "propensity_score": pg_insert(ScoringResult).excluded.propensity_score,
                    "segment": pg_insert(ScoringResult).excluded.segment,
                    "days_since_last_buy": pg_insert(ScoringResult).excluded.days_since_last_buy,
                    "rank": pg_insert(ScoringResult).excluded.rank,
                    "is_hot": pg_insert(ScoringResult).excluded.is_hot,
                },
            )

        total_all = _chunk_upsert(db, _stmt_all, rows_all)
        logger.info(f"[Loader] scoring_results (all): {total_all:,} записей")

    hot_path = PROPENSITY_DIR / f"hot_customers_{date_str}.csv"
    if not hot_path.exists():
        raise FileNotFoundError(f"Файл не найден: {hot_path}")

    logger.info(f"[Loader] Загрузка hot_customers из {hot_path}")
    df_hot = pd.read_csv(hot_path)
    logger.info(f"[Loader] hot_customers: {len(df_hot):,} строк")

    rows_hot = _build_scoring_rows(df_hot, snapshot_date, is_hot=True)

    def _stmt_hot(chunk):
        return pg_insert(ScoringResult).values(chunk).on_conflict_do_update(
            constraint="idx_scoring_customer_date",
            set_={
                "propensity_score": pg_insert(ScoringResult).excluded.propensity_score,
                "segment": pg_insert(ScoringResult).excluded.segment,
                "days_since_last_buy": pg_insert(ScoringResult).excluded.days_since_last_buy,
                "rank": pg_insert(ScoringResult).excluded.rank,
                "is_hot": pg_insert(ScoringResult).excluded.is_hot,
            },
        )

    total_hot = _chunk_upsert(db, _stmt_hot, rows_hot)
    logger.info(f"[Loader] scoring_results (hot): {total_hot:,} записей")

    return total_all + total_hot


def _build_scoring_rows(df: pd.DataFrame, snapshot_date: date, is_hot: bool) -> list[dict]:
    col_map = {
        "customerID": "customer_id",
        "propensity_score": "propensity_score",
        "segment": "segment",
        "days_since_last_buy": "days_since_last_buy",
        "rank": "rank",
    }

    rows = []
    for _, row in df.iterrows():
        customer_id = str(row.get("customerID", row.get("customer_id", "")))
        if not customer_id:
            continue

        rows.append({
            "customer_id": customer_id,
            "snapshot_date": snapshot_date,
            "propensity_score": _safe_float(row.get("propensity_score")),
            "segment": str(row.get("segment", "")) or None,
            "days_since_last_buy": _safe_int(row.get("days_since_last_buy")),
            "rank": _safe_int(row.get("rank")),
            "is_hot": is_hot,
        })
    return rows



# 6. Загрузка recommendations
def load_recommendations(db: Session, date_str: str) -> int:
    """Загружает recommendations_{DATE}.csv → таблица recommendations."""
    snapshot_date = datetime.strptime(date_str, "%Y%m%d").date()
    path = RECOMMENDER_DIR / f"recommendations_{date_str}.csv"

    if not path.exists():
        logger.warning(f"[Loader] {path} не найден — пропускаем recommendations")
        return 0

    logger.info(f"[Loader] Загрузка recommendations из {path}")
    df = pd.read_csv(path)
    logger.info(f"[Loader] recommendations: {len(df):,} строк")

    rows = []
    for _, row in df.iterrows():
        customer_id = str(row.get("customerID", row.get("customer_id", "")))
        if not customer_id:
            continue

        rows.append({
            "customer_id": customer_id,
            "snapshot_date": snapshot_date,
            "rec_type": str(row.get("rec_type", "")) or None,
            "risk_profile_verified": _safe_bool(row.get("risk_profile_verified")),
            "n_history_isins": _safe_int(row.get("n_history_isins")),
            # Рекомендация 1
            "rank_1_isin": str(row.get("rank_1_isin", "")) or None,
            "rank_1_category": str(row.get("rank_1_category", "")) or None,
            "rank_1_score": _safe_float(row.get("rank_1_score")),
            "rank_1_justification": str(row.get("rank_1_justification", "")) or None,
            "rank_1_outside_hist": _safe_bool(row.get("rank_1_outside_hist")),
            # Рекомендация 2
            "rank_2_isin": str(row.get("rank_2_isin", "")) or None,
            "rank_2_category": str(row.get("rank_2_category", "")) or None,
            "rank_2_score": _safe_float(row.get("rank_2_score")),
            "rank_2_justification": str(row.get("rank_2_justification", "")) or None,
            "rank_2_outside_hist": _safe_bool(row.get("rank_2_outside_hist")),
            # Рекомендация 3
            "rank_3_isin": str(row.get("rank_3_isin", "")) or None,
            "rank_3_category": str(row.get("rank_3_category", "")) or None,
            "rank_3_score": _safe_float(row.get("rank_3_score")),
            "rank_3_justification": str(row.get("rank_3_justification", "")) or None,
            "rank_3_outside_hist": _safe_bool(row.get("rank_3_outside_hist")),
        })

    def _stmt(chunk):
        return pg_insert(Recommendation).values(chunk).on_conflict_do_update(
            constraint="idx_rec_customer_date",
            set_={
                "rec_type": pg_insert(Recommendation).excluded.rec_type,
                "risk_profile_verified": pg_insert(Recommendation).excluded.risk_profile_verified,
                "n_history_isins": pg_insert(Recommendation).excluded.n_history_isins,
                "rank_1_isin": pg_insert(Recommendation).excluded.rank_1_isin,
                "rank_1_category": pg_insert(Recommendation).excluded.rank_1_category,
                "rank_1_score": pg_insert(Recommendation).excluded.rank_1_score,
                "rank_1_justification": pg_insert(Recommendation).excluded.rank_1_justification,
                "rank_1_outside_hist": pg_insert(Recommendation).excluded.rank_1_outside_hist,
                "rank_2_isin": pg_insert(Recommendation).excluded.rank_2_isin,
                "rank_2_category": pg_insert(Recommendation).excluded.rank_2_category,
                "rank_2_score": pg_insert(Recommendation).excluded.rank_2_score,
                "rank_2_justification": pg_insert(Recommendation).excluded.rank_2_justification,
                "rank_2_outside_hist": pg_insert(Recommendation).excluded.rank_2_outside_hist,
                "rank_3_isin": pg_insert(Recommendation).excluded.rank_3_isin,
                "rank_3_category": pg_insert(Recommendation).excluded.rank_3_category,
                "rank_3_score": pg_insert(Recommendation).excluded.rank_3_score,
                "rank_3_justification": pg_insert(Recommendation).excluded.rank_3_justification,
                "rank_3_outside_hist": pg_insert(Recommendation).excluded.rank_3_outside_hist,
            },
        )

    n = _chunk_upsert(db, _stmt, rows)
    logger.info(f"[Loader] recommendations: {n:,} записей загружено")
    return n



# 7. Синтетические советники
_LAST_NAMES_M = [
    "Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев",
    "Петров", "Соколов", "Михайлов", "Новиков", "Фёдоров",
    "Морозов", "Волков", "Алексеев", "Лебедев", "Семёнов",
    "Егоров", "Павлов", "Козлов", "Степанов", "Николаев",
    "Орлов", "Захаров", "Чернов", "Медведев", "Карпов",
    "Голубев", "Виноградов", "Богданов", "Воробьёв", "Романов",
]

_LAST_NAMES_F = [
    "Иванова", "Смирнова", "Кузнецова", "Попова", "Васильева",
    "Петрова", "Соколова", "Михайлова", "Новикова", "Фёдорова",
    "Морозова", "Волкова", "Алексеева", "Лебедева", "Семёнова",
    "Егорова", "Павлова", "Козлова", "Степанова", "Николаева",
    "Орлова", "Захарова", "Чернова", "Медведева", "Карпова",
    "Голубева", "Виноградова", "Богданова", "Воробьёва", "Романова",
]

_FIRST_NAMES_M = [
    "Александр", "Михаил", "Дмитрий", "Андрей", "Сергей",
    "Алексей", "Иван", "Артём", "Николай", "Владимир",
    "Павел", "Роман", "Денис", "Кирилл", "Максим",
    "Виктор", "Пётр", "Илья", "Евгений", "Константин",
]

_FIRST_NAMES_F = [
    "Анна", "Елена", "Ольга", "Наталья", "Екатерина",
    "Мария", "Татьяна", "Юлия", "Ирина", "Светлана",
]

N_ADVISORS = 60
_RNG_SEED  = 42


def _generate_unique_advisors(n: int) -> list[dict]:
    rng    = random.Random(_RNG_SEED + 999)
    result = []
    used_names = set()

    pool = []
    for first in _FIRST_NAMES_M:
        for last in _LAST_NAMES_M:
            pool.append((first, last, True))
    for first in _FIRST_NAMES_F:
        for last in _LAST_NAMES_F:
            pool.append((first, last, False))

    rng.shuffle(pool)

    for first, last, is_male in pool:
        name = f"{first} {last}"
        if name in used_names:
            continue
        used_names.add(name)
        slug = f"{last.lower()[:4]}{len(result):02d}"
        result.append({
            "advisor_name": name,
            "email":        f"{slug}@bank.demo",
        })
        if len(result) == n:
            break

    return result

def load_advisors(db: Session, date_str: str) -> None:
    logger.info(f"[Loader] Создание {N_ADVISORS} синтетических советников...")

    existing = {a.advisor_name for a in db.query(Advisor).all()}
    all_generated = _generate_unique_advisors(N_ADVISORS)
    new_advisors = [d for d in all_generated if d["advisor_name"] not in existing]

    if new_advisors:
        db.execute(pg_insert(Advisor).values(new_advisors).on_conflict_do_nothing())
        db.commit()

    advisors = db.query(Advisor).order_by(Advisor.advisor_id).all()
    logger.info(f"[Loader] Советников в БД: {len(advisors)}")

    all_customer_ids = [r[0] for r in db.query(Customer.customer_id).all()]
    logger.info(f"[Loader] Клиентов для распределения: {len(all_customer_ids):,}")

    rng = random.Random(_RNG_SEED)
    shuffled = all_customer_ids[:]
    rng.shuffle(shuffled)

    db.query(AdvisorClient).delete()
    db.commit()

    links = [
        {"advisor_id": advisors[i % len(advisors)].advisor_id, "customer_id": cid}
        for i, cid in enumerate(shuffled)
    ]

    for chunk_start in range(0, len(links), 500):
        chunk = links[chunk_start:chunk_start + 500]
        db.execute(pg_insert(AdvisorClient).values(chunk).on_conflict_do_nothing())
        db.commit()

    snapshot_date_obj = __import__('datetime').datetime.strptime(date_str, "%Y%m%d").date()
    hot_ids = {r[0] for r in db.query(ScoringResult.customer_id).filter(
        ScoringResult.snapshot_date == snapshot_date_obj,
        ScoringResult.is_hot == True,
    ).all()}

    advisor_hot_counts = {}
    for link in links:
        if link["customer_id"] in hot_ids:
            aid = link["advisor_id"]
            advisor_hot_counts[aid] = advisor_hot_counts.get(aid, 0) + 1

    counts = list(advisor_hot_counts.values())
    zeros  = len(advisors) - len(counts)
    logger.info(
        f"[Loader] Hot-клиентов на советника: "
        f"min={min(counts) if counts else 0}, "
        f"max={max(counts) if counts else 0}, "
        f"avg={sum(counts)/len(counts) if counts else 0:.1f}, "
        f"советников без hot-клиентов={zeros}"
    )



# Точка входа: load_all()
def load_all(skip_close_prices: bool = True) -> None:
    logger.info("=" * 60)
    logger.info("  ЗАГРУЗКА ДАННЫХ В БД")
    logger.info("=" * 60)

    # Создаём таблицы если не существуют
    create_tables()
    logger.info("[Loader] Таблицы созданы/проверены")

    # Определяем дату snapshot
    date_str = _get_snapshot_date()
    logger.info(f"[Loader] snapshot_date: {date_str}")

    db = SessionLocal()
    try:
        load_customers(db)
        load_assets(db)
        load_transactions(db)
        if not skip_close_prices:
            load_close_prices(db)

        load_scoring_results(db, date_str)
        load_recommendations(db, date_str)

        load_advisors(db, date_str)

    except Exception as e:
        logger.error(f"[Loader] Ошибка при загрузке: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()

    logger.info("=" * 60)
    logger.info("  ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО")
    logger.info("=" * 60)



# CLI запуск
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка данных в PostgreSQL")
    parser.add_argument(
        "--with-prices",
        action="store_true",
        help="Загрузить close_prices (большая таблица, по умолчанию пропускается)",
    )
    args = parser.parse_args()

    load_all(skip_close_prices=not args.with_prices)
