import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from db.models import Base


# Engine
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://advisory_user:secure_password@localhost:5432/advisory_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def check_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[DB] Ошибка подключения: {e}")
        return False


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
