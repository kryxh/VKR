from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass



# Исходные данные FAR-Trans
class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True)
    customer_type = Column(String)
    risk_level = Column(String)
    investment_capacity = Column(String)
    is_profile_predicted = Column(Boolean, default=False)
    is_capacity_missing = Column(Boolean, default=False)
    profile_date = Column(DateTime)

    
    transactions = relationship("Transaction", back_populates="customer", lazy="dynamic")
    scoring_results = relationship("ScoringResult", back_populates="customer")
    recommendations = relationship("Recommendation", back_populates="customer")
    advisor_links = relationship("AdvisorClient", back_populates="customer")


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(BigInteger, nullable=False)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    isin = Column(String, ForeignKey("assets.isin"), nullable=True)
    transaction_type = Column(String, nullable=False)  # 'Buy' или 'Sell'
    total_value = Column(Numeric(18, 2))
    channel = Column(String)
    timestamp = Column(DateTime, nullable=False)
    is_synthetic = Column(Boolean, default=False)

    customer = relationship("Customer", back_populates="transactions")
    asset = relationship("Asset", back_populates="transactions")

    __table_args__ = (
        PrimaryKeyConstraint("transaction_id", "customer_id", "transaction_type", name="pk_transactions"),
        Index("idx_tx_customer_date", "customer_id", "timestamp"),
        Index("idx_tx_date", "timestamp"),
    )


class Asset(Base):
    __tablename__ = "assets"

    isin = Column(String, primary_key=True)
    asset_category = Column(String)
    asset_sub_category = Column(String)
    sector = Column(String)
    market = Column(String)
    asset_name = Column(String)

    transactions = relationship("Transaction", back_populates="asset")


class ClosePrice(Base):
    __tablename__ = "close_prices"

    isin = Column(String, ForeignKey("assets.isin"), primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    close_price = Column(Numeric(18, 4))



# Результаты ML-пайплайна
class ScoringResult(Base):
    __tablename__ = "scoring_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    snapshot_date = Column(Date, nullable=False)
    propensity_score = Column(Numeric(8, 6))
    segment = Column(String)
    days_since_last_buy = Column(Integer)
    rank = Column(Integer)
    is_hot = Column(Boolean, default=False)

    created_at = Column(DateTime, default=func.now())

    customer = relationship("Customer", back_populates="scoring_results")

    __table_args__ = (
        Index("idx_scoring_date", "snapshot_date"),
        UniqueConstraint("customer_id", "snapshot_date", name="idx_scoring_customer_date"),
    )


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    snapshot_date = Column(Date, nullable=False)
    rec_type = Column(String)                    # 'als' или 'fallback'
    risk_profile_verified = Column(Boolean)
    n_history_isins = Column(Integer)

    # Рекомендация 1
    rank_1_isin = Column(String)
    rank_1_category = Column(String)
    rank_1_score = Column(Numeric(12, 6))
    rank_1_justification = Column(String)
    rank_1_outside_hist = Column(Boolean)

    # Рекомендация 2
    rank_2_isin = Column(String)
    rank_2_category = Column(String)
    rank_2_score = Column(Numeric(12, 6))
    rank_2_justification = Column(String)
    rank_2_outside_hist = Column(Boolean)

    # Рекомендация 3
    rank_3_isin = Column(String)
    rank_3_category = Column(String)
    rank_3_score = Column(Numeric(12, 6))
    rank_3_justification = Column(String)
    rank_3_outside_hist = Column(Boolean)

    created_at = Column(DateTime, default=func.now())

    customer = relationship("Customer", back_populates="recommendations")

    __table_args__ = (
        UniqueConstraint("customer_id", "snapshot_date", name="idx_rec_customer_date"),
    )



# Советники
class Advisor(Base):
    __tablename__ = "advisors"

    advisor_id = Column(Integer, primary_key=True, autoincrement=True)
    advisor_name = Column(String, nullable=False)
    email = Column(String)

    
    client_links = relationship("AdvisorClient", back_populates="advisor")


class AdvisorClient(Base):
    __tablename__ = "advisor_clients"

    advisor_id = Column(Integer, ForeignKey("advisors.advisor_id"), primary_key=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), primary_key=True)

    advisor = relationship("Advisor", back_populates="client_links")
    customer = relationship("Customer", back_populates="advisor_links")
