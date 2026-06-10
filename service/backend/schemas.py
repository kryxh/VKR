from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, field_validator


JUSTIFICATION_MAP = {
    "risk_profile":         "Соответствует риск-профилю клиента",
    "similar_clients":      "Покупали клиенты с похожим портфелем",
    "portfolio_complement": "Дополнит текущую структуру портфеля клиента",
    "popularity":           "Популярно среди клиентов банка",
    "item_cf":              "Похоже на активы из портфеля клиента",
    "user_cf":              "Покупали клиенты с похожим профилем",
    "portfolio":            "Дополнит текущую структуру портфеля клиента",
    "popular_sector":       "Популярно среди клиентов аналогичного сектора",
    "popular_category":     "Популярно среди клиентов с похожими предпочтениями",
}


# GET /api/clients — список клиентов
class ClientListItem(BaseModel):
    customer_id: str
    rank: int
    segment: str
    propensity_score: float
    days_since_last_buy: Optional[int]
    last_asset_category: Optional[str]

    model_config = {"from_attributes": True}


class ClientListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ClientListItem]



# GET /api/clients/{customer_id}
class ClientProfile(BaseModel):
    customer_id: str
    customer_type: Optional[str]
    risk_level: Optional[str]
    investment_capacity: Optional[str]
    days_in_system: Optional[int]
    days_since_last_buy: Optional[int]
    propensity_score: float
    segment: str

    model_config = {"from_attributes": True}


class Portfolio(BaseModel):
    n_unique_assets: int
    n_buy_transactions: int
    first_buy_date: Optional[date]
    last_buy_date: Optional[date]
    top_category: Optional[str]
    top_sector: Optional[str]

    model_config = {"from_attributes": True}


class RecommendationItem(BaseModel):
    rank: int
    isin: Optional[str]
    asset_name: Optional[str]
    category: Optional[str]
    asset_sub_category: Optional[str]
    sector: Optional[str]
    score: Optional[float]
    justification: Optional[str]
    justification_text: Optional[str]
    outside_hist: Optional[bool]

    model_config = {"from_attributes": True}


class TransactionItem(BaseModel):
    date: Optional[date]
    isin: Optional[str]
    asset_name: Optional[str]
    asset_category: Optional[str]
    transaction_type: str
    total_value: Optional[float]

    model_config = {"from_attributes": True}


class ClientCardResponse(BaseModel):
    profile: ClientProfile
    portfolio: Portfolio
    recommendations: list[RecommendationItem]
    recent_transactions: list[TransactionItem]

    model_config = {"from_attributes": True}



# GET /api/snapshot-date
class SnapshotDateResponse(BaseModel):
    snapshot_date: Optional[date]



# POST /api/pipeline/run
class PipelineRunRequest(BaseModel):
    snapshot_date: Optional[str] = None


class PipelineRunResponse(BaseModel):
    status: str
    message: str
