from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import Advisor, AdvisorClient, Asset, Customer, Recommendation, ScoringResult, Transaction
from db.session import get_db
from schemas import (
    JUSTIFICATION_MAP,
    ClientCardResponse,
    ClientListItem,
    ClientListResponse,
    ClientProfile,
    Portfolio,
    RecommendationItem,
    SnapshotDateResponse,
    TransactionItem,
)

from sqlalchemy import text

router = APIRouter()


def _get_latest_snapshot(db: Session) -> Optional[date]:
    return db.query(func.max(ScoringResult.snapshot_date)).scalar()


def _get_last_asset_category(db: Session, customer_id: str, snapshot_date: date) -> Optional[str]:
    row = (
        db.query(Asset.asset_category)
        .join(Transaction, Transaction.isin == Asset.isin)
        .filter(
            Transaction.customer_id == customer_id,
            Transaction.transaction_type == "Buy",
            Transaction.timestamp <= snapshot_date,
        )
        .order_by(Transaction.timestamp.desc())
        .first()
    )
    return row[0] if row else None



@router.get("/advisors")
def get_advisors(db: Session = Depends(get_db)):
    """Список советников для фильтра на фронтенде."""
    advisors = db.query(Advisor).order_by(Advisor.advisor_id).all()
    return {
        "items": [
            {"advisor_id": a.advisor_id, "advisor_name": a.advisor_name}
            for a in advisors
        ]
    }



# GET /api/clients/stats
@router.get("/clients/stats")
def get_clients_stats(db: Session = Depends(get_db)):
    """Возвращает агрегаты для настройки фильтров на фронтенде."""
    snapshot_date = _get_latest_snapshot(db)
    if snapshot_date is None:
        return {"max_days": 365, "total_hot": 0}

    max_days = db.query(func.max(ScoringResult.days_since_last_buy)).filter(
        ScoringResult.snapshot_date == snapshot_date,
        ScoringResult.is_hot == True,
    ).scalar() or 365

    total_hot = db.query(func.count(ScoringResult.id)).filter(
        ScoringResult.snapshot_date == snapshot_date,
        ScoringResult.is_hot == True,
    ).scalar() or 0

    return {"max_days": int(max_days), "total_hot": int(total_hot)}



# GET /api/clients
@router.get("/clients", response_model=ClientListResponse)
def get_clients(
    segment:      Optional[str]   = Query(None),
    score_min:    float            = Query(0.0, ge=0.0, le=1.0),
    score_max:    float            = Query(1.0, ge=0.0, le=1.0),
    days_min:     int              = Query(0,   ge=0),
    days_max:     int              = Query(9999, ge=0),
    advisor_name: Optional[str]   = Query(None),
    search_id:    Optional[str]   = Query(None),
    page:         int              = Query(1,   ge=1),
    page_size:    int              = Query(50,  ge=1, le=200),
    db:           Session          = Depends(get_db),
):
    snapshot_date = _get_latest_snapshot(db)
    if snapshot_date is None:
        raise HTTPException(status_code=404, detail="Данные не загружены.")

    query = db.query(ScoringResult).filter(
        ScoringResult.snapshot_date == snapshot_date,
        ScoringResult.is_hot == True,
    )

    if segment and segment.lower() in ("warm", "dormant"):
        query = query.filter(ScoringResult.segment == segment.lower())

    query = query.filter(
        ScoringResult.propensity_score >= score_min,
        ScoringResult.propensity_score <= score_max,
    )

    if days_max < 9999:
        query = query.filter(ScoringResult.days_since_last_buy <= days_max)
    if days_min > 0:
        query = query.filter(ScoringResult.days_since_last_buy >= days_min)

    if advisor_name and advisor_name != "Все советники":
        advisor = db.query(Advisor).filter(Advisor.advisor_name == advisor_name).first()
        if advisor:
            advisor_customer_ids = (
                db.query(AdvisorClient.customer_id)
                .filter(AdvisorClient.advisor_id == advisor.advisor_id)
                .subquery()
            )
            query = query.filter(ScoringResult.customer_id.in_(advisor_customer_ids))

    if search_id and search_id.strip():
        query = query.filter(
            ScoringResult.customer_id.ilike(f"%{search_id.strip()}%")
        )

    total  = query.count()
    offset = (page - 1) * page_size
    results = query.order_by(ScoringResult.rank).offset(offset).limit(page_size).all()

    items = []
    for sr in results:
        last_cat = _get_last_asset_category(db, sr.customer_id, snapshot_date)
        items.append(ClientListItem(
            customer_id=sr.customer_id,
            rank=sr.rank or 0,
            segment=sr.segment or "unknown",
            propensity_score=float(sr.propensity_score or 0),
            days_since_last_buy=sr.days_since_last_buy,
            last_asset_category=last_cat,
        ))

    return ClientListResponse(total=total, page=page, page_size=page_size, items=items)



# GET /api/clients/{customer_id}
@router.get("/clients/{customer_id}", response_model=ClientCardResponse)
def get_client_card(customer_id: str, db: Session = Depends(get_db)):
    snapshot_date = _get_latest_snapshot(db)
    if snapshot_date is None:
        raise HTTPException(status_code=404, detail="Данные не загружены.")

    customer = db.query(Customer).filter_by(customer_id=customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Клиент {customer_id} не найден")

    scoring = (
        db.query(ScoringResult)
        .filter_by(customer_id=customer_id, snapshot_date=snapshot_date)
        .first()
    )
    if not scoring:
        raise HTTPException(status_code=404, detail=f"Скоринг для {customer_id} не найден")

    min_ts = db.query(func.min(Transaction.timestamp)).filter_by(customer_id=customer_id).scalar()
    days_in_system = (snapshot_date - min_ts.date()).days if min_ts else None

    profile = ClientProfile(
        customer_id=customer_id,
        customer_type=customer.customer_type,
        risk_level=customer.risk_level,
        investment_capacity=customer.investment_capacity,
        days_in_system=days_in_system,
        days_since_last_buy=scoring.days_since_last_buy,
        propensity_score=float(scoring.propensity_score or 0),
        segment=scoring.segment or "unknown",
    )

    portfolio     = _compute_portfolio(db, customer_id, snapshot_date)
    recommendations = _build_recommendations(db, customer_id, snapshot_date)
    recent_tx     = _get_recent_transactions(db, customer_id, snapshot_date)

    return ClientCardResponse(
        profile=profile,
        portfolio=portfolio,
        recommendations=recommendations,
        recent_transactions=recent_tx,
    )


def _compute_portfolio(db: Session, customer_id: str, snapshot_date: date) -> Portfolio:
    buy_txs = (
        db.query(Transaction)
        .filter(
            Transaction.customer_id == customer_id,
            Transaction.transaction_type == "Buy",
            Transaction.timestamp <= snapshot_date,
        )
        .all()
    )

    if not buy_txs:
        return Portfolio(
            n_unique_assets=0, n_buy_transactions=0,
            first_buy_date=None, last_buy_date=None,
            top_category=None, top_sector=None,
        )

    n_unique = len({tx.isin for tx in buy_txs if tx.isin})
    n_total  = len(buy_txs)
    dates    = [tx.timestamp.date() for tx in buy_txs if tx.timestamp]
    first_buy = min(dates) if dates else None
    last_buy  = max(dates) if dates else None

    from collections import Counter
    isins = [tx.isin for tx in buy_txs if tx.isin]
    assets = db.query(Asset).filter(Asset.isin.in_(set(isins))).all()
    asset_map = {a.isin: a for a in assets}

    cat_counts    = Counter()
    sector_counts = Counter()
    for tx in buy_txs:
        a = asset_map.get(tx.isin)
        if a:
            if a.asset_category:
                cat_counts[a.asset_category] += 1
            if a.sector:
                sector_counts[a.sector] += 1

    top_category = cat_counts.most_common(1)[0][0] if cat_counts else None
    top_sector   = sector_counts.most_common(1)[0][0] if sector_counts else None

    return Portfolio(
        n_unique_assets=n_unique,
        n_buy_transactions=n_total,
        first_buy_date=first_buy,
        last_buy_date=last_buy,
        top_category=top_category,
        top_sector=top_sector,
    )


def _build_recommendations(db: Session, customer_id: str, snapshot_date: date) -> list[RecommendationItem]:
    rec = db.query(Recommendation).filter_by(customer_id=customer_id, snapshot_date=snapshot_date).first()
    if not rec:
        return []

    result = []
    for rank in (1, 2, 3):
        isin          = getattr(rec, f"rank_{rank}_isin", None)
        category      = getattr(rec, f"rank_{rank}_category", None)
        score         = getattr(rec, f"rank_{rank}_score", None)
        justification = getattr(rec, f"rank_{rank}_justification", None)
        outside_hist  = getattr(rec, f"rank_{rank}_outside_hist", None)
        if not isin:
            continue
        asset = db.query(Asset).filter_by(isin=isin).first()
        result.append(RecommendationItem(
            rank=rank,
            isin=isin,
            asset_name=asset.asset_name if asset else isin,
            category=category,
            asset_sub_category=getattr(asset, "asset_sub_category", None) if asset else None,
            sector=asset.sector if asset else None,
            score=float(score) if score is not None else None,
            justification=justification,
            justification_text=JUSTIFICATION_MAP.get(justification or "", justification),
            outside_hist=bool(outside_hist) if outside_hist is not None else None,
        ))
    return result


def _get_recent_transactions(db: Session, customer_id: str, snapshot_date: date, limit: int = 10) -> list[TransactionItem]:
    rows = (
        db.query(Transaction, Asset.asset_name, Asset.asset_category)
        .outerjoin(Asset, Asset.isin == Transaction.isin)
        .filter(
            Transaction.customer_id == customer_id,
            Transaction.timestamp <= snapshot_date,
        )
        .order_by(Transaction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        TransactionItem(
            date=tx.timestamp.date() if tx.timestamp else None,
            isin=tx.isin,
            asset_name=asset_name,
            asset_category=asset_category,
            transaction_type=tx.transaction_type,
            total_value=float(tx.total_value) if tx.total_value else None,
        )
        for tx, asset_name, asset_category in rows
    ]



# GET /api/snapshot-date
@router.get("/snapshot-date", response_model=SnapshotDateResponse)
def get_snapshot_date(db: Session = Depends(get_db)):
    return SnapshotDateResponse(snapshot_date=_get_latest_snapshot(db))
