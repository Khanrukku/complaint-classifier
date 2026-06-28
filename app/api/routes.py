"""
app/api/routes.py
-----------------
REST API endpoints:

  POST /complaints/classify     — classify a single complaint
  POST /complaints/bulk         — classify up to 50 complaints concurrently
  GET  /complaints              — list all complaints (filterable)
  GET  /complaints/{id}         — get a single complaint
  POST /complaints/{id}/resolve — mark complaint as resolved
  GET  /dashboard               — aggregated stats for reporting
  GET  /health                  — health check
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Complaint, AuditLog, get_db
from app.models.schemas import (
    ComplaintSubmit, BulkComplaintSubmit, ResolveRequest,
    ClassificationResult, ComplaintListItem, DashboardStats, HealthResponse,
)
from app.nlp.classifier import classify_complaint, classify_batch
from app.core.config import settings

router = APIRouter()


# ── Classify single ───────────────────────────────────────────────────────────

@router.post("/complaints/classify", response_model=ClassificationResult, status_code=201)
async def classify_single(req: ComplaintSubmit, db: AsyncSession = Depends(get_db)):
    """
    Submit and classify a single complaint.
    Classification happens in under 1 second via Gemini.
    """
    result = await classify_complaint(req.text)

    complaint = Complaint(
        id               = str(uuid.uuid4()),
        text             = req.text,
        category         = result["category"],
        subcategory      = result.get("subcategory", ""),
        sentiment        = result["sentiment"],
        priority         = result["priority"],
        confidence       = result["confidence"],
        summary          = result["summary"],
        suggested_action = result["suggested_action"],
        source           = req.source,
    )
    db.add(complaint)

    log = AuditLog(
        complaint_id = complaint.id,
        action       = "classified",
        details      = f"category={result['category']} confidence={result['confidence']:.2f}",
    )
    db.add(log)
    await db.commit()
    await db.refresh(complaint)

    return ClassificationResult(
        id               = complaint.id,
        text             = complaint.text,
        category         = complaint.category,
        subcategory      = complaint.subcategory,
        sentiment        = complaint.sentiment,
        priority         = complaint.priority,
        confidence       = complaint.confidence,
        summary          = complaint.summary,
        suggested_action = complaint.suggested_action,
        created_at       = complaint.created_at,
    )


# ── Bulk classify ─────────────────────────────────────────────────────────────

@router.post("/complaints/bulk", status_code=201)
async def classify_bulk(req: BulkComplaintSubmit, db: AsyncSession = Depends(get_db)):
    """
    Classify up to 50 complaints concurrently using asyncio.gather.
    Demonstrates parallel AI inference for high-throughput scenarios.
    """
    texts   = [c.text for c in req.complaints]
    results = await classify_batch(texts)

    complaints = []
    for i, result in enumerate(results):
        complaint = Complaint(
            id               = str(uuid.uuid4()),
            text             = req.complaints[i].text,
            category         = result["category"],
            subcategory      = result.get("subcategory", ""),
            sentiment        = result["sentiment"],
            priority         = result["priority"],
            confidence       = result["confidence"],
            summary          = result["summary"],
            suggested_action = result["suggested_action"],
            source           = req.complaints[i].source,
        )
        complaints.append(complaint)
        db.add(complaint)

    await db.commit()

    return {
        "processed":  len(complaints),
        "complaints": [
            {
                "id":       c.id,
                "category": c.category,
                "priority": c.priority,
                "summary":  c.summary,
            }
            for c in complaints
        ]
    }


# ── List complaints ───────────────────────────────────────────────────────────

@router.get("/complaints", response_model=List[ComplaintListItem])
async def list_complaints(
    category:    Optional[str] = Query(None),
    priority:    Optional[str] = Query(None),
    is_resolved: Optional[bool] = Query(None),
    limit:       int = Query(50, ge=1, le=200),
    offset:      int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List complaints with optional filters."""
    stmt = select(Complaint).order_by(Complaint.created_at.desc())

    if category:
        stmt = stmt.where(Complaint.category == category)
    if priority:
        stmt = stmt.where(Complaint.priority == priority)
    if is_resolved is not None:
        stmt = stmt.where(Complaint.is_resolved == is_resolved)

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    complaints = result.scalars().all()

    return [
        ComplaintListItem(
            id          = c.id,
            category    = c.category,
            priority    = c.priority,
            confidence  = c.confidence,
            is_resolved = c.is_resolved,
            created_at  = c.created_at,
            summary     = c.summary or "",
        )
        for c in complaints
    ]


# ── Get single complaint ──────────────────────────────────────────────────────

@router.get("/complaints/{complaint_id}", response_model=ClassificationResult)
async def get_complaint(complaint_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Complaint).where(Complaint.id == complaint_id))
    complaint = result.scalar_one_or_none()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    return ClassificationResult(
        id               = complaint.id,
        text             = complaint.text,
        category         = complaint.category,
        subcategory      = complaint.subcategory,
        sentiment        = complaint.sentiment,
        priority         = complaint.priority,
        confidence       = complaint.confidence,
        summary          = complaint.summary or "",
        suggested_action = complaint.suggested_action or "",
        created_at       = complaint.created_at,
    )


# ── Resolve complaint ─────────────────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/resolve")
async def resolve_complaint(complaint_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Complaint).where(Complaint.id == complaint_id))
    complaint = result.scalar_one_or_none()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    complaint.is_resolved = True
    complaint.resolved_at = datetime.utcnow()

    log = AuditLog(complaint_id=complaint_id, action="resolved")
    db.add(log)
    await db.commit()

    return {"message": "Complaint resolved", "complaint_id": complaint_id}


# ── Dashboard stats ───────────────────────────────────────────────────────────

@router.get("/dashboard", response_model=DashboardStats)
async def dashboard(db: AsyncSession = Depends(get_db)):
    """
    Aggregated statistics for management dashboards.
    Demonstrates data analytics and reporting capability.
    """
    total    = await db.scalar(select(func.count(Complaint.id)))
    resolved = await db.scalar(select(func.count(Complaint.id)).where(Complaint.is_resolved == True))
    avg_conf = await db.scalar(select(func.avg(Complaint.confidence)))

    # By category
    cat_rows = await db.execute(
        select(Complaint.category, func.count(Complaint.id))
        .group_by(Complaint.category)
    )
    by_category = {row[0]: row[1] for row in cat_rows}

    # By priority
    pri_rows = await db.execute(
        select(Complaint.priority, func.count(Complaint.id))
        .group_by(Complaint.priority)
    )
    by_priority = {row[0]: row[1] for row in pri_rows}

    # By sentiment
    sent_rows = await db.execute(
        select(Complaint.sentiment, func.count(Complaint.id))
        .group_by(Complaint.sentiment)
    )
    by_sentiment = {row[0]: row[1] for row in sent_rows}

    return DashboardStats(
        total_complaints = total or 0,
        resolved         = resolved or 0,
        unresolved       = (total or 0) - (resolved or 0),
        by_category      = by_category,
        by_priority      = by_priority,
        by_sentiment     = by_sentiment,
        avg_confidence   = round(float(avg_conf or 0), 3),
        resolution_rate  = round((resolved or 0) / max(total or 1, 1), 3),
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.scalar(select(func.count(Complaint.id)))
        db_status = "ok"
    except Exception:
        db_status = "error"

    return HealthResponse(
        status  = "ok",
        model   = settings.gemini_model,
        db      = db_status,
    )
