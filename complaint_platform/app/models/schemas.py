"""
app/models/schemas.py
----------------------
Pydantic request/response models.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List


# ── Categories ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "billing",
    "technical_support",
    "delivery",
    "product_quality",
    "customer_service",
    "refund_return",
    "account_access",
    "other",
]

PRIORITIES = ["low", "medium", "high", "critical"]
SENTIMENTS = ["positive", "negative", "neutral"]


# ── Request models ────────────────────────────────────────────────────────────

class ComplaintSubmit(BaseModel):
    text:   str = Field(..., min_length=10, max_length=2000,
                        description="The complaint text to classify")
    source: str = Field("api", description="Source of complaint")


class BulkComplaintSubmit(BaseModel):
    complaints: List[ComplaintSubmit] = Field(..., min_length=1, max_length=50)


class ResolveRequest(BaseModel):
    complaint_id: str


# ── Response models ───────────────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    id:               str
    text:             str
    category:         str
    subcategory:      Optional[str]
    sentiment:        str
    priority:         str
    confidence:       float
    summary:          str
    suggested_action: str
    created_at:       datetime


class ComplaintListItem(BaseModel):
    id:          str
    category:    str
    priority:    str
    confidence:  float
    is_resolved: bool
    created_at:  datetime
    summary:     str


class DashboardStats(BaseModel):
    total_complaints:    int
    resolved:            int
    unresolved:          int
    by_category:         dict
    by_priority:         dict
    by_sentiment:        dict
    avg_confidence:      float
    resolution_rate:     float


class HealthResponse(BaseModel):
    status:   str
    model:    str
    db:       str
    version:  str = "1.0.0"
