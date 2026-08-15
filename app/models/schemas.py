"""
app/models/schemas.py
----------------------
Pydantic request/response models and shared classification constants.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

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

PRIORITIES = [
    "low",
    "medium",
    "high",
    "critical",
]

SENTIMENTS = [
    "positive",
    "negative",
    "neutral",
]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ComplaintSubmit(BaseModel):
    """Request model for submitting one complaint."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The complaint text to classify",
    )

    source: str = Field(
        default="api",
        description="Source of the complaint",
    )


class BulkComplaintSubmit(BaseModel):
    """Request model for submitting multiple complaints."""

    complaints: list[ComplaintSubmit] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Complaints to classify",
    )


class ResolveRequest(BaseModel):
    """Request model for resolving a complaint."""

    complaint_id: str


# ---------------------------------------------------------------------------
# AI classification model
# ---------------------------------------------------------------------------


class AIClassification(BaseModel):
    """
    Structured classification returned by the AI model.

    Confidence is the model's self-reported confidence score.
    It should not be interpreted as a statistically calibrated probability.
    """

    category: str
    subcategory: str = ""
    sentiment: str
    priority: str
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    summary: str
    suggested_action: str


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Complete complaint classification returned by the API."""

    id: str
    text: str
    category: str
    subcategory: Optional[str] = None
    sentiment: str
    priority: str
    confidence: float
    summary: str
    suggested_action: str
    created_at: datetime


class ComplaintListItem(BaseModel):
    """Compact complaint representation used in list endpoints."""

    id: str
    category: str
    priority: str
    confidence: float
    is_resolved: bool
    created_at: datetime
    summary: str


class DashboardStats(BaseModel):
    """Aggregated complaint dashboard statistics."""

    total_complaints: int
    resolved: int
    unresolved: int
    by_category: dict
    by_priority: dict
    by_sentiment: dict
    avg_confidence: float
    resolution_rate: float


class HealthResponse(BaseModel):
    """Application health information."""

    status: str
    model: str
    db: str
    version: str = "1.0.0"
