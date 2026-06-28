"""
app/models/database.py
-----------------------
SQLAlchemy async database models.

Tables:
  complaints  — every submitted complaint with classification result
  audit_log   — tracks classification decisions for compliance
"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Boolean, Index
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

Base = declarative_base()


class Complaint(Base):
    __tablename__ = "complaints"

    id              = Column(String(36), primary_key=True)
    text            = Column(Text, nullable=False)
    category        = Column(String(50), nullable=False)
    subcategory     = Column(String(100), nullable=True)
    sentiment       = Column(String(20), nullable=True)   # positive/negative/neutral
    priority        = Column(String(10), nullable=True)   # low/medium/high/critical
    confidence      = Column(Float, nullable=False)
    summary         = Column(Text, nullable=True)
    suggested_action = Column(Text, nullable=True)
    is_resolved     = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    resolved_at     = Column(DateTime, nullable=True)
    source          = Column(String(50), default="api")   # api / bulk / web

    __table_args__ = (
        Index("idx_category",   "category"),
        Index("idx_priority",   "priority"),
        Index("idx_created_at", "created_at"),
        Index("idx_resolved",   "is_resolved"),
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id   = Column(String(36), nullable=False)
    action         = Column(String(50), nullable=False)   # classified / resolved / updated
    details        = Column(Text, nullable=True)
    timestamp      = Column(DateTime, default=datetime.utcnow)


# ── Engine & session ──────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
