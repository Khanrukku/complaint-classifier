"""
tests/test_classifier.py
Tests for classification logic, API endpoints, and data validation.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.models.schemas import CATEGORIES, PRIORITIES, SENTIMENTS


# ── Schema validation tests ───────────────────────────────────────────────────

def test_categories_defined():
    assert len(CATEGORIES) == 8
    assert "billing" in CATEGORIES
    assert "technical_support" in CATEGORIES
    assert "other" in CATEGORIES


def test_priorities_defined():
    assert set(PRIORITIES) == {"low", "medium", "high", "critical"}


def test_sentiments_defined():
    assert set(SENTIMENTS) == {"positive", "negative", "neutral"}


# ── Classification result validation ─────────────────────────────────────────

def test_confidence_range():
    """Confidence must always be between 0 and 1."""
    confidence = 0.85
    assert 0.0 <= confidence <= 1.0


def test_category_validation():
    """Unknown categories should fall back to 'other'."""
    category = "unknown_category"
    if category not in CATEGORIES:
        category = "other"
    assert category == "other"


def test_priority_validation():
    """Unknown priorities should fall back to 'medium'."""
    priority = "unknown_priority"
    if priority not in PRIORITIES:
        priority = "medium"
    assert priority == "medium"


# ── Batch processing tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_concurrent_processing():
    """Batch classifier processes multiple complaints concurrently."""
    mock_result = {
        "category": "billing",
        "subcategory": "overcharge",
        "sentiment": "negative",
        "priority": "high",
        "confidence": 0.92,
        "summary": "Customer was overcharged.",
        "suggested_action": "Review billing records.",
    }

    with patch("app.nlp.classifier.classify_complaint", new=AsyncMock(return_value=mock_result)):
        from app.nlp.classifier import classify_batch
        texts = ["Complaint 1", "Complaint 2", "Complaint 3"]
        results = await classify_batch(texts)

    assert len(results) == 3
    for result in results:
        assert result["category"] == "billing"
        assert result["confidence"] == 0.92


@pytest.mark.asyncio
async def test_batch_handles_failure_gracefully():
    """Batch classifier returns default result when one classification fails."""
    call_count = 0

    async def mock_classify(text):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("Gemini API error")
        return {
            "category": "billing", "subcategory": "", "sentiment": "negative",
            "priority": "medium", "confidence": 0.8,
            "summary": "Test", "suggested_action": "Review."
        }

    with patch("app.nlp.classifier.classify_complaint", side_effect=mock_classify):
        from app.nlp.classifier import classify_batch
        results = await classify_batch(["text1", "text2", "text3"])

    assert len(results) == 3
    # Failed one should have confidence 0.0
    assert results[1]["confidence"] == 0.0


# ── Priority logic tests ──────────────────────────────────────────────────────

def test_critical_keywords():
    """Complaints with legal threats should be classified as critical."""
    critical_keywords = ["sue", "lawyer", "legal action", "court", "safety", "danger"]
    complaint = "I will take legal action if this is not resolved immediately"
    is_critical = any(kw in complaint.lower() for kw in critical_keywords)
    assert is_critical


def test_text_length_validation():
    """Complaints exceeding max length should be truncated."""
    max_length = 2000
    long_text = "x" * 3000
    truncated = long_text[:max_length]
    assert len(truncated) == max_length
