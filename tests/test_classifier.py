"""
tests/test_classifier.py
------------------------
Unit tests for the complaint classifier.
"""

import pytest

from app.models.schemas import AIClassification
from app.nlp.classifier import (
    _extract_json,
    _validate_result,
)


# ---------------------------------------------------------------------------
# Valid fixture
# ---------------------------------------------------------------------------

VALID_RESULT = {
    "category": "billing",
    "subcategory": "incorrect charge",
    "sentiment": "negative",
    "priority": "high",
    "confidence": 0.92,
    "summary": "Customer was charged incorrectly.",
    "suggested_action": "Review the customer's billing transaction.",
}


# ---------------------------------------------------------------------------
# JSON extraction tests
# ---------------------------------------------------------------------------


def test_extract_valid_json():
    """Valid JSON should be parsed successfully."""

    result = _extract_json(
        '{"category": "billing", "confidence": 0.9}'
    )

    assert result["category"] == "billing"
    assert result["confidence"] == 0.9


def test_extract_markdown_json():
    """JSON wrapped in Markdown fences should still be parsed."""

    result = _extract_json(
        """```json
        {"category": "billing", "confidence": 0.9}
        ```"""
    )

    assert result["category"] == "billing"


def test_extract_json_from_extra_text():
    """JSON surrounded by accidental text should be extracted."""

    result = _extract_json(
        'Here is the result: {"category": "billing", "confidence": 0.9}'
    )

    assert result["category"] == "billing"


def test_extract_empty_response():
    """Empty Gemini responses should raise ValueError."""

    with pytest.raises(ValueError, match="empty response"):
        _extract_json("")


def test_extract_whitespace_response():
    """Whitespace-only Gemini responses should raise ValueError."""

    with pytest.raises(ValueError, match="empty response"):
        _extract_json("   ")


def test_extract_invalid_json():
    """Malformed JSON should raise ValueError."""

    with pytest.raises(ValueError):
        _extract_json("this is definitely not json")


def test_extract_json_array_is_rejected():
    """A JSON array is not a valid classification response."""

    with pytest.raises(ValueError, match="JSON object"):
        _extract_json('["billing", "high"]')


# ---------------------------------------------------------------------------
# Classification validation tests
# ---------------------------------------------------------------------------


def test_valid_classification():
    """A valid classification should pass validation."""

    result = _validate_result(VALID_RESULT)

    assert isinstance(result, AIClassification)
    assert result.category == "billing"
    assert result.subcategory == "incorrect charge"
    assert result.sentiment == "negative"
    assert result.priority == "high"
    assert result.confidence == 0.92


def test_invalid_category():
    """Unknown categories must be rejected."""

    result = {
        **VALID_RESULT,
        "category": "invalid_category",
    }

    with pytest.raises(ValueError, match="Invalid category"):
        _validate_result(result)


def test_invalid_sentiment():
    """Unknown sentiments must be rejected."""

    result = {
        **VALID_RESULT,
        "sentiment": "angry",
    }

    with pytest.raises(ValueError, match="Invalid sentiment"):
        _validate_result(result)


def test_invalid_priority():
    """Unknown priorities must be rejected."""

    result = {
        **VALID_RESULT,
        "priority": "urgent",
    }

    with pytest.raises(ValueError, match="Invalid priority"):
        _validate_result(result)


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------


def test_confidence_below_zero():
    """Confidence below 0 must be rejected."""

    result = {
        **VALID_RESULT,
        "confidence": -0.1,
    }

    with pytest.raises(ValueError):
        _validate_result(result)


def test_confidence_above_one():
    """Confidence above 1 must be rejected."""

    result = {
        **VALID_RESULT,
        "confidence": 1.1,
    }

    with pytest.raises(ValueError):
        _validate_result(result)


def test_confidence_zero_is_valid():
    """A confidence of exactly 0 is valid."""

    result = {
        **VALID_RESULT,
        "confidence": 0.0,
    }

    validated = _validate_result(result)

    assert validated.confidence == 0.0


def test_confidence_one_is_valid():
    """A confidence of exactly 1 is valid."""

    result = {
        **VALID_RESULT,
        "confidence": 1.0,
    }

    validated = _validate_result(result)

    assert validated.confidence == 1.0


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "category",
        "sentiment",
        "priority",
        "confidence",
        "summary",
        "suggested_action",
    ],
)
def test_missing_required_field(field):
    """Required classification fields must be present."""

    result = {
        **VALID_RESULT,
    }

    del result[field]

    with pytest.raises(ValueError):
        _validate_result(result)


# ---------------------------------------------------------------------------
# Data-type validation
# ---------------------------------------------------------------------------


def test_confidence_string_is_rejected():
    """Confidence must be numeric."""

    result = {
        **VALID_RESULT,
        "confidence": "very confident",
    }

    with pytest.raises(ValueError):
        _validate_result(result)


def test_category_number_is_rejected():
    """Category must be a string."""

    result = {
        **VALID_RESULT,
        "category": 123,
    }

    with pytest.raises(ValueError):
        _validate_result(result)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_classification_serialization():
    """Validated classifications should serialize correctly."""

    result = _validate_result(VALID_RESULT)

    serialized = result.model_dump()

    assert serialized["category"] == "billing"
    assert serialized["subcategory"] == "incorrect charge"
    assert serialized["sentiment"] == "negative"
    assert serialized["priority"] == "high"
    assert serialized["confidence"] == 0.92


# ---------------------------------------------------------------------------
# Empty text handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_complaint_is_rejected():
    """Empty complaints should fail before calling Gemini."""

    from app.nlp.classifier import classify_complaint

    with pytest.raises(ValueError, match="cannot be empty"):
        await classify_complaint("")


@pytest.mark.asyncio
async def test_whitespace_complaint_is_rejected():
    """Whitespace-only complaints should fail before calling Gemini."""

    from app.nlp.classifier import classify_complaint

    with pytest.raises(ValueError, match="cannot be empty"):
        await classify_complaint("   ")
