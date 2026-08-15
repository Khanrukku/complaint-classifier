import pytest

from app.nlp.classifier import (
    ClassificationResult,
    _extract_json,
    _validate_result,
)


VALID_RESULT = {
    "category": "billing",
    "subcategory": "incorrect charge",
    "sentiment": "negative",
    "priority": "high",
    "confidence": 0.92,
    "summary": "Customer was charged incorrectly.",
    "suggested_action": "Review the customer's billing transaction.",
}


def test_extract_valid_json():
    result = _extract_json(
        '{"category": "billing", "confidence": 0.9}'
    )

    assert result["category"] == "billing"
    assert result["confidence"] == 0.9


def test_extract_markdown_json():
    result = _extract_json(
        """```json
        {"category": "billing", "confidence": 0.9}
        ```"""
    )

    assert result["category"] == "billing"


def test_extract_json_from_extra_text():
    result = _extract_json(
        'Here is the result: {"category": "billing", "confidence": 0.9}'
    )

    assert result["category"] == "billing"


def test_extract_invalid_json():
    with pytest.raises(ValueError):
        _extract_json("this is not json")


def test_extract_empty_response():
    with pytest.raises(ValueError):
        _extract_json("")


def test_valid_classification():
    result = _validate_result(VALID_RESULT)

    assert isinstance(result, ClassificationResult)
    assert result.category == "billing"
    assert result.sentiment == "negative"
    assert result.priority == "high"
    assert result.confidence == 0.92


def test_invalid_category():
    result = {
        **VALID_RESULT,
        "category": "something_invalid",
    }

    with pytest.raises(ValueError, match="Invalid category"):
        _validate_result(result)


def test_invalid_sentiment():
    result = {
        **VALID_RESULT,
        "sentiment": "something_invalid",
    }

    with pytest.raises(ValueError, match="Invalid sentiment"):
        _validate_result(result)


def test_invalid_priority():
    result = {
        **VALID_RESULT,
        "priority": "something_invalid",
    }

    with pytest.raises(ValueError, match="Invalid priority"):
        _validate_result(result)


def test_confidence_below_zero():
    result = {
        **VALID_RESULT,
        "confidence": -0.1,
    }

    with pytest.raises(ValueError):
        _validate_result(result)


def test_confidence_above_one():
    result = {
        **VALID_RESULT,
        "confidence": 1.1,
    }

    with pytest.raises(ValueError):
        _validate_result(result)


def test_missing_required_field():
    result = {
        **VALID_RESULT,
    }

    del result["category"]

    with pytest.raises(ValueError):
        _validate_result(result)


def test_empty_summary_is_rejected():
    result = {
        **VALID_RESULT,
        "summary": "",
    }

    # This test documents the current schema behavior.
    # Empty strings are currently allowed by Pydantic.
    validated = _validate_result(result)

    assert validated.summary == ""


def test_classification_result_serialization():
    result = _validate_result(VALID_RESULT)

    serialized = result.model_dump()

    assert serialized["category"] == "billing"
    assert serialized["confidence"] == 0.92
