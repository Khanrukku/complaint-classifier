"""
app/nlp/classifier.py
---------------------
Gemini-powered complaint classification.

Responsibilities:
- Send complaints to Gemini for classification.
- Parse and validate Gemini's JSON response.
- Enforce application-level classification rules.
- Retry transient Gemini/API failures.
- Limit concurrent requests during batch processing.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.models.schemas import (
    AIClassification,
    CATEGORIES,
    PRIORITIES,
    SENTIMENTS,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT_REQUESTS = 5


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


def _get_client() -> genai.Client:
    """Create and return a Gemini client."""

    if not settings.gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not configured.")

    return genai.Client(api_key=settings.gemini_api_key)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = f"""
You are an expert customer-support complaint classifier.

Your task is to classify the provided customer complaint.

Return ONLY a valid JSON object using exactly this structure:

{{
  "category": "one of {CATEGORIES}",
  "subcategory": "specific label describing the complaint",
  "sentiment": "one of {SENTIMENTS}",
  "priority": "one of {PRIORITIES}",
  "confidence": 0.0,
  "summary": "one-sentence summary of the complaint",
  "suggested_action": "recommended next action for the support team"
}}

Classification rules:

CATEGORY:
Choose exactly one category from:
{CATEGORIES}

SENTIMENT:
Choose exactly one:
{SENTIMENTS}

PRIORITY:
- critical: safety issues, legal threats, or complete service outages
- high: significant financial loss, repeated failures, or urgent unresolved issues
- medium: service degradation, delays, or meaningful inconvenience
- low: minor inconvenience, general feedback, or non-urgent requests

CONFIDENCE:
Return a number between 0.0 and 1.0 representing the model's
estimated certainty about the classification.

Important:
The confidence score is a model-estimated score and is NOT a
statistically calibrated probability.

Do not invent categories.
Do not return markdown.
Do not wrap the JSON in code fences.
Do not add explanations outside the JSON object.
""".strip()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _extract_json(raw_text: str) -> dict[str, Any]:
    """
    Extract a JSON object from Gemini's response.

    Handles:
    - standard JSON
    - JSON wrapped in Markdown code fences
    - JSON surrounded by accidental explanatory text
    """

    if not raw_text or not raw_text.strip():
        raise ValueError("Gemini returned an empty response.")

    cleaned = raw_text.strip()

    # Remove Markdown code fences if Gemini accidentally adds them.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    # First attempt: entire response is valid JSON.
    try:
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response must be a JSON object.")

        return parsed

    except json.JSONDecodeError:
        pass

    # Second attempt: find a JSON object inside surrounding text.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if not match:
        raise ValueError(
            f"Gemini returned invalid JSON: {cleaned[:200]!r}"
        )

    try:
        parsed = json.loads(match.group())

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned malformed JSON: {cleaned[:200]!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object.")

    return parsed


# ---------------------------------------------------------------------------
# Classification validation
# ---------------------------------------------------------------------------


def _validate_result(result: dict[str, Any]) -> AIClassification:
    """
    Validate Gemini's classification response.

    Schema validation checks:
    - required fields
    - data types
    - confidence range

    Business-rule validation checks:
    - category
    - sentiment
    - priority
    """

    try:
        classification = AIClassification.model_validate(result)

    except Exception as exc:
        raise ValueError(
            f"Gemini response failed schema validation: {exc}"
        ) from exc

    if classification.category not in CATEGORIES:
        raise ValueError(
            f"Invalid category: {classification.category!r}. "
            f"Expected one of: {CATEGORIES}"
        )

    if classification.sentiment not in SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment: {classification.sentiment!r}. "
            f"Expected one of: {SENTIMENTS}"
        )

    if classification.priority not in PRIORITIES:
        raise ValueError(
            f"Invalid priority: {classification.priority!r}. "
            f"Expected one of: {PRIORITIES}"
        )

    return classification


# ---------------------------------------------------------------------------
# Single complaint classification
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=8,
    ),
    reraise=True,
)
async def classify_complaint(text: str) -> dict[str, Any]:
    """
    Classify a single complaint using Gemini.

    Retries transient Gemini/API failures up to three times.

    Args:
        text: Customer complaint text.

    Returns:
        Validated classification dictionary.

    Raises:
        ValueError:
            If the complaint is empty or Gemini returns invalid data.
        EnvironmentError:
            If GEMINI_API_KEY is missing.
        Exception:
            If the Gemini request repeatedly fails.
    """

    if not text or not text.strip():
        raise ValueError("Complaint text cannot be empty.")

    client = _get_client()

    prompt = f"""
Complaint:

{text.strip()}
""".strip()

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.gemini_model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        )

    except Exception:
        # Allow Tenacity to retry transient provider/API errors.
        raise

    raw_text = getattr(response, "text", None)

    if not raw_text:
        raise ValueError("Gemini returned no text content.")

    parsed_result = _extract_json(raw_text)

    validated_result = _validate_result(parsed_result)

    return validated_result.model_dump()


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------


async def classify_batch(
    texts: list[str],
) -> list[dict[str, Any]]:
    """
    Classify multiple complaints concurrently.

    A semaphore limits concurrent Gemini requests to prevent an
    uncontrolled number of simultaneous API calls.

    Results preserve the same order as the input complaints.
    """

    if not texts:
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def classify_with_limit(
        index: int,
        text: str,
    ) -> tuple[int, dict[str, Any]]:

        async with semaphore:

            try:
                result = await classify_complaint(text)

                return index, result

            except Exception as exc:
                # Keep batch processing alive when one complaint fails.
                # Failed complaints are explicitly marked for manual review.
                return index, {
                    "category": "other",
                    "subcategory": "",
                    "sentiment": "neutral",
                    "priority": "medium",
                    "confidence": 0.0,
                    "summary": text[:100],
                    "suggested_action": "Manual review required.",
                    "error": str(exc),
                }

    results = await asyncio.gather(
        *(
            classify_with_limit(index, text)
            for index, text in enumerate(texts)
        )
    )

    # Restore original input order.
    results.sort(key=lambda item: item[0])

    return [
        result
        for _, result in results
    ]
