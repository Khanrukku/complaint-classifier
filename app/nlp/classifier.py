"""
Gemini-powered complaint classification.

The classifier:
- Sends complaints to Gemini for structured classification.
- Validates the returned JSON strictly.
- Retries transient Gemini failures.
- Limits concurrent batch requests.
- Returns explicit failures instead of silently hiding them.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from google import genai
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.models.schemas import CATEGORIES, PRIORITIES, SENTIMENTS


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Validated classification returned by Gemini."""

    category: str
    subcategory: str = ""
    sentiment: str
    priority: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    suggested_action: str

    def validate_business_rules(self) -> None:
        """Validate fields against application-level enums."""

        if self.category not in CATEGORIES:
            raise ValueError(
                f"Invalid category: {self.category!r}. "
                f"Expected one of: {CATEGORIES}"
            )

        if self.sentiment not in SENTIMENTS:
            raise ValueError(
                f"Invalid sentiment: {self.sentiment!r}. "
                f"Expected one of: {SENTIMENTS}"
            )

        if self.priority not in PRIORITIES:
            raise ValueError(
                f"Invalid priority: {self.priority!r}. "
                f"Expected one of: {PRIORITIES}"
            )


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


def _get_client() -> genai.Client:
    """Create a Gemini client using the configured API key."""

    if not settings.gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not configured.")

    return genai.Client(api_key=settings.gemini_api_key)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = f"""
You are an expert customer-support complaint classifier.

Classify the provided customer complaint and return ONLY a valid JSON object.

Required JSON schema:

{{
  "category": "one of {CATEGORIES}",
  "subcategory": "specific label describing the complaint",
  "sentiment": "one of {SENTIMENTS}",
  "priority": "one of {PRIORITIES}",
  "confidence": "number between 0.0 and 1.0",
  "summary": "one-sentence summary",
  "suggested_action": "recommended next action for the support team"
}}

Priority guidance:

- critical:
  safety issues, legal threats, complete service outages
- high:
  significant financial loss, repeated failures, urgent unresolved issues
- medium:
  service degradation, delayed resolution, meaningful inconvenience
- low:
  general feedback, minor inconvenience, non-urgent requests

Confidence should represent the model's estimated certainty in its classification.
It is NOT a statistically calibrated probability.

Return ONLY the JSON object.
Do not use markdown.
Do not add explanations.
""".strip()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_json(raw_text: str) -> dict[str, Any]:
    """
    Extract a JSON object from Gemini's response.

    Handles:
    - normal JSON
    - markdown JSON fences
    - JSON surrounded by accidental prose
    """

    if not raw_text or not raw_text.strip():
        raise ValueError("Gemini returned an empty response.")

    cleaned = raw_text.strip()

    # Remove markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response must be a JSON object.")

        return parsed

    except json.JSONDecodeError:
        pass

    # Fallback: locate the first JSON object.
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
# Validation
# ---------------------------------------------------------------------------


def _validate_result(
    result: dict[str, Any],
) -> ClassificationResult:
    """
    Validate Gemini's response.

    Invalid AI output is rejected instead of silently replaced with
    potentially misleading defaults.
    """

    try:
        classification = ClassificationResult.model_validate(result)

    except ValidationError as exc:
        raise ValueError(
            f"Gemini response failed schema validation: {exc}"
        ) from exc

    classification.validate_business_rules()

    return classification


# ---------------------------------------------------------------------------
# Single complaint classification
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def classify_complaint(text: str) -> dict[str, Any]:
    """
    Classify a single complaint using Gemini.

    Raises:
        EnvironmentError: Gemini API key is missing.
        ValueError: Gemini returns invalid or unusable data.
        Exception: Gemini/API errors after retries are exhausted.
    """

    if not text or not text.strip():
        raise ValueError("Complaint text cannot be empty.")

    client = _get_client()

    prompt = f"Complaint:\n{text.strip()}"

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.gemini_model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        )

    except Exception:
        # Let Tenacity retry transient provider/API failures.
        raise

    raw_text = getattr(response, "text", None)

    if not raw_text:
        raise ValueError("Gemini returned no text content.")

    result = _extract_json(raw_text)
    validated = _validate_result(result)

    return validated.model_dump()


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------


MAX_CONCURRENT_REQUESTS = 5


async def classify_batch(
    texts: list[str],
) -> list[dict[str, Any]]:
    """
    Classify multiple complaints concurrently.

    Concurrency is limited to avoid sending an uncontrolled number
    of simultaneous requests to Gemini.
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
        *(classify_with_limit(i, text) for i, text in enumerate(texts))
    )

    # Preserve the original input order.
    results.sort(key=lambda item: item[0])

    return [result for _, result in results]
