"""
app/nlp/classifier.py
----------------------
Gemini-powered complaint classifier.

For each complaint it returns:
  - category        (one of 8 predefined categories)
  - subcategory     (freeform, more specific)
  - sentiment       (positive / negative / neutral)
  - priority        (low / medium / high / critical)
  - confidence      (0.0 – 1.0)
  - summary         (one-sentence summary)
  - suggested_action (what the support team should do)

Uses structured JSON output via prompt engineering.
Falls back gracefully if Gemini returns malformed JSON.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.models.schemas import CATEGORIES, PRIORITIES, SENTIMENTS


# ── Gemini client ─────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY not set.")
    return genai.Client(api_key=settings.gemini_api_key)


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are an expert customer support complaint classifier.

Given a complaint text, respond ONLY with a valid JSON object — no prose, no markdown.

JSON schema:
{{
  "category": one of {CATEGORIES},
  "subcategory": "more specific label (string)",
  "sentiment": one of {SENTIMENTS},
  "priority": one of {PRIORITIES},
  "confidence": float between 0.0 and 1.0,
  "summary": "one sentence summary of the complaint",
  "suggested_action": "what the support team should do next"
}}

Priority rules:
  - critical: legal threats, safety issues, complete service outage
  - high: significant financial loss, repeated failures, very angry customer
  - medium: service degradation, delayed resolution
  - low: general feedback, minor inconvenience

Respond with ONLY the JSON object. No explanation.
""".strip()


# ── Classifier ────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def classify_complaint(text: str) -> dict[str, Any]:
    """
    Classify a single complaint using Gemini.

    Returns a dict with all classification fields.
    Raises ValueError if classification fails after retries.
    """
    client  = _get_client()
    loop    = asyncio.get_event_loop()
    prompt  = f"Complaint:\n{text.strip()}"

    raw = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=settings.gemini_model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        )
    )

    raw_text = raw.text.strip()

    # Strip markdown fences if present
    raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().strip("`").strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try extracting JSON block from response
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"Gemini returned non-JSON: {raw_text[:200]}")

    # Validate and sanitize fields
    result["category"]   = result.get("category", "other")
    if result["category"] not in CATEGORIES:
        result["category"] = "other"

    result["sentiment"]  = result.get("sentiment", "neutral")
    if result["sentiment"] not in SENTIMENTS:
        result["sentiment"] = "neutral"

    result["priority"]   = result.get("priority", "medium")
    if result["priority"] not in PRIORITIES:
        result["priority"] = "medium"

    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.7))))
    result["summary"]    = result.get("summary", text[:100])
    result["suggested_action"] = result.get("suggested_action", "Review and respond to customer.")
    result["subcategory"] = result.get("subcategory", "")

    return result


async def classify_batch(texts: list[str]) -> list[dict[str, Any]]:
    """
    Classify multiple complaints concurrently.
    Uses asyncio.gather for parallel processing.
    """
    tasks = [classify_complaint(text) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Return a default classification on failure
            processed.append({
                "category":        "other",
                "subcategory":     "",
                "sentiment":       "neutral",
                "priority":        "medium",
                "confidence":      0.0,
                "summary":         texts[i][:100],
                "suggested_action": "Manual review required.",
                "error":           str(result),
            })
        else:
            processed.append(result)

    return processed
