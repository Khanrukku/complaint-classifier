# AI-Powered Unified Complaint & Analytics Platform

[![CI](https://github.com/Khanrukku/complaint-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Khanrukku/complaint-classifier/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Gemini](https://img.shields.io/badge/Gemini-AI-4285F4)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![License](https://img.shields.io/badge/License-MIT-green)

An **end-to-end NLP pipeline** for automatic complaint classification using **Google Gemini AI**. Classifies customer complaints across 8 categories in under one second, with sentiment analysis, priority scoring, and management dashboards.

---

## What It Does

| Input | Output |
|---|---|
| Raw complaint text | Category (8 types) |
| | Subcategory (specific label) |
| | Sentiment (positive/negative/neutral) |
| | Priority (low/medium/high/critical) |
| | Confidence score (0.0–1.0) |
| | One-sentence summary |
| | Suggested action for support team |

---

## Architecture

```
Customer Complaint (text)
         │
         ▼
  ┌──────────────┐
  │  FastAPI API  │  ← REST endpoints
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Gemini AI NLP │  ← Classification via prompt engineering
  │  Classifier   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐      ┌─────────────────┐
  │   SQLite /   │      │  Dashboard API  │
  │  PostgreSQL  │─────▶│  (Analytics &   │
  │  (Storage)   │      │   Reporting)    │
  └──────────────┘      └─────────────────┘
```

---

## Categories

| Category | Description |
|---|---|
| `billing` | Payment, charges, invoices |
| `technical_support` | App bugs, outages, errors |
| `delivery` | Shipping, tracking, delays |
| `product_quality` | Defects, wrong items |
| `customer_service` | Agent behaviour, response times |
| `refund_return` | Return requests, refund delays |
| `account_access` | Login, password, account issues |
| `other` | Everything else |

---

## Quick Start

```bash
git clone https://github.com/Khanrukku/complaint-classifier
cd complaint-classifier

cp .env.example .env
# Add your GEMINI_API_KEY to .env

pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at **http://localhost:8000/docs**

---

## API Usage

### Classify a complaint
```bash
curl -X POST http://localhost:8000/api/v1/complaints/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my subscription and nobody is responding to my emails!"}'
```

**Response:**
```json
{
  "id": "uuid",
  "category": "billing",
  "subcategory": "duplicate charge",
  "sentiment": "negative",
  "priority": "high",
  "confidence": 0.94,
  "summary": "Customer was charged twice and cannot reach support.",
  "suggested_action": "Review billing records and issue refund immediately.",
  "created_at": "2026-06-27T..."
}
```

### Bulk classify (concurrent)
```bash
curl -X POST http://localhost:8000/api/v1/complaints/bulk \
  -H "Content-Type: application/json" \
  -d '{"complaints": [{"text": "App keeps crashing"}, {"text": "Wrong item delivered"}]}'
```

### Dashboard stats
```bash
curl http://localhost:8000/api/v1/dashboard
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.12** | Core language |
| **Google Gemini AI** | NLP classification via prompt engineering |
| **FastAPI** | REST API with automatic OpenAPI docs |
| **SQLAlchemy (async)** | ORM with async PostgreSQL/SQLite support |
| **SQLite / PostgreSQL** | Persistent complaint storage |
| **Docker** | Containerisation |
| **GitHub Actions** | CI — automated tests on every push |
| **Pydantic v2** | Data validation |

---

## Key Engineering Decisions

**Prompt Engineering for structured output** — Gemini is prompted with a strict JSON schema and validation rules. A regex fallback handles malformed responses gracefully.

**Concurrent batch processing** — `asyncio.gather` runs multiple classifications in parallel, dramatically reducing latency for bulk submissions.

**Audit logging** — Every classification and resolution is logged to the `audit_log` table for compliance and debugging.

**Graceful degradation** — If Gemini fails, batch processing returns a default classification rather than failing the entire batch.

---

## Author

**Rukaiya Khan**
- GitHub: [@Khanrukku](https://github.com/Khanrukku)
- LinkedIn: [linkedin.com/in/rukaiyakhan](https://linkedin.com/in/rukaiyakhan)
