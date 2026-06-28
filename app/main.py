"""
app/main.py
-----------
FastAPI application entrypoint.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.models.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AI Complaint Classifier starting up...")
    await init_db()
    print("✅ Database ready")
    yield
    print("👋 Shutting down")


app = FastAPI(
    title       = "AI-Powered Complaint Classification Platform",
    description = (
        "End-to-end NLP pipeline for automatic complaint classification "
        "using Google Gemini AI. Classifies complaints across 8 categories "
        "with sentiment analysis, priority scoring, and management dashboards."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
