"""
app/core/config.py
------------------
Centralised settings loaded from .env
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    gemini_api_key:       str  = os.getenv("GEMINI_API_KEY", "")
    gemini_model:         str  = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    database_url:         str  = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./complaints.db")
    api_host:             str  = os.getenv("API_HOST", "0.0.0.0")
    api_port:             int  = int(os.getenv("API_PORT", 8000))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    max_text_length:      int  = int(os.getenv("MAX_TEXT_LENGTH", 2000))


settings = Settings()
