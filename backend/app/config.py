"""
Application configuration using pydantic-settings.

All values can be overridden via environment variables or a .env file.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM — Groq ────────────────────────────────────────────────────────────
    groq_api_key: str = Field("", description="Groq API key (free at console.groq.com)")
    llm_model: str = Field(
        "llama-3.1-8b-instant",
        description="Groq chat model. Options: llama-3.1-8b-instant, llama3-70b-8192, mixtral-8x7b-32768",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(5, description="Number of chunks to retrieve")
    chunk_size: int = Field(800, description="Target chunk size in characters")
    chunk_overlap: int = Field(100, description="Overlap between consecutive chunks")

    # ── API ───────────────────────────────────────────────────────────────────
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "https://tech-stalwarts-rag-chatbot.vercel.app",
        ],
        description="Allowed CORS origins",
    )

    # ── Environment ───────────────────────────────────────────────────────────
    environment: str = Field("development", description="Runtime environment")
    log_level: str = Field("INFO", description="Logging level")


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()


# Module-level alias for convenience
settings: Settings = get_settings()
