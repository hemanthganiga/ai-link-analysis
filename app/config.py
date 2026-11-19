"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the API, extraction, caching, and LLM provider."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b").strip()
    llm_api_key: str = (
        os.getenv("LLM_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    ).strip()
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    website_timeout_seconds: float = float(os.getenv("WEBSITE_TIMEOUT_SECONDS", "12"))
    max_response_bytes: int = int(os.getenv("MAX_RESPONSE_BYTES", "2000000"))
    max_content_chars: int = int(os.getenv("MAX_CONTENT_CHARS", "18000"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
    cache_max_entries: int = int(os.getenv("CACHE_MAX_ENTRIES", "256"))

    @classmethod
    def from_env(cls) -> Settings:
        """Create a fresh configuration snapshot after loading ``.env``."""
        load_dotenv(override=False)
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b").strip(),
            llm_api_key=(
                os.getenv("LLM_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
            ).strip(),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
            website_timeout_seconds=float(os.getenv("WEBSITE_TIMEOUT_SECONDS", "12")),
            max_response_bytes=int(os.getenv("MAX_RESPONSE_BYTES", "2000000")),
            max_content_chars=int(os.getenv("MAX_CONTENT_CHARS", "18000")),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "1800")),
            cache_max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "256")),
        )


settings = Settings.from_env()
