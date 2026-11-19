"""Orchestrates extraction, cached LLM analysis, and response construction."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.config import Settings
from app.schemas import AnalysisResponse, Brand, ExtractionInfo, TokenUsage
from app.services.competitor_service import build_analysis_prompt
from app.services.llm_service import LLMConfigurationError, LLMService, create_provider
from app.services.website_extractor import WebsiteContent, WebsiteExtractor


@dataclass
class _CacheEntry:
    expires_at: float
    response: AnalysisResponse


class AnalysisService:
    """Coordinate website extraction and a cached, schema-validated LLM call."""

    def __init__(
        self,
        settings: Settings,
        extractor: WebsiteExtractor | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.settings = settings
        self.extractor = extractor or WebsiteExtractor(
            timeout_seconds=settings.website_timeout_seconds,
            max_response_bytes=settings.max_response_bytes,
            max_content_chars=settings.max_content_chars,
            apify_api_token=settings.apify_api_token,
            apify_instagram_actor_id=settings.apify_instagram_actor_id,
        )
        if llm_service is not None:
            self.llm_service = llm_service
        else:
            self.llm_service = LLMService(
                create_provider(settings), settings.llm_provider, settings.llm_model
            )
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    async def analyze(self, url: str) -> AnalysisResponse:
        """Analyze a URL and return the stable response contract."""
        if not self.settings.llm_api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is required. Add the selected provider's API key to the environment."
            )
        content = await self.extractor.extract(url)
        prompt = build_analysis_prompt(content, self.settings.max_content_chars)
        cache_key = hashlib.sha256(
            f"{self.settings.llm_provider}:{self.settings.llm_model}:{url}:{prompt}".encode()
        ).hexdigest()

        async with self._cache_lock:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached.model_copy(
                    update={"usage": cached.usage.model_copy(update={"cache_hit": True})}
                )

        analysis, usage = await self.llm_service.analyze(prompt)
        name, category, summary = _brand_details(content, analysis.competitor_analysis.industry)
        response = AnalysisResponse(
            website=url,
            brand=Brand(name=name, category=category, summary=summary),
            analysis=analysis,
            usage=TokenUsage(
                provider=self.settings.llm_provider,
                model=self.settings.llm_model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
            ),
            extraction=ExtractionInfo(
                platform=content.platform,
                content_type=content.content_type,
                extraction_status=content.extraction_status,
                source=content.source,
                limitations=content.limitations,
            ),
        )
        async with self._cache_lock:
            self._store(cache_key, response)
        return response

    def _get_cached(self, key: str) -> AnalysisResponse | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry.response

    def _store(self, key: str, response: AnalysisResponse) -> None:
        if self.settings.cache_ttl_seconds <= 0 or self.settings.cache_max_entries <= 0:
            return
        self._cache[key] = _CacheEntry(
            expires_at=time.monotonic() + self.settings.cache_ttl_seconds,
            response=response,
        )
        self._cache.move_to_end(key)
        while len(self._cache) > self.settings.cache_max_entries:
            self._cache.popitem(last=False)


def _brand_details(content: WebsiteContent, industry: str) -> tuple[str, str, str]:
    """Build conservative response-level brand details from page evidence."""
    username = content.metadata.get("username", "").strip()
    if content.platform in {"instagram", "x"} and username:
        name = f"@{username}"
    else:
        name = ""
    page_title = content.title.strip()
    meta_description = content.meta_description.strip()
    headings = content.headings
    if not name:
        name = page_title.split("|")[0].split("—")[0].split("–")[0].strip()
    if not name and headings:
        name = headings[0][:120]
    if not name:
        from urllib.parse import urlsplit

        name = urlsplit(content.url).hostname or "Unknown brand"
    summary = meta_description or (headings[0] if headings else "Business details inferred from website content.")
    return name[:160], industry or "Not clearly identified", summary[:500]
