"""Orchestrates extraction, cached LLM analysis, and response construction."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.config import Settings
from app.schemas import (
    AnalysisResponse,
    Brand,
    BusinessAnalysis,
    CompetitorAnalysis,
    ExtractionInfo,
    SearchQuery,
    SEOImprovements,
    ServiceRecommendation,
    TargetAudience,
    TokenUsage,
)
from app.services.competitor_service import build_analysis_prompt
from app.services.llm_service import (
    LLMConfigurationError,
    LLMProviderError,
    LLMService,
    create_provider,
)
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
            key_name = "GROQ_API_KEY or LLM_API_KEY" if self.settings.llm_provider == "groq" else "LLM_API_KEY"
            raise LLMConfigurationError(
                f"{key_name} is required. Add the selected provider's API key to the environment."
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

        try:
            analysis, usage = await self.llm_service.analyze(prompt)
        except LLMProviderError as exc:
            if "429" not in str(exc) and "rate" not in str(exc).lower():
                raise
            analysis = _degraded_social_analysis(content) if content.platform != "website" else _degraded_website_analysis(content)
            usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
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


def _degraded_social_analysis(content: WebsiteContent) -> BusinessAnalysis:
    """Create an explicitly limited, evidence-safe response during provider outages."""
    username = content.metadata.get("username", "").strip()
    identity = f"@{username}" if username else content.platform.title()
    limitations = "; ".join(content.limitations) or "The public source did not provide enough profile content."
    competitor = CompetitorAnalysis(
        brand_overview=f"Limited public metadata is available for {identity}; detailed brand facts could not be verified.",
        industry="Unknown",
        brand_strengths=[],
        opportunities=["Add a public bio, products, or services to the profile"],
        research_limitations=f"AI analysis was unavailable due to provider rate limiting. {limitations}",
    )
    platforms = ["Google", "Facebook", "Instagram", "LinkedIn", "X"]
    queries = [
        SearchQuery(
            query=f"{identity} official profile {platform}",
            platforms=[platform],
            intent="Brand discovery",
            business_relevance="Unverified brand and presence discovery",
            target_audience="People seeking the official profile",
        )
        for platform in platforms
    ]
    return BusinessAnalysis(
        competitor_analysis=competitor,
        target_search_queries=queries,
        seo_improvements=SEOImprovements(
            primary_keyword=identity,
            meta_description=f"Official profile for {identity}",
            secondary_keywords=[],
            long_tail_keywords=[],
            recommended_title=f"{identity} | Official Profile",
        ),
        top_services=[
            ServiceRecommendation(
                service_name="Verify public profile",
                description="Confirm the official profile, public bio, and offered services from authoritative sources.",
                target_industry="Unknown",
                target_customer="Potential customers",
                main_benefit="Prevents unsupported brand assumptions",
                source_status="recommended",
            )
            for _ in range(5)
        ],
        target_audience=TargetAudience(
            primary_audience=["Potential customers"],
            secondary_audience=[],
            industries=[],
            customer_profiles=[],
            customer_pain_points=[],
            customer_interests=[],
            geographic_markets=[],
            potential_requirements=[],
        ),
    )


def _degraded_website_analysis(content: WebsiteContent) -> BusinessAnalysis:
    """Create a minimal evidence-safe fallback for non-social pages during provider outages."""
    identity = content.title or content.url
    competitor = CompetitorAnalysis(
        brand_overview=f"Public metadata was extracted for {identity}, but AI analysis was unavailable.",
        industry="Unknown",
        brand_strengths=[],
        opportunities=["Retry analysis when the LLM provider is available"],
        research_limitations="Provider rate limiting prevented complete AI analysis.",
    )
    platforms = ["Google", "Facebook", "Instagram", "LinkedIn", "X"]
    queries = [
        SearchQuery(query=f"{identity} official information", platforms=[platform], intent="Brand discovery", business_relevance="Official source discovery", target_audience="Potential customers")
        for platform in platforms
    ]
    return BusinessAnalysis(
        competitor_analysis=competitor,
        target_search_queries=queries,
        seo_improvements=SEOImprovements(primary_keyword=identity[:200], meta_description=content.meta_description[:500], recommended_title=identity[:200]),
        top_services=[
            ServiceRecommendation(service_name="Verify official offerings", description="Confirm the organization's real services from authoritative page evidence.", target_industry="Unknown", target_customer="Potential customers", main_benefit="Avoid unsupported recommendations", source_status="recommended")
            for _ in range(5)
        ],
        target_audience=TargetAudience(primary_audience=["Potential customers"], secondary_audience=[], industries=[], customer_profiles=[], customer_pain_points=[], customer_interests=[], geographic_markets=[], potential_requirements=[]),
    )


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
    if content.platform in {"instagram", "facebook", "x", "youtube"}:
        summary = meta_description or (
            f"Public {content.platform.title()} profile for {name}; detailed public content was not available."
            if content.extraction_status == "limited"
            else (headings[0] if headings else f"Public {content.platform.title()} metadata")
        )
    else:
        summary = meta_description or (headings[0] if headings else "Business details inferred from website content.")
    return name[:160], industry or "Not clearly identified", summary[:500]
