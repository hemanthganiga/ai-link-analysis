"""Orchestrates extraction, cached LLM analysis, and response construction."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.config import Settings
from app.schemas import (
    AnalysisResponse,
    Brand,
    BusinessAnalysis,
    ExtractedContent,
    ExtractionInfo,
    TargetAudience,
    TokenUsage,
)
from app.services.competitor_service import build_analysis_prompt
from app.services.llm_service import (
    LLMConfigurationError,
    LLMService,
    create_provider,
)
from app.services.website_extractor import (
    EmptyWebsiteContentError,
    WebsiteContent,
    WebsiteExtractor,
)


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
        if content.platform != "website":
            if (
                content.platform == "youtube"
                and content.content_type == "video"
                and content.source == "youtube_oembed"
            ):
                source_text = " ".join(
                    [content.title, content.metadata.get("author_name", "")]
                ).strip()
            else:
                source_text = " ".join(
                    [
                        content.meta_description,
                        *content.paragraphs,
                        *content.service_descriptions,
                    ]
                ).strip()
            if len(source_text) < 20:
                limitation = " ".join(content.limitations)
                detail = (
                    f"No usable public content was retrieved from {content.platform}."
                    if not limitation
                    else f"No usable public content was retrieved from {content.platform}. {limitation}"
                )
                raise EmptyWebsiteContentError(detail, content=content)
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
        analysis = _qualify_limited_social_analysis(content, analysis)
        analysis = _qualify_audience_geography(content, analysis)
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
            extraction=build_extraction_info(
                content, self.settings.max_content_chars
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


def build_extraction_info(
    content: WebsiteContent,
    max_content_chars: int = 18_000,
    *,
    extraction_status: str | None = None,
    error_detail: str | None = None,
) -> ExtractionInfo:
    """Build bounded response evidence without presenting metadata as page text."""
    remaining = max(0, max_content_chars)

    def clipped_items(values: list[str], max_items: int) -> list[str]:
        nonlocal remaining
        clipped: list[str] = []
        for value in values[:max_items]:
            if remaining <= 0:
                break
            item = value[:remaining]
            if item:
                clipped.append(item)
                remaining -= len(item)
        return clipped

    limitations = list(content.limitations)
    if error_detail and not any(
        limitation and limitation in error_detail for limitation in limitations
    ):
        limitations.append(error_detail)

    return ExtractionInfo(
        platform=content.platform,
        content_type=content.content_type,
        extraction_status=extraction_status or content.extraction_status,
        source=content.source,
        limitations=limitations,
        content=ExtractedContent(
            title=content.title[:500],
            meta_description=content.meta_description[:1000],
            headings=clipped_items(content.headings, 30),
            paragraphs=clipped_items(content.paragraphs, 100),
            service_descriptions=clipped_items(content.service_descriptions, 20),
            metadata={
                key[:100]: value[:500]
                for key, value in list(content.metadata.items())[:30]
            },
        ),
    )


def _qualify_limited_social_analysis(
    content: WebsiteContent, analysis: BusinessAnalysis
) -> BusinessAnalysis:
    """Remove unsupported profile facts from limited social and metadata-only analyses."""
    metadata_only_video = (
        content.platform == "youtube"
        and content.content_type == "video"
        and content.source == "youtube_oembed"
    )
    public_post = content.content_type == "public_post"
    limited_profile = (
        content.platform in {"instagram", "facebook", "x"}
        and content.extraction_status == "limited"
    )
    if not (metadata_only_video or public_post or limited_profile):
        return analysis

    if metadata_only_video:
        scope = "Only YouTube oEmbed title and uploader metadata was retrieved; the video itself was not reviewed."
        opportunity = "Provide a transcript or written video description for content-level analysis."
        limitation = "Industry, audience, services, and competitors cannot be verified from title/uploader metadata alone."
    elif content.platform == "x" and public_post:
        scope = "Analysis is limited to the extracted public post; profile details were not retrieved."
        opportunity = "Provide accessible public profile information for profile-level analysis."
        limitation = "No profile bio, account statistics, or independent competitor research was retrieved."
    else:
        scope = f"Public {content.platform} content was limited; profile facts cannot be verified from the retrieved source."
        opportunity = "Provide accessible public profile details for deeper analysis."
        limitation = "Industry, audience, profile details, and competitor facts are not verified."

    competitor_analysis = analysis.competitor_analysis.model_copy(
        update={
            "brand_overview": scope,
            "industry": "Unknown",
            "competitors": [],
            "brand_strengths": [],
            "opportunities": [opportunity],
            "research_limitations": limitation,
        }
    )
    recommended_services = [
        service.model_copy(update={"source_status": "recommended"})
        for service in analysis.top_services
    ]
    return analysis.model_copy(
        update={
            "competitor_analysis": competitor_analysis,
            "target_audience": TargetAudience(),
            "top_services": recommended_services,
        }
    )


def _qualify_audience_geography(
    content: WebsiteContent, analysis: BusinessAnalysis
) -> BusinessAnalysis:
    """Keep geographic-market claims only when the extracted page states them."""
    source_text = " ".join(
        [
            content.title,
            content.meta_description,
            *content.headings,
            *content.paragraphs,
            *content.service_descriptions,
        ]
    ).casefold()
    audience = analysis.target_audience
    verified_markets = [
        market
        for market in audience.geographic_markets
        if re.search(
            rf"(?<!\w){re.escape(market.casefold())}(?!\w)",
            source_text,
        )
    ]
    if verified_markets == audience.geographic_markets:
        return analysis

    return analysis.model_copy(
        update={
            "target_audience": audience.model_copy(
                update={"geographic_markets": verified_markets}
            )
        }
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
    if content.platform == "youtube" and content.content_type == "video":
        author = content.metadata.get("author_name", "")
        summary = f"YouTube video title: {page_title}."
        if author:
            summary += f" Uploader: {author}."
        summary += " Video content and transcript were not retrieved."
    elif content.platform == "x" and content.content_type == "public_post":
        summary = meta_description or "The public post text was not available."
    elif content.platform in {"instagram", "facebook", "x", "youtube"}:
        summary = meta_description or (
            f"Public {content.platform.title()} content for {name}; detailed public content was not available."
            if content.extraction_status == "limited"
            else (headings[0] if headings else f"Public {content.platform.title()} metadata")
        )
    else:
        summary = meta_description or (
            content.paragraphs[0]
            if content.paragraphs
            else (headings[0] if headings else "No descriptive text was retrieved.")
        )
    return name[:160], industry or "Not clearly identified", summary[:500]
