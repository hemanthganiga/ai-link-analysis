"""Analysis orchestration and cache behavior tests."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.schemas import BusinessAnalysis
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMProviderError, ProviderResult
from app.services.website_extractor import WebsiteContent


def _analysis_json() -> str:
    return json.dumps(
        {
            "competitor_analysis": {
                "brand_overview": "A business based on the submitted page.",
                "industry": "Technology",
                "competitors": [],
                "brand_strengths": ["Clear service description"],
                "opportunities": ["Explain customer outcomes"],
                "research_limitations": "External research was not performed.",
            },
            "target_search_queries": [
                {
                    "query": f"technology services {platform.lower()}",
                    "platforms": [platform],
                    "intent": "Service discovery",
                    "business_relevance": "Matches the business category.",
                    "target_audience": "Technology buyers",
                }
                for platform in ["Google", "Facebook", "Instagram", "LinkedIn", "X"]
            ],
            "seo_improvements": {
                "primary_keyword": "technology services",
                "secondary_keywords": ["business technology"],
                "long_tail_keywords": [],
                "recommended_title": "Technology services for business",
                "meta_description": "Technology services for business customers.",
                "heading_improvements": [],
                "content_optimization": [],
                "supporting_content": [],
                "blog_topics": [],
                "faq_recommendations": [],
                "technical_seo": [],
            },
            "top_services": [
                {
                    "service_name": f"Service {index}",
                    "description": "A recommended service.",
                    "target_industry": "Technology",
                    "target_customer": "Businesses",
                    "main_benefit": "Supports business needs.",
                    "source_status": "recommended",
                }
                for index in range(5)
            ],
            "target_audience": {
                "primary_audience": ["Technology buyers"],
                "secondary_audience": [],
                "industries": ["Technology"],
                "customer_profiles": [],
                "customer_pain_points": [],
                "customer_interests": [],
                "geographic_markets": [],
                "potential_requirements": [],
            },
        }
    )


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, url: str) -> WebsiteContent:
        self.calls += 1
        return WebsiteContent(
            url=url,
            title="Acme | Technology",
            meta_description="Technology services for growing businesses.",
            headings=["Acme technology services"],
            paragraphs=["Acme delivers software and technology services."],
        )


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, _: str) -> tuple[BusinessAnalysis, dict[str, int | None]]:
        self.calls += 1
        return (
            BusinessAnalysis.model_validate_json(_analysis_json()),
            {"input_tokens": 400, "output_tokens": 250, "total_tokens": 650},
        )


@pytest.mark.asyncio
async def test_analysis_service_returns_brand_insights_and_caches_llm_result() -> None:
    settings = Settings(
        llm_provider="anthropic",
        llm_model="claude-haiku-5-5",
        llm_api_key="test-key",
        max_content_chars=500,
    )
    extractor = FakeExtractor()
    llm = FakeLLM()
    service = AnalysisService(settings, extractor=extractor, llm_service=llm)

    first = await service.analyze("https://acme.example")
    second = await service.analyze("https://acme.example")

    assert first.brand.name == "Acme"
    assert first.brand.category == "Technology"
    assert len(first.analysis.target_search_queries) == 5
    assert first.usage.total_tokens == 650
    assert first.usage.cache_hit is False
    assert second.usage.cache_hit is True
    assert extractor.calls == 2
    assert llm.calls == 1


class RateLimitedLLM:
    async def analyze(self, _: str) -> tuple[BusinessAnalysis, dict[str, int | None]]:
        raise LLMProviderError("The LLM provider returned HTTP 429.")


@pytest.mark.asyncio
async def test_social_analysis_returns_explicit_fallback_when_llm_is_rate_limited() -> None:
    settings = Settings(llm_provider="groq", llm_model="openai/gpt-oss-20b", llm_api_key="test-key")
    extractor = FakeExtractor()

    async def extract_instagram(_: str) -> WebsiteContent:
        return WebsiteContent(
            url="https://www.instagram.com/crestspot",
            title="Instagram",
            meta_description="",
            platform="instagram",
            content_type="profile",
            extraction_status="limited",
            source="public_html",
            limitations=["Public HTML contained a login restriction."],
            metadata={"username": "crestspot"},
        )

    extractor.extract = extract_instagram  # type: ignore[method-assign]
    service = AnalysisService(settings, extractor=extractor, llm_service=RateLimitedLLM())

    result = await service.analyze("https://www.instagram.com/crestspot")

    assert result.brand.name == "@crestspot"
    assert result.brand.category == "Insufficient data available to identify industry." or result.brand.category == "Unknown"
    assert "provider rate limiting" in result.analysis.competitor_analysis.research_limitations.lower()
    assert len(result.analysis.target_search_queries) == 5
