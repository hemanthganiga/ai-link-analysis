"""Analysis orchestration and cache behavior tests."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.schemas import BusinessAnalysis
from app.services.analysis_service import AnalysisService, _brand_details
from app.services.llm_service import LLMProviderError
from app.services.website_extractor import EmptyWebsiteContentError, WebsiteContent


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
    assert first.extraction is not None
    assert first.extraction.content.title == "Acme | Technology"
    assert first.extraction.content.paragraphs == [
        "Acme delivers software and technology services."
    ]
    assert first.usage.cache_hit is False
    assert second.usage.cache_hit is True
    assert extractor.calls == 2
    assert llm.calls == 1


class RateLimitedLLM:
    async def analyze(self, _: str) -> tuple[BusinessAnalysis, dict[str, int | None]]:
        raise LLMProviderError("The LLM provider returned HTTP 429.")


@pytest.mark.asyncio
async def test_rate_limited_llm_error_is_not_replaced_with_fabricated_analysis() -> None:
    settings = Settings(llm_provider="groq", llm_model="openai/gpt-oss-20b", llm_api_key="test-key")
    extractor = FakeExtractor()

    async def extract_instagram(_: str) -> WebsiteContent:
        return WebsiteContent(
            url="https://www.instagram.com/crestspot",
            title="CrestSpot",
            meta_description="CrestSpot is an independent creative studio for growing brands.",
            platform="instagram",
            content_type="profile",
            extraction_status="success",
            source="public_html",
            paragraphs=["CrestSpot is an independent creative studio for growing brands."],
            limitations=["Public HTML contained a login restriction."],
            metadata={"username": "crestspot"},
        )

    extractor.extract = extract_instagram  # type: ignore[method-assign]
    service = AnalysisService(settings, extractor=extractor, llm_service=RateLimitedLLM())

    with pytest.raises(LLMProviderError, match="HTTP 429"):
        await service.analyze("https://www.instagram.com/crestspot")


@pytest.mark.asyncio
async def test_social_page_without_public_content_is_rejected_before_llm() -> None:
    settings = Settings(llm_provider="groq", llm_model="openai/gpt-oss-20b", llm_api_key="test-key")
    extractor = FakeExtractor()

    async def extract_blocked_profile(_: str) -> WebsiteContent:
        return WebsiteContent(
            url="https://www.instagram.com/crestspot",
            title="@crestspot",
            meta_description="",
            platform="instagram",
            content_type="profile",
            extraction_status="limited",
            limitations=["The public page returned HTTP 403."],
            metadata={"username": "crestspot"},
        )

    extractor.extract = extract_blocked_profile  # type: ignore[method-assign]
    llm = FakeLLM()
    service = AnalysisService(settings, extractor=extractor, llm_service=llm)

    with pytest.raises(
        EmptyWebsiteContentError, match="No usable public content"
    ) as error:
        await service.analyze("https://www.instagram.com/crestspot")
    assert llm.calls == 0
    assert error.value.content is not None
    assert error.value.content.metadata["username"] == "crestspot"


def _overclaim_analysis() -> BusinessAnalysis:
    payload = json.loads(_analysis_json())
    payload["competitor_analysis"]["competitors"] = [
        {
            "name": "Invented competitor",
            "website": None,
            "comparison": "Unsupported comparison",
            "strengths": [],
            "weaknesses_or_gaps": [],
            "evidence_status": "suggestion",
            "evidence": "AI-generated and not independently verified.",
        }
    ]
    payload["target_audience"]["primary_audience"] = ["Unverified audience"]
    for service in payload["top_services"]:
        service["source_status"] = "observed"
    return BusinessAnalysis.model_validate(payload)


@pytest.mark.parametrize(
    ("content", "expected_prompt_rule"),
    [
        (
            WebsiteContent(
                url="https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                title="A public demonstration video",
                meta_description="",
                platform="youtube",
                content_type="video",
                extraction_status="limited",
                source="youtube_oembed",
                limitations=["The video transcript was not retrieved."],
                metadata={"author_name": "Example Channel"},
            ),
            "No video, audio, captions, or transcript was retrieved",
        ),
        (
            WebsiteContent(
                url="https://x.com/jack/status/20",
                title="Post by jack",
                meta_description="just setting up my twttr",
                paragraphs=["just setting up my twttr"],
                platform="x",
                content_type="public_post",
                extraction_status="limited",
                limitations=["Profile information was restricted."],
                metadata={
                    "username": "jack",
                    "username_source": "submitted_url_path",
                },
            ),
            "not a profile URL",
        ),
    ],
)
@pytest.mark.asyncio
async def test_limited_social_analysis_exposes_evidence_and_removes_unsupported_facts(
    content: WebsiteContent, expected_prompt_rule: str
) -> None:
    class FixedExtractor:
        async def extract(self, _: str) -> WebsiteContent:
            return content

    class OverclaimLLM:
        prompt = ""

        async def analyze(
            self, prompt: str
        ) -> tuple[BusinessAnalysis, dict[str, int | None]]:
            self.prompt = prompt
            return (
                _overclaim_analysis(),
                {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            )

    llm = OverclaimLLM()
    service = AnalysisService(
        Settings(
            llm_provider="groq",
            llm_model="openai/gpt-oss-20b",
            llm_api_key="test-key",
        ),
        extractor=FixedExtractor(),
        llm_service=llm,
    )

    result = await service.analyze(content.url)

    assert expected_prompt_rule in llm.prompt
    assert result.brand.category == "Unknown"
    assert result.analysis.competitor_analysis.industry == "Unknown"
    assert result.analysis.competitor_analysis.competitors == []
    assert result.analysis.competitor_analysis.brand_strengths == []
    assert all(
        not values for values in result.analysis.target_audience.model_dump().values()
    )
    assert all(
        service.source_status == "recommended"
        for service in result.analysis.top_services
    )
    assert result.extraction is not None
    assert result.extraction.extraction_status == "limited"
    assert result.extraction.content.title == content.title
    if content.platform == "youtube":
        assert result.extraction.content.metadata["author_name"] == "Example Channel"
        assert "transcript were not retrieved" in result.brand.summary
    else:
        assert "just setting up my twttr" in llm.prompt
        assert "jack" not in llm.prompt
        assert "https://x.com/" not in llm.prompt
        assert "Metadata:" not in llm.prompt
        assert result.extraction.content.paragraphs == ["just setting up my twttr"]
        assert result.brand.summary == "just setting up my twttr"


def test_website_summary_uses_extracted_text_instead_of_a_generic_fallback() -> None:
    content = WebsiteContent(
        url="https://example.com/",
        title="Example",
        meta_description="",
        paragraphs=["This exact paragraph was retrieved from the page."],
    )

    _, _, summary = _brand_details(content, "Unknown")

    assert summary == "This exact paragraph was retrieved from the page."
