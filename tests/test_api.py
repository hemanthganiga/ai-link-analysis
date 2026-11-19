"""FastAPI integration tests with mocked website and model services."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app, get_analysis_service
from app.schemas import AnalysisResponse, Brand, BusinessAnalysis, CompetitorAnalysis, SEOImprovements, TargetAudience, SearchQuery, ServiceRecommendation, TokenUsage
from app.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMProviderError
from app.services.website_extractor import WebsiteContent, WebsiteTimeoutError


def _response() -> AnalysisResponse:
    return AnalysisResponse(
        website="https://example.com",
        brand=Brand(name="Example", category="Technology", summary="Technology services."),
        analysis=BusinessAnalysis(
            competitor_analysis=CompetitorAnalysis(
                brand_overview="Example provides technology services.",
                industry="Technology",
                competitors=[],
                brand_strengths=[],
                opportunities=[],
                research_limitations="Candidates are not independently researched.",
            ),
            target_search_queries=[
                SearchQuery(
                    query=f"technology service query {index}",
                    platforms=[["Google"], ["Facebook"], ["Instagram"], ["LinkedIn"], ["X"]][index],
                    intent="Service discovery",
                    business_relevance="Relevant",
                    target_audience="Companies",
                )
                for index in range(5)
            ],
            seo_improvements=SEOImprovements(
                primary_keyword="technology services",
                secondary_keywords=[],
                long_tail_keywords=[],
                recommended_title="Technology services",
                meta_description="Technology services for companies.",
            ),
            top_services=[
                ServiceRecommendation(
                    service_name=f"Service {index}",
                    description="Helpful service",
                    target_industry="Technology",
                    target_customer="Business",
                    main_benefit="Improves operations",
                    source_status="recommended",
                )
                for index in range(5)
            ],
            target_audience=TargetAudience(
                primary_audience=["Companies"],
                secondary_audience=[],
                industries=["Technology"],
                customer_profiles=[],
                customer_pain_points=[],
                customer_interests=[],
                geographic_markets=[],
                potential_requirements=[],
            ),
        ),
        usage=TokenUsage(provider="test", model="test-model", input_tokens=10, output_tokens=20),
    )


class FakeAnalysisService:
    async def analyze(self, url: str) -> AnalysisResponse:
        if url == "https://timeout.example":
            raise WebsiteTimeoutError("Website request timed out.")
        if url == "https://llm-failure.example":
            raise LLMProviderError("The LLM provider is unavailable.")
        if url == "https://unexpected.example":
            raise RuntimeError("secret implementation details")
        result = _response()
        return result.model_copy(update={"website": url})


app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService()
client = TestClient(app, raise_server_exceptions=False)


def test_root_health_and_swagger_are_available() -> None:
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/docs").status_code == 200


def test_post_returns_valid_five_section_response() -> None:
    response = client.post("/link-analysis", json={"url": "https://example.com"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["website"] == "https://example.com"
    assert len(payload["analysis"]["target_search_queries"]) == 5
    assert len(payload["analysis"]["top_services"]) == 5
    assert AnalysisResponse.model_validate(payload)


def test_invalid_and_missing_urls_are_rejected() -> None:
    assert client.post("/link-analysis", json={"url": "ftp://example.com"}).status_code == 422
    assert client.post("/link-analysis", json={}).status_code == 422
    assert client.post("/link-analysis", json={"url": "http://127.0.0.1"}).status_code == 422


def test_timeout_provider_failure_and_unexpected_errors_are_explicit() -> None:
    assert client.post("/link-analysis", json={"url": "https://timeout.example"}).status_code == 504
    failure = client.post("/link-analysis", json={"url": "https://llm-failure.example"})
    assert failure.status_code == 502
    assert "unavailable" in failure.json()["detail"]
    unexpected = client.post("/link-analysis", json={"url": "https://unexpected.example"})
    assert unexpected.status_code == 500
    assert unexpected.json()["detail"] == "An unexpected server error occurred."


def test_missing_provider_key_fails_before_website_fetch() -> None:
    class MustNotFetch:
        async def extract(self, _: str) -> WebsiteContent:
            raise AssertionError("website should not be fetched without an LLM key")

    service = AnalysisService(
        Settings(llm_provider="anthropic", llm_model="claude-haiku-5-5", llm_api_key=""),
        extractor=MustNotFetch(),
    )
    app.dependency_overrides[get_analysis_service] = lambda: service
    try:
        response = client.post("/link-analysis", json={"url": "https://example.com"})
    finally:
        app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService()
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
