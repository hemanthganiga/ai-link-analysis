"""Tests for provider adapters and malformed/provider-failure handling."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.services.llm_service import (
    GeminiProvider,
    GroqProvider,
    AnthropicProvider,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMService,
    OpenAIProvider,
    ProviderResult,
    create_provider,
)
from app.schemas import BusinessAnalysis


def _valid_analysis_json() -> str:
    empty_competitor = {
        "brand_overview": "A software business.",
        "industry": "Software",
        "competitors": [],
        "brand_strengths": [],
        "opportunities": [],
        "research_limitations": "No independent competitor research was performed.",
    }
    return json.dumps(
        {
            "competitor_analysis": empty_competitor,
            "target_search_queries": [
                {
                    "query": f"business software query {index}",
                    "platforms": [["Google"], ["Facebook"], ["Instagram"], ["LinkedIn"], ["X"]][index],
                    "intent": "service discovery",
                    "business_relevance": "Relevant",
                    "target_audience": "Small businesses",
                }
                for index in range(5)
            ],
            "seo_improvements": {
                "primary_keyword": "business software",
                "secondary_keywords": [],
                "long_tail_keywords": [],
                "recommended_title": "Business software",
                "meta_description": "Business software solutions.",
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
                    "description": "A useful service.",
                    "target_industry": "Business",
                    "target_customer": "Companies",
                    "main_benefit": "Saves time.",
                    "source_status": "recommended",
                }
                for index in range(5)
            ],
            "target_audience": {
                "primary_audience": [],
                "secondary_audience": [],
                "industries": [],
                "customer_profiles": [],
                "customer_pain_points": [],
                "customer_interests": [],
                "geographic_markets": [],
                "potential_requirements": [],
            },
        }
    )


class FakeProvider:
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(self, _: str) -> ProviderResult:
        return ProviderResult(self.content, 80, 120, 200)


@pytest.mark.asyncio
async def test_llm_service_validates_structured_output_and_usage() -> None:
    service = LLMService(FakeProvider(_valid_analysis_json()), "fake", "test-model")
    analysis, usage = await service.analyze("prompt")
    assert isinstance(analysis, BusinessAnalysis)
    assert len(analysis.target_search_queries) == 5
    assert usage == {"input_tokens": 80, "output_tokens": 120, "total_tokens": 200}


@pytest.mark.asyncio
async def test_malformed_or_wrong_shape_llm_json_is_rejected() -> None:
    service = LLMService(FakeProvider("{not json"), "fake", "test-model")
    with pytest.raises(LLMInvalidResponseError, match="not valid analysis JSON"):
        await service.analyze("prompt")
    service = LLMService(FakeProvider("{}"), "fake", "test-model")
    with pytest.raises(LLMInvalidResponseError):
        await service.analyze("prompt")


def test_provider_configuration_is_explicit() -> None:
    provider = create_provider(Settings(llm_provider="openai", llm_api_key=""))
    with pytest.raises(LLMConfigurationError, match="LLM_PROVIDER"):
        create_provider(Settings(llm_provider="unknown", llm_api_key="key"))
    assert isinstance(provider, OpenAIProvider)
    groq_provider = create_provider(Settings(llm_provider="groq", llm_api_key="groq-key"))
    assert isinstance(groq_provider, GroqProvider)


@pytest.mark.asyncio
async def test_missing_api_key_is_reported_when_analysis_is_requested() -> None:
    provider = OpenAIProvider("", "gpt-test", 2)
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        await provider.complete("prompt")


@pytest.mark.asyncio
async def test_openai_adapter_parses_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider("test-key", "gpt-test", 2)

    async def fake_post_json(*_: object, **__: object) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(provider, "post_json", fake_post_json)
    result = await provider.complete("prompt")
    assert result.content == '{"ok": true}'
    assert result.total_tokens == 15


@pytest.mark.asyncio
async def test_groq_adapter_parses_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GroqProvider("test-key", "openai/gpt-oss-20b", 2)

    async def fake_post_json(*_: object, **__: object) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        }

    monkeypatch.setattr(provider, "post_json", fake_post_json)
    result = await provider.complete("prompt")
    assert result.content == '{"ok": true}'
    assert result.total_tokens == 20


@pytest.mark.asyncio
async def test_gemini_adapter_parses_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GeminiProvider("test-key", "gemini-test", 2)

    async def fake_post_json(*_: object, **__: object) -> dict[str, object]:
        return {
            "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 6,
                "totalTokenCount": 17,
            },
        }

    monkeypatch.setattr(provider, "post_json", fake_post_json)
    result = await provider.complete("prompt")
    assert result.content == '{"ok": true}'
    assert result.input_tokens == 11


@pytest.mark.asyncio
async def test_anthropic_adapter_uses_json_schema_and_parses_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicProvider("test-key", "claude-haiku-5-5", 2)
    captured_payload: dict[str, object] = {}

    async def fake_post_json(
        _: str, __: dict[str, str], payload: dict[str, object]
    ) -> dict[str, object]:
        captured_payload.update(payload)
        return {
            "content": [{"type": "text", "text": _valid_analysis_json()}],
            "usage": {"input_tokens": 12, "output_tokens": 18},
        }

    monkeypatch.setattr(provider, "post_json", fake_post_json)
    result = await provider.complete("prompt")
    assert result.input_tokens == 12
    assert result.output_tokens == 18
    output_config = captured_payload.get("output_config")
    assert isinstance(output_config, dict)
    output_format = output_config.get("format")
    assert isinstance(output_format, dict)
    assert output_format.get("type") == "json_schema"


@pytest.mark.asyncio
async def test_provider_http_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider("test-key", "gpt-test", 2)

    async def fail(*_: object, **__: object) -> dict[str, object]:
        raise LLMProviderError("provider unavailable")

    monkeypatch.setattr(provider, "post_json", fail)
    with pytest.raises(LLMProviderError, match="unavailable"):
        await provider.complete("prompt")
