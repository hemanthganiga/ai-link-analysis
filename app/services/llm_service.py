"""Provider-neutral asynchronous LLM integration with JSON validation and usage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.schemas import BusinessAnalysis

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Base class for provider setup and response errors."""


class LLMConfigurationError(LLMServiceError):
    """Raised when provider settings are incomplete or unsupported."""


class LLMProviderError(LLMServiceError):
    """Raised when the provider API request fails."""


class LLMInvalidResponseError(LLMServiceError):
    """Raised when a provider returns malformed or schema-invalid JSON."""


@dataclass(frozen=True)
class ProviderResult:
    """Raw JSON text and provider-reported token usage."""

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class Provider(Protocol):
    """Minimal interface implemented by each supported provider."""

    async def complete(self, prompt: str) -> ProviderResult: ...


class HTTPProvider:
    """HTTP-backed provider with shared timeout and response error handling."""

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is required. Add the selected provider's API key to the environment."
            )
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=True) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise LLMProviderError("The LLM provider request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            logger.warning("LLM provider returned HTTP %s", exc.response.status_code)
            raise LLMProviderError(
                f"The LLM provider returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("The LLM provider could not be reached.") from exc
        except ValueError as exc:
            raise LLMInvalidResponseError("The LLM provider returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise LLMInvalidResponseError("The LLM provider returned an unexpected response.")
        return data


class OpenAIProvider(HTTPProvider):
    """OpenAI Chat Completions adapter using JSON response mode."""

    async def complete(self, prompt: str) -> ProviderResult:
        response = await self.post_json(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        try:
            choice = response["choices"][0]["message"]
            content = choice["content"]
            usage = response.get("usage", {})
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            return ProviderResult(
                content=content,
                input_tokens=_optional_int(usage.get("prompt_tokens")),
                output_tokens=_optional_int(usage.get("completion_tokens")),
                total_tokens=_optional_int(usage.get("total_tokens")),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError("OpenAI returned an unexpected response shape.") from exc


class GroqProvider(HTTPProvider):
    """Groq Chat Completions adapter using OpenAI-compatible JSON output."""

    async def complete(self, prompt: str) -> ProviderResult:
        response = await self.post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        try:
            choice = response["choices"][0]["message"]
            content = choice["content"]
            usage = response.get("usage", {})
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            return ProviderResult(
                content=content,
                input_tokens=_optional_int(usage.get("prompt_tokens")),
                output_tokens=_optional_int(usage.get("completion_tokens")),
                total_tokens=_optional_int(usage.get("total_tokens")),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError("Groq returned an unexpected response shape.") from exc


class GeminiProvider(HTTPProvider):
    """Google Gemini generateContent adapter using JSON MIME output."""

    async def complete(self, prompt: str) -> ProviderResult:
        response = await self.post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.api_key},
            {
                "systemInstruction": {
                    "parts": [{"text": "Return valid JSON only, matching the requested schema."}]
                },
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
        )
        try:
            candidate = response["candidates"][0]["content"]["parts"]
            content = "".join(part["text"] for part in candidate if isinstance(part.get("text"), str))
            usage = response.get("usageMetadata", {})
            if not content:
                raise TypeError("candidate has no text")
            return ProviderResult(
                content=content,
                input_tokens=_optional_int(usage.get("promptTokenCount")),
                output_tokens=_optional_int(usage.get("candidatesTokenCount")),
                total_tokens=_optional_int(usage.get("totalTokenCount")),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError("Gemini returned an unexpected response shape.") from exc


class AnthropicProvider(HTTPProvider):
    """Anthropic Messages adapter that requests schema-constrained JSON output."""

    async def complete(self, prompt: str) -> ProviderResult:
        response = await self.post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            {
                "model": self.model,
                "max_tokens": 6000,
                "system": "Return valid JSON only, matching the requested schema.",
                "messages": [{"role": "user", "content": prompt}],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": BusinessAnalysis.model_json_schema(),
                    }
                },
            },
        )
        try:
            content_blocks = response["content"]
            content = "".join(
                block["text"] for block in content_blocks if block.get("type") == "text"
            )
            usage = response.get("usage", {})
            if not content:
                raise TypeError("response has no text")
            input_tokens = _optional_int(usage.get("input_tokens"))
            output_tokens = _optional_int(usage.get("output_tokens"))
            total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
            return ProviderResult(content, input_tokens, output_tokens, total)
        except (KeyError, TypeError) as exc:
            raise LLMInvalidResponseError("Anthropic returned an unexpected response shape.") from exc


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def create_provider(settings: Settings) -> Provider:
    """Build the selected provider adapter from application settings."""
    provider_type = settings.llm_provider
    provider_class: type[HTTPProvider]
    if provider_type == "openai":
        provider_class = OpenAIProvider
    elif provider_type == "groq":
        provider_class = GroqProvider
    elif provider_type == "gemini":
        provider_class = GeminiProvider
    elif provider_type == "anthropic":
        provider_class = AnthropicProvider
    else:
        raise LLMConfigurationError(
            "LLM_PROVIDER must be one of: groq, openai, gemini, anthropic."
        )
    return provider_class(settings.llm_api_key, settings.llm_model, settings.llm_timeout_seconds)


class LLMService:
    """Call a provider once and validate its output against the analysis model."""

    def __init__(self, provider: Provider, provider_name: str, model: str) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model = model

    async def analyze(self, prompt: str) -> tuple[BusinessAnalysis, dict[str, int | None]]:
        """Generate structured analysis and return validated output plus usage."""
        result = await self.provider.complete(prompt)
        try:
            parsed_json = json.loads(result.content)
            analysis = BusinessAnalysis.model_validate(parsed_json)
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMInvalidResponseError(
                "The LLM response was not valid analysis JSON matching the required schema."
            ) from exc
        usage = {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
        }
        return analysis, usage

