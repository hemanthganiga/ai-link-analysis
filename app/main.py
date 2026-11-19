"""FastAPI application exposing website link analysis endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings
from app.schemas import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.llm_service import (
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMProviderError,
)
from app.services.website_extractor import (
    EmptyWebsiteContentError,
    WebsiteHTTPError,
    WebsiteTimeoutError,
)
from app.utils.validators import UnsafeWebsiteURLError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.analysis_service = AnalysisService(Settings.from_env())
    yield


app = FastAPI(
    title="AI Website Link Analysis API",
    description="Extracts website content and generates evidence-qualified business and SEO insights.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_analysis_service(request: Request) -> AnalysisService:
    """Resolve the application-scoped analysis service."""
    return request.app.state.analysis_service


@app.exception_handler(UnsafeWebsiteURLError)
async def unsafe_url_handler(_: Request, exc: UnsafeWebsiteURLError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(WebsiteTimeoutError)
async def website_timeout_handler(_: Request, exc: WebsiteTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": str(exc)})


@app.exception_handler(WebsiteHTTPError)
async def website_error_handler(_: Request, exc: WebsiteHTTPError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(EmptyWebsiteContentError)
async def empty_content_handler(_: Request, exc: EmptyWebsiteContentError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(LLMConfigurationError)
async def llm_config_handler(_: Request, exc: LLMConfigurationError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMProviderError)
async def llm_provider_handler(_: Request, exc: LLMProviderError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(LLMInvalidResponseError)
async def llm_response_handler(_: Request, exc: LLMInvalidResponseError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected application error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )


@app.get("/", tags=["status"])
async def root() -> dict[str, str]:
    """Describe the API and link to its interactive documentation."""
    return {"name": "AI Website Link Analysis API", "status": "ok", "docs": "/docs"}


@app.get("/health", tags=["status"])
async def health() -> dict[str, str]:
    """Return a lightweight liveness status."""
    return {"status": "healthy"}


@app.post(
    "/link-analysis",
    response_model=AnalysisResponse,
    tags=["analysis"],
    summary="Analyze a public company website",
)
async def link_analysis(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """Fetch a public website and return five structured business insight sections."""
    return await service.analyze(request.url)
