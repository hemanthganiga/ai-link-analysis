"""Pydantic request and response models for link analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.validators import validate_url_syntax


class AnalysisRequest(BaseModel):
    """A website URL submitted for business and marketing analysis."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=8, max_length=2048, description="HTTP or HTTPS website URL")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return validate_url_syntax(value)


class Brand(BaseModel):
    """Business identity extracted from the submitted website."""

    name: str
    category: str
    summary: str


class Competitor(BaseModel):
    """A competitor finding or an explicitly unverified research suggestion."""

    name: str
    website: str | None = None
    comparison: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses_or_gaps: list[str] = Field(default_factory=list)
    evidence_status: Literal["researched", "suggestion"]
    evidence: str


class CompetitorAnalysis(BaseModel):
    """Brand-versus-market observations and opportunities."""

    brand_overview: str
    industry: str
    competitors: list[Competitor] = Field(default_factory=list)
    brand_strengths: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    research_limitations: str


class SearchQuery(BaseModel):
    """A customer query recommendation for one or more platforms."""

    query: str
    platforms: list[Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]] = Field(
        min_length=1
    )
    intent: str
    business_relevance: str
    target_audience: str


class SEOImprovements(BaseModel):
    """Evidence-led on-page and technical search recommendations."""

    primary_keyword: str
    secondary_keywords: list[str] = Field(default_factory=list)
    long_tail_keywords: list[str] = Field(default_factory=list)
    recommended_title: str
    meta_description: str
    heading_improvements: list[str] = Field(default_factory=list)
    content_optimization: list[str] = Field(default_factory=list)
    supporting_content: list[str] = Field(default_factory=list)
    blog_topics: list[str] = Field(default_factory=list)
    faq_recommendations: list[str] = Field(default_factory=list)
    technical_seo: list[str] = Field(default_factory=list)


class ServiceRecommendation(BaseModel):
    """A website-observed service or clearly labeled service suggestion."""

    service_name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source_status: Literal["observed", "recommended"]


class TargetAudience(BaseModel):
    """Audience segments inferred from the source website."""

    primary_audience: list[str] = Field(default_factory=list)
    secondary_audience: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    customer_profiles: list[str] = Field(default_factory=list)
    customer_pain_points: list[str] = Field(default_factory=list)
    customer_interests: list[str] = Field(default_factory=list)
    geographic_markets: list[str] = Field(default_factory=list)
    potential_requirements: list[str] = Field(default_factory=list)


class BusinessAnalysis(BaseModel):
    """The five analysis sections required by the assessment."""

    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery] = Field(min_length=5, max_length=5)
    seo_improvements: SEOImprovements
    top_services: list[ServiceRecommendation] = Field(min_length=5, max_length=5)
    target_audience: TargetAudience

    @model_validator(mode="after")
    def require_all_search_platforms(self) -> BusinessAnalysis:
        """Require one or more recommendations for every requested platform."""
        required = {"Google", "Facebook", "Instagram", "LinkedIn", "X"}
        present = {platform for query in self.target_search_queries for platform in query.platforms}
        missing = required - present
        if missing:
            raise ValueError(f"Search queries must cover all platforms: {', '.join(sorted(missing))}.")
        return self


class TokenUsage(BaseModel):
    """Provider-reported usage; values are null when not returned by a provider."""

    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_hit: bool = False


class AnalysisResponse(BaseModel):
    """Validated API response containing brand details and five analyses."""

    success: Literal[True] = True
    website: str
    brand: Brand
    analysis: BusinessAnalysis
    usage: TokenUsage
