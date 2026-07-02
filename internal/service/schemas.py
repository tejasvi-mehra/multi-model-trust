"""Pydantic schemas for API requests, validation output, and consensus groups."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Citation(BaseModel):
    """A source reference that should support one model claim."""

    url: HttpUrl
    quote: str = Field(min_length=1)


class Claim(BaseModel):
    """A single factual statement emitted by a model."""

    statement: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class ModelResponse(BaseModel):
    """A normalized mock response from one target model."""

    model: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)


class TrustRequest(BaseModel):
    """Input payload for one multi-model orchestration run."""

    prompt: str = Field(min_length=1)
    responses: list[ModelResponse] = Field(min_length=2)

    @field_validator("responses")
    @classmethod
    def require_distinct_models(
        cls, responses: list[ModelResponse]
    ) -> list[ModelResponse]:
        """Reject duplicate model names so agreement is measured across models."""

        models = [response.model for response in responses]
        if len(models) != len(set(models)):
            raise ValueError("responses must use distinct model names")
        return responses


class CitationValidation(BaseModel):
    """Validation result for one citation URL and quote pair."""

    url: HttpUrl
    quote: str
    valid: bool
    quote_found: bool = False
    unreachable: bool = False
    support_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: str | None = None


class ClaimValidation(BaseModel):
    """Validation result for one claim and all of its citations."""

    model: str
    statement: str
    citations: list[CitationValidation]
    all_citations_valid: bool
    average_support_score: float = Field(ge=0.0, le=1.0, default=0.0)
    citation_status: Literal["VALID", "INVALID", "MISSING", "UNREACHABLE"]
    invalid_reason: str | None = None


ConsensusLabel = Literal["AGREE", "DISAGREE", "PARTIAL", "INVALID"]
CitationStatus = Literal["VALID", "INVALID", "MISSING", "UNREACHABLE"]
JudgeDecision = Literal["AGREE", "DISAGREE", "NEUTRAL"]


class ConsensusMember(BaseModel):
    """One model claim inside a consensus group."""

    model: str
    statement: str
    citation_urls: list[HttpUrl]
    citation_status: CitationStatus = "MISSING"
    citation_errors: list[str] = Field(default_factory=list)


class CrossLlmJudgePrompt(BaseModel):
    """Sample judge prompt emitted when borderline comparison is enabled."""

    model_name: str
    prompt: str
    decision: JudgeDecision


class ConsensusGroup(BaseModel):
    """Claims grouped by similarity and labeled by relationship."""

    label: ConsensusLabel
    summary: str
    members: list[ConsensusMember]
    invalidated_members: list[ConsensusMember] = Field(default_factory=list)
    cross_llm_judge: CrossLlmJudgePrompt | None = None


class Metrics(BaseModel):
    """Lightweight quality metrics for the submitted payload."""

    citation_precision: float
    validated_citations: int
    attempted_citations: int
    unreachable_citations: int
    average_citation_support_score: float
    agreement_groups: int
    disagreement_groups: int
    partial_groups: int
    invalid_response_groups: int


class TrustResponse(BaseModel):
    """Full response returned by the trust orchestration endpoint."""

    prompt: str
    validations: list[ClaimValidation]
    consensus: list[ConsensusGroup]
    metrics: Metrics
