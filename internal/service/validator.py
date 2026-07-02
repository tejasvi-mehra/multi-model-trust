"""Citation fetching, quote validation, and support-score metrics."""

import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from internal.framework.runner import map_bounded
from internal.service.schemas import (
    Citation,
    CitationValidation,
    ClaimValidation,
    ModelResponse,
)

DEFAULT_USER_AGENT = (
    "multi-model-trust/0.1.0 (+https://github.com/tejasvi-mehra/multi-model-trust)"
)


class UrlFetcher(Protocol):
    """Interface for fetching citation source documents."""

    async def fetch(self, url: str) -> str:
        """Return text content from a citation URL."""


@dataclass(frozen=True)
class HttpxUrlFetcher:
    """HTTPX-backed fetcher used by the API runtime."""

    timeout_seconds: float
    user_agent: str = DEFAULT_USER_AGENT

    async def fetch(self, url: str) -> str:
        """Fetch a URL and return its response body as text."""

        headers = {"User-Agent": self.user_agent}
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text


def _normalize_text(value: str) -> str:
    """Collapse whitespace for stable quote matching across HTML pages."""

    return re.sub(r"\s+", " ", value).strip().lower()


def _tokenize(value: str) -> set[str]:
    """Tokenize text for support-score overlap."""

    return set(re.findall(r"[a-z0-9]+", value.lower()))


def citation_support_score(statement: str, quote: str) -> float:
    """Score how well a quote substantiates a claim using token overlap."""

    statement_tokens = _tokenize(statement)
    quote_tokens = _tokenize(quote)
    if not statement_tokens or not quote_tokens:
        return 0.0
    return len(statement_tokens & quote_tokens) / len(
        statement_tokens | quote_tokens
    )


def _average(values: list[float]) -> float:
    """Return the mean of numeric values, or zero when empty."""

    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass
class CitationValidator:
    """Validates citations by fetching sources and scoring quote support."""

    fetcher: UrlFetcher
    concurrency: int

    async def validate(self, responses: list[ModelResponse]) -> list[ClaimValidation]:
        """Validate every citation attached to every model claim."""

        claim_jobs = [
            (response.model, claim.statement, citation)
            for response in responses
            for claim in response.claims
            for citation in claim.citations
        ]
        citation_results = await map_bounded(
            claim_jobs,
            self._validate_citation_job,
            self.concurrency,
        )

        by_claim: dict[tuple[str, str], list[CitationValidation]] = {}
        for model, statement, validation in citation_results:
            by_claim.setdefault((model, statement), []).append(validation)

        return [
            self._build_claim_validation(
                response.model,
                claim.statement,
                by_claim.get((response.model, claim.statement), []),
            )
            for response in responses
            for claim in response.claims
        ]

    def _build_claim_validation(
        self,
        model: str,
        statement: str,
        citations: list[CitationValidation],
    ) -> ClaimValidation:
        """Build claim-level citation status from citation checks."""

        if not citations:
            return ClaimValidation(
                model=model,
                statement=statement,
                citations=[],
                all_citations_valid=False,
                average_support_score=0.0,
                citation_status="MISSING",
                invalid_reason="missing citation",
            )

        average_support = round(
            _average([citation.support_score for citation in citations]),
            4,
        )
        if all(citation.unreachable for citation in citations):
            return ClaimValidation(
                model=model,
                statement=statement,
                citations=citations,
                all_citations_valid=False,
                average_support_score=average_support,
                citation_status="UNREACHABLE",
                invalid_reason="source not reachable",
            )
        if not all(result.valid for result in citations):
            return ClaimValidation(
                model=model,
                statement=statement,
                citations=citations,
                all_citations_valid=False,
                average_support_score=average_support,
                citation_status="INVALID",
                invalid_reason="one or more citations are invalid",
            )
        return ClaimValidation(
            model=model,
            statement=statement,
            citations=citations,
            all_citations_valid=True,
            average_support_score=average_support,
            citation_status="VALID",
        )

    async def _validate_citation_job(
        self,
        job: tuple[str, str, Citation],
    ) -> tuple[str, str, CitationValidation]:
        """Validate one citation and keep claim context for regrouping."""

        model, statement, citation = job
        support_score = round(citation_support_score(statement, citation.quote), 4)
        try:
            body = await self.fetcher.fetch(str(citation.url))
        except Exception:
            validation = CitationValidation(
                url=citation.url,
                quote=citation.quote,
                valid=False,
                quote_found=False,
                unreachable=True,
                support_score=support_score,
                reason="source not reachable",
            )
            return model, statement, validation

        quote_found = _normalize_text(citation.quote) in _normalize_text(body)
        validation = CitationValidation(
            url=citation.url,
            quote=citation.quote,
            valid=quote_found,
            quote_found=quote_found,
            unreachable=False,
            support_score=support_score,
            reason=None if quote_found else "quote not found in fetched source",
        )
        return model, statement, validation


def citation_precision(validations: list[ClaimValidation]) -> tuple[float, int, int]:
    """Calculate valid citations divided by attempted citations."""

    attempted = sum(len(validation.citations) for validation in validations)
    valid = sum(
        1
        for validation in validations
        for citation in validation.citations
        if citation.valid
    )
    if attempted == 0:
        return 0.0, valid, attempted
    return valid / attempted, valid, attempted


def unreachable_citation_count(validations: list[ClaimValidation]) -> int:
    """Count citations whose source could not be fetched."""

    return sum(
        1
        for validation in validations
        for citation in validation.citations
        if citation.unreachable
    )


def average_citation_support_score(validations: list[ClaimValidation]) -> float:
    """Return the mean support score across all attempted citations."""

    scores = [
        citation.support_score
        for validation in validations
        for citation in validation.citations
    ]
    return round(_average(scores), 4)
