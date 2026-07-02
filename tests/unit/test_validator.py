import pytest

from internal.service.schemas import ModelResponse
from internal.service.validator import (
    CitationValidator,
    citation_precision,
    citation_support_score,
    average_citation_support_score,
    unreachable_citation_count,
)
from tests.fixtures.prefetched_fetcher import prefetched_fetcher


@pytest.mark.asyncio
async def test_validator_normalizes_whitespace_when_matching_quotes() -> None:
    """Verify citation matching tolerates page whitespace differences."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[
                {
                    "statement": "Docs are available.",
                    "citations": [
                        {
                            "url": "https://fastapi.tiangolo.com/",
                            "quote": "Interactive API docs",
                        }
                    ],
                }
            ],
        )
    ]
    validator = CitationValidator(fetcher=prefetched_fetcher(), concurrency=1)

    validations = await validator.validate(responses)

    assert validations[0].all_citations_valid is True
    assert validations[0].citations[0].valid is True
    assert validations[0].citations[0].quote_found is True
    assert validations[0].citation_status == "VALID"
    assert validations[0].citations[0].support_score >= 0.0


@pytest.mark.asyncio
async def test_validator_marks_failed_fetch_as_unreachable() -> None:
    """Verify missing prefetched pages become unreachable citation results."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[
                {
                    "statement": "Docs are available.",
                    "citations": [
                        {
                            "url": "https://www.python.org/doc/essays/trust-mvp-unreachable-page/",
                            "quote": "Interactive API docs",
                        }
                    ],
                }
            ],
        )
    ]
    validator = CitationValidator(fetcher=prefetched_fetcher(), concurrency=1)

    validations = await validator.validate(responses)

    assert validations[0].all_citations_valid is False
    assert validations[0].citations[0].valid is False
    assert validations[0].citations[0].unreachable is True
    assert validations[0].citation_status == "UNREACHABLE"
    assert validations[0].citations[0].reason == "source not reachable"


@pytest.mark.asyncio
async def test_validator_marks_missing_citation_as_invalid_claim() -> None:
    """Verify claims without citations are marked as missing citation."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[{"statement": "Docs are available.", "citations": []}],
        )
    ]
    validator = CitationValidator(fetcher=prefetched_fetcher(), concurrency=1)

    validations = await validator.validate(responses)

    assert validations[0].all_citations_valid is False
    assert validations[0].citation_status == "MISSING"
    assert validations[0].invalid_reason == "missing citation"


def test_citation_support_score_measures_claim_quote_overlap() -> None:
    """Verify support score rises when quote substantiates the claim."""

    low = citation_support_score("Redis caches pages.", "Interactive API docs")
    high = citation_support_score(
        "FastAPI supports automatic API docs.",
        "Automatic API docs for FastAPI services.",
    )

    assert high > low


@pytest.mark.asyncio
async def test_validator_populates_support_and_unreachable_metrics() -> None:
    """Verify aggregate metrics include support score and unreachable counts."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[
                {
                    "statement": "FastAPI supports automatic API docs.",
                    "citations": [
                        {
                            "url": "https://fastapi.tiangolo.com/features/",
                            "quote": "Automatic docs",
                        },
                        {
                            "url": "https://www.python.org/doc/essays/trust-mvp-unreachable-page/",
                            "quote": "missing",
                        },
                    ],
                }
            ],
        )
    ]
    validator = CitationValidator(fetcher=prefetched_fetcher(), concurrency=2)

    validations = await validator.validate(responses)

    assert unreachable_citation_count(validations) == 1
    assert average_citation_support_score(validations) > 0.0
    assert citation_precision(validations)[0] == 0.5


def test_citation_precision_returns_zero_for_no_attempts() -> None:
    """Verify empty citation sets produce a deterministic zero metric."""

    assert citation_precision([]) == (0.0, 0, 0)
