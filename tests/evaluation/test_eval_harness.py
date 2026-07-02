import json
from pathlib import Path

import pytest

from internal.service.consensus import ConsensusEngine
from internal.service.schemas import ModelResponse, TrustRequest
from internal.service.validator import (
    CitationValidator,
    average_citation_support_score,
    citation_precision,
    unreachable_citation_count,
)
from tests.fixtures.prefetched_fetcher import prefetched_fetcher

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "with",
}
OPPOSING_TERMS = [
    ("suitable", "not suitable"),
    ("recommended", "not recommended"),
]


def _engine(**overrides) -> ConsensusEngine:
    """Build the eval consensus engine with env-style configurable terms."""

    defaults = {
        "stopwords": STOPWORDS,
        "opposing_terms": OPPOSING_TERMS,
    }
    defaults.update(overrides)
    return ConsensusEngine(**defaults)


def _benchmark_request() -> TrustRequest:
    """Build a benchmark payload with one intentional disagreement."""

    return TrustRequest(
        prompt="Should teams use FastAPI for backend APIs?",
        responses=[
            ModelResponse(
                model="mock-alpha",
                claims=[
                    {
                        "statement": "FastAPI is suitable for backend APIs.",
                        "citations": [
                            {
                                "url": "https://fastapi.tiangolo.com/features/",
                                "quote": "Automatic docs",
                            }
                        ],
                    }
                ],
            ),
            ModelResponse(
                model="mock-beta",
                claims=[
                    {
                        "statement": "FastAPI is not suitable for backend APIs.",
                        "citations": [
                            {
                                "url": "https://fastapi.tiangolo.com/features/",
                                "quote": "Automatic docs",
                            }
                        ],
                    }
                ],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_citation_correctness_and_precision_eval() -> None:
    """Measure citation validation by mixing one valid and one invalid quote."""

    request = _benchmark_request()
    request.responses[1].claims[0].citations[0].quote = "missing quote"
    validator = CitationValidator(
        fetcher=prefetched_fetcher(),
        concurrency=2,
    )

    validations = await validator.validate(request.responses)
    precision, valid, attempted = citation_precision(validations)

    assert attempted == 2
    assert valid == 1
    assert precision == 0.5


def test_disagreement_detection_eval() -> None:
    """Measure whether opposing but similar claims are labeled as disagreement."""

    consensus = _engine().build(_benchmark_request().responses)

    assert any(group.label == "DISAGREE" for group in consensus)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample_name",
    [
        "agreement_request.json",
        "disagreement_request.json",
        "mixed_opinion_request.json",
        "llm_judge_request.json",
        "all_invalid_citations_request.json",
        "partial_invalid_citations_request.json",
        "large_mixed_request.json",
    ],
)
async def test_sample_payloads_match_expected_summary(sample_name: str) -> None:
    """Verify sample payloads against prefetched pages and expected summaries."""

    sample = json.loads((SAMPLES_DIR / sample_name).read_text())
    expected = json.loads((SAMPLES_DIR / "expected_response_summary.json").read_text())[
        sample_name
    ]
    request = TrustRequest.model_validate(sample)
    validator = CitationValidator(
        fetcher=prefetched_fetcher(),
        concurrency=4,
    )

    validations = await validator.validate(request.responses)
    engine_kwargs = expected.get("engine", {})
    consensus = _engine(**engine_kwargs).build(request.responses, validations)
    precision, _, _ = citation_precision(validations)
    support_score = average_citation_support_score(validations)
    unreachable = unreachable_citation_count(validations)

    assert [group.label for group in consensus] == expected["expected_consensus_labels"]
    assert precision == expected["expected_citation_precision_if_sources_are_reachable"]
    assert support_score == pytest.approx(
        expected["expected_average_citation_support_score"]
    )
    assert unreachable == expected["expected_unreachable_citations"]
    assert sum(1 for group in consensus if group.label == "AGREE") == (
        expected["expected_agreement_groups"]
    )
    assert sum(1 for group in consensus if group.label == "DISAGREE") == (
        expected["expected_disagreement_groups"]
    )
    assert sum(1 for group in consensus if group.label == "PARTIAL") == (
        expected["expected_partial_groups"]
    )
    assert sum(1 for group in consensus if group.label == "INVALID") == (
        expected["expected_invalid_response_groups"]
    )
    if expected_invalid_groups := expected.get("expected_invalid_response_groups"):
        assert sum(1 for group in consensus if group.label == "INVALID") == (
            expected_invalid_groups
        )
    if expected_invalidated := expected.get("expected_invalidated_members"):
        assert sum(len(group.invalidated_members) for group in consensus) == (
            expected_invalidated
        )
    if expected.get("expected_cross_llm_judge"):
        assert any(group.cross_llm_judge is not None for group in consensus)
        assert any(
            "Respond in exactly one word" in group.cross_llm_judge.prompt
            for group in consensus
            if group.cross_llm_judge is not None
        )
        assert all(
            member.citation_status == "VALID"
            for group in consensus
            if group.cross_llm_judge is not None
            for member in group.members
        )
    if expected_reason := expected.get("expected_invalid_citation_reason"):
        reasons = [
            citation.reason
            for validation in validations
            for citation in validation.citations
        ]
        assert expected_reason in reasons
    if expected_unreachable := expected.get("expected_unreachable_reason"):
        reasons = [
            citation.reason
            for validation in validations
            for citation in validation.citations
        ]
        assert expected_unreachable in reasons
    if in_favor := expected.get("expected_in_favor_models"):
        disagree = next(group for group in consensus if group.label == "DISAGREE")
        favored = [
            member.model
            for member in disagree.members
            if "not recommended" not in member.statement.lower()
        ]
        assert set(favored) == set(in_favor)
    if opposed := expected.get("expected_opposed_models"):
        disagree = next(group for group in consensus if group.label == "DISAGREE")
        opposed_models = [
            member.model
            for member in disagree.members
            if "not recommended" in member.statement.lower()
        ]
        assert set(opposed_models) == set(opposed)
    if prioritized := expected.get("expected_prioritized_models"):
        agree = next(group for group in consensus if group.label == "AGREE")
        assert {member.model for member in agree.members} == set(prioritized)
    if min_claims := expected.get("expected_minimum_claim_validations"):
        assert len(validations) >= min_claims
