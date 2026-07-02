from internal.service.consensus import ConsensusEngine
from internal.service.schemas import ClaimValidation, ModelResponse

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
OPPOSING_TERMS = [("suitable", "not suitable")]


def _engine(**overrides) -> ConsensusEngine:
    """Build a consensus engine with env-style configurable terms."""

    defaults = {
        "stopwords": STOPWORDS,
        "opposing_terms": OPPOSING_TERMS,
    }
    defaults.update(overrides)
    return ConsensusEngine(**defaults)


def test_consensus_labels_agreement_for_similar_claims() -> None:
    """Verify similar non-opposing claims are grouped as agreement."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[{"statement": "FastAPI supports API docs."}],
        ),
        ModelResponse(
            model="beta",
            claims=[{"statement": "FastAPI supports API documentation."}],
        ),
    ]

    groups = _engine().build(responses)

    assert groups[0].label == "AGREE"


def test_consensus_labels_partial_for_single_model_claim() -> None:
    """Verify unshared claims remain partial instead of forced agreement."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[{"statement": "Redis can cache source fetches."}],
        ),
        ModelResponse(
            model="beta",
            claims=[{"statement": "Postgres stores run history."}],
        ),
    ]

    groups = _engine().build(responses)

    assert {group.label for group in groups} == {"PARTIAL"}


def test_consensus_labels_disagreement_for_negated_claims() -> None:
    """Verify explicit negation creates a disagreement group."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[
                {
                    "statement": (
                        "Python is suitable for backend MVP APIs because it is "
                        "high-level and dynamic."
                    )
                }
            ],
        ),
        ModelResponse(
            model="beta",
            claims=[
                {
                    "statement": (
                        "Python is not suitable for backend MVP APIs when raw "
                        "runtime speed is the top priority."
                    )
                }
            ],
        ),
    ]

    groups = _engine().build(responses)

    assert [group.label for group in groups] == ["DISAGREE"]


def test_consensus_prioritizes_valid_source_over_invalid_source() -> None:
    """Verify valid cited claims are used while invalid model claims are excluded."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[{"statement": "FastAPI supports automatic API docs."}],
        ),
        ModelResponse(
            model="beta",
            claims=[{"statement": "FastAPI supports automatic API documentation."}],
        ),
    ]
    validations = [
        ClaimValidation(
            model="alpha",
            statement="FastAPI supports automatic API docs.",
            citations=[],
            all_citations_valid=False,
            citation_status="MISSING",
            invalid_reason="missing citation",
        ),
        ClaimValidation(
            model="beta",
            statement="FastAPI supports automatic API documentation.",
            citations=[],
            all_citations_valid=True,
            citation_status="VALID",
        ),
    ]

    groups = _engine().build(responses, validations)

    assert groups[0].label == "PARTIAL"
    assert [member.model for member in groups[0].members] == ["beta"]
    assert [member.model for member in groups[0].invalidated_members] == ["alpha"]
    assert groups[0].invalidated_members[0].citation_errors == ["missing citation"]


def test_consensus_marks_group_invalid_when_all_sources_are_bad() -> None:
    """Verify a claim group is invalid only when no valid source remains."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[{"statement": "FastAPI supports automatic API docs."}],
        ),
        ModelResponse(
            model="beta",
            claims=[{"statement": "FastAPI supports automatic API documentation."}],
        ),
    ]
    validations = [
        ClaimValidation(
            model="alpha",
            statement="FastAPI supports automatic API docs.",
            citations=[],
            all_citations_valid=False,
            citation_status="MISSING",
            invalid_reason="missing citation",
        ),
        ClaimValidation(
            model="beta",
            statement="FastAPI supports automatic API documentation.",
            citations=[],
            all_citations_valid=False,
            citation_status="MISSING",
            invalid_reason="missing citation",
        ),
    ]

    groups = _engine().build(responses, validations)

    assert groups[0].label == "INVALID"
    assert groups[0].members == []
    assert [member.model for member in groups[0].invalidated_members] == [
        "alpha",
        "beta",
    ]


def test_consensus_emits_cross_llm_judge_prompt_for_borderline_similarity() -> None:
    """Verify borderline comparisons include the simulated judge prompt."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[
                {
                    "statement": (
                        "FastAPI supports automatic API documentation for backend teams."
                    )
                }
            ],
        ),
        ModelResponse(
            model="beta",
            claims=[
                {
                    "statement": (
                        "FastAPI supports automatic API documentation for backend services."
                    )
                }
            ],
        ),
    ]
    engine = _engine(
        similarity_threshold=0.9,
        cross_llm_judge_enabled=True,
        cross_llm_judge_model_name="LLM judge model",
        cross_llm_judge_borderline_low=0.75,
        cross_llm_judge_borderline_high=0.85,
    )

    groups = engine.build(responses)

    assert groups[0].label == "AGREE"
    assert groups[0].cross_llm_judge is not None
    assert groups[0].cross_llm_judge.model_name == "LLM judge model"
    assert "Respond in exactly one word" in groups[0].cross_llm_judge.prompt


def test_consensus_emits_judge_when_threshold_also_matches() -> None:
    """Verify judge prompt is omitted once threshold consensus is already reached."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[
                {
                    "statement": (
                        "FastAPI supports automatic API documentation for backend teams."
                    )
                }
            ],
        ),
        ModelResponse(
            model="beta",
            claims=[
                {
                    "statement": (
                        "FastAPI supports automatic API documentation for backend services."
                    )
                }
            ],
        ),
    ]
    engine = _engine(
        similarity_threshold=0.35,
        cross_llm_judge_enabled=True,
        cross_llm_judge_borderline_low=0.75,
        cross_llm_judge_borderline_high=0.85,
    )

    groups = engine.build(responses)

    assert groups[0].label == "AGREE"
    assert groups[0].cross_llm_judge is None


def test_consensus_skips_judge_for_invalid_citation_groups() -> None:
    """Verify cross-LLM judge is not attached to invalid-only groups."""

    responses = [
        ModelResponse(
            model="alpha",
            claims=[{"statement": "FastAPI supports automatic API docs."}],
        ),
        ModelResponse(
            model="beta",
            claims=[{"statement": "FastAPI supports automatic API documentation."}],
        ),
    ]
    validations = [
        ClaimValidation(
            model="alpha",
            statement="FastAPI supports automatic API docs.",
            citations=[],
            all_citations_valid=False,
            citation_status="INVALID",
            invalid_reason="one or more citations are invalid",
        ),
        ClaimValidation(
            model="beta",
            statement="FastAPI supports automatic API documentation.",
            citations=[],
            all_citations_valid=False,
            citation_status="INVALID",
            invalid_reason="one or more citations are invalid",
        ),
    ]
    engine = _engine(
        cross_llm_judge_enabled=True,
        cross_llm_judge_borderline_low=0.75,
        cross_llm_judge_borderline_high=0.85,
    )

    groups = engine.build(responses, validations)

    assert groups[0].label == "INVALID"
    assert groups[0].cross_llm_judge is None
