"""Environment-backed settings and consensus engine factory."""

import os
from dataclasses import dataclass

DEFAULT_CONSENSUS_STOPWORDS = "a,an,and,are,for,in,is,of,or,the,to,with"
DEFAULT_CONSENSUS_OPPOSING_TERMS = (
    "suitable|not suitable,"
    "recommended|not recommended,"
    "supported|unsupported,"
    "safe|unsafe,"
    "correct|incorrect,"
    "increase|decrease,"
    "higher|lower"
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by the API and service dependencies."""

    service_name: str
    log_level: str
    citation_timeout_seconds: float
    max_citation_concurrency: int
    min_model_responses: int
    citation_user_agent: str
    consensus_similarity_threshold: float
    consensus_opposition_similarity_threshold: float
    cross_llm_judge_enabled: bool
    cross_llm_judge_model_name: str
    cross_llm_judge_borderline_low: float
    cross_llm_judge_borderline_high: float
    consensus_stopwords: set[str]
    consensus_opposing_terms: list[tuple[str, str]]


def _csv_set(value: str) -> set[str]:
    """Parse comma-separated env values into a normalized set."""

    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _opposing_terms(value: str) -> list[tuple[str, str]]:
    """Parse comma-separated positive|negative opposition term pairs."""

    terms: list[tuple[str, str]] = []
    for raw_pair in value.split(","):
        if "|" not in raw_pair:
            continue
        positive, negative = raw_pair.split("|", 1)
        if positive.strip() and negative.strip():
            terms.append((positive.strip().lower(), negative.strip().lower()))
    return terms


def load_settings() -> Settings:
    """Build settings from environment variables for dependency injection."""

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    return Settings(
        service_name=os.getenv("SERVICE_NAME", "multi-model-trust"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        citation_timeout_seconds=float(os.getenv("CITATION_TIMEOUT_SECONDS", "5")),
        max_citation_concurrency=int(os.getenv("MAX_CITATION_CONCURRENCY", "8")),
        min_model_responses=int(os.getenv("MIN_MODEL_RESPONSES", "2")),
        citation_user_agent=os.getenv(
            "CITATION_USER_AGENT",
            "multi-model-trust/0.1.0 (+https://github.com/tejasvi-mehra/multi-model-trust)",
        ),
        consensus_similarity_threshold=float(
            os.getenv("CONSENSUS_SIMILARITY_THRESHOLD", "0.35")
        ),
        consensus_opposition_similarity_threshold=float(
            os.getenv("CONSENSUS_OPPOSITION_SIMILARITY_THRESHOLD", "0.35")
        ),
        cross_llm_judge_enabled=os.getenv("CROSS_LLM_JUDGE", "false").lower()
        in {"1", "true", "yes"},
        cross_llm_judge_model_name=os.getenv(
            "CROSS_LLM_JUDGE_MODEL_NAME", "LLM judge model"
        ),
        cross_llm_judge_borderline_low=float(
            os.getenv("CROSS_LLM_JUDGE_BORDERLINE_LOW", "0.75")
        ),
        cross_llm_judge_borderline_high=float(
            os.getenv("CROSS_LLM_JUDGE_BORDERLINE_HIGH", "0.85")
        ),
        consensus_stopwords=_csv_set(
            os.getenv("CONSENSUS_STOPWORDS", DEFAULT_CONSENSUS_STOPWORDS)
        ),
        consensus_opposing_terms=_opposing_terms(
            os.getenv("CONSENSUS_OPPOSING_TERMS", DEFAULT_CONSENSUS_OPPOSING_TERMS)
        ),
    )


def build_consensus_engine(settings: Settings):
    """Build a consensus engine from the latest env-backed settings."""

    # Import here to keep config usable without pulling service modules at import time.
    from internal.service.consensus import ConsensusEngine

    return ConsensusEngine(
        similarity_threshold=settings.consensus_similarity_threshold,
        opposition_similarity_threshold=settings.consensus_opposition_similarity_threshold,
        cross_llm_judge_enabled=settings.cross_llm_judge_enabled,
        cross_llm_judge_model_name=settings.cross_llm_judge_model_name,
        cross_llm_judge_borderline_low=settings.cross_llm_judge_borderline_low,
        cross_llm_judge_borderline_high=settings.cross_llm_judge_borderline_high,
        stopwords=settings.consensus_stopwords,
        opposing_terms=settings.consensus_opposing_terms,
    )
