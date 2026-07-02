import asyncio
import logging

from config import load_settings
from internal.framework.logger import build_logger
from internal.framework.runner import map_bounded


def test_load_settings_reads_environment(monkeypatch) -> None:
    """Verify runtime settings are sourced from environment variables."""

    monkeypatch.setenv("SERVICE_NAME", "trust-test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CITATION_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("MAX_CITATION_CONCURRENCY", "4")
    monkeypatch.setenv("CONSENSUS_SIMILARITY_THRESHOLD", "0.6")
    monkeypatch.setenv("CONSENSUS_OPPOSITION_SIMILARITY_THRESHOLD", "0.25")
    monkeypatch.setenv("CROSS_LLM_JUDGE", "true")
    monkeypatch.setenv("CROSS_LLM_JUDGE_MODEL_NAME", "LLM judge model")
    monkeypatch.setenv("CROSS_LLM_JUDGE_BORDERLINE_LOW", "0.75")
    monkeypatch.setenv("CROSS_LLM_JUDGE_BORDERLINE_HIGH", "0.85")
    monkeypatch.setenv("CONSENSUS_STOPWORDS", "a,the,with")
    monkeypatch.setenv(
        "CONSENSUS_OPPOSING_TERMS",
        "suitable|not suitable,recommended|not recommended",
    )

    settings = load_settings()

    assert settings.service_name == "trust-test"
    assert settings.log_level == "DEBUG"
    assert settings.citation_timeout_seconds == 3.5
    assert settings.max_citation_concurrency == 4
    assert settings.consensus_similarity_threshold == 0.6
    assert settings.consensus_opposition_similarity_threshold == 0.25
    assert settings.cross_llm_judge_enabled is True
    assert settings.cross_llm_judge_model_name == "LLM judge model"
    assert settings.cross_llm_judge_borderline_low == 0.75
    assert settings.cross_llm_judge_borderline_high == 0.85
    assert settings.consensus_stopwords == {"a", "the", "with"}
    assert settings.consensus_opposing_terms == [
        ("suitable", "not suitable"),
        ("recommended", "not recommended"),
    ]


def test_build_logger_reuses_handlers() -> None:
    """Verify repeated logger construction does not duplicate handlers."""

    logger = build_logger("trust-test-logger", "INFO")
    handler_count = len(logger.handlers)

    same_logger = build_logger("trust-test-logger", "DEBUG")

    assert same_logger is logger
    assert len(same_logger.handlers) == handler_count
    assert same_logger.level == logging.DEBUG


def test_map_bounded_preserves_order() -> None:
    """Verify bounded async execution returns results in input order."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    result = asyncio.run(map_bounded([3, 1, 2], worker, concurrency=2))

    assert result == [6, 2, 4]
