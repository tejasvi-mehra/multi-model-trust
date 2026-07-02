import httpx
import pytest

import main as app_module
from internal.service.schemas import ModelResponse, TrustRequest
from internal.service.validator import CitationValidator
from tests.fixtures.prefetched_fetcher import prefetched_fetcher


def _request() -> TrustRequest:
    """Build a valid request for API endpoint tests."""

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
async def test_api_happy_path_with_prefetched_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the FastAPI endpoint using prefetched citation pages."""

    monkeypatch.setattr(
        app_module,
        "validator",
        CitationValidator(fetcher=prefetched_fetcher(), concurrency=2),
    )
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/trust-orchestrate",
            json=_request().model_dump(mode="json"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["citation_precision"] == 1.0
    assert body["metrics"]["disagreement_groups"] == 1
    assert "average_citation_support_score" in body["metrics"]
    assert "unreachable_citations" in body["metrics"]


@pytest.mark.asyncio
async def test_api_rejects_fewer_than_two_models() -> None:
    """Verify schema validation enforces multi-model input."""

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/trust-orchestrate",
            json={
                "prompt": "single model?",
                "responses": [
                    {
                        "model": "mock-alpha",
                        "claims": [{"statement": "One claim.", "citations": []}],
                    }
                ],
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_marks_missing_citations_as_invalid_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify claims without citations do not produce an agree label."""

    monkeypatch.setattr(
        app_module,
        "validator",
        CitationValidator(fetcher=prefetched_fetcher(), concurrency=2),
    )
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/trust-orchestrate",
            json={
                "prompt": "Do docs exist?",
                "responses": [
                    {
                        "model": "mock-alpha",
                        "claims": [
                            {
                                "statement": "FastAPI supports automatic API docs.",
                                "citations": [],
                            }
                        ],
                    },
                    {
                        "model": "mock-beta",
                        "claims": [
                            {
                                "statement": (
                                    "FastAPI supports automatic API documentation."
                                ),
                                "citations": [],
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["consensus"][0]["label"] == "INVALID"
    assert body["validations"][0]["citation_status"] == "MISSING"


@pytest.mark.asyncio
async def test_api_prioritizes_valid_citation_over_invalid_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify invalid model claims are excluded when a valid source exists."""

    monkeypatch.setattr(
        app_module,
        "validator",
        CitationValidator(fetcher=prefetched_fetcher(), concurrency=2),
    )
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/trust-orchestrate",
            json={
                "prompt": "Do docs exist?",
                "responses": [
                    {
                        "model": "mock-alpha",
                        "claims": [
                            {
                                "statement": "FastAPI supports automatic API docs.",
                                "citations": [
                                    {
                                        "url": "https://fastapi.tiangolo.com/features/",
                                        "quote": "not actually on the page",
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "model": "mock-beta",
                        "claims": [
                            {
                                "statement": (
                                    "FastAPI supports automatic API documentation."
                                ),
                                "citations": [
                                    {
                                        "url": "https://fastapi.tiangolo.com/features/",
                                        "quote": "Automatic docs",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["consensus"][0]["label"] == "PARTIAL"
    assert body["consensus"][0]["members"][0]["model"] == "mock-beta"
    assert body["consensus"][0]["invalidated_members"][0]["model"] == "mock-alpha"


@pytest.mark.asyncio
async def test_api_returns_cross_llm_judge_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify API response includes judge prompt when env enables borderline judge."""

    monkeypatch.setenv("CROSS_LLM_JUDGE", "true")
    monkeypatch.setenv("CONSENSUS_SIMILARITY_THRESHOLD", "0.9")
    monkeypatch.setenv("CROSS_LLM_JUDGE_MODEL_NAME", "LLM judge model")
    monkeypatch.setattr(
        app_module,
        "validator",
        CitationValidator(fetcher=prefetched_fetcher(), concurrency=2),
    )
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/trust-orchestrate",
            json={
                "prompt": "Do FastAPI docs help backend teams?",
                "responses": [
                    {
                        "model": "mock-alpha",
                        "claims": [
                            {
                                "statement": (
                                    "FastAPI supports automatic API documentation "
                                    "for backend teams building HTTP services."
                                ),
                                "citations": [
                                    {
                                        "url": "https://fastapi.tiangolo.com/features/",
                                        "quote": "Automatic docs",
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "model": "mock-beta",
                        "claims": [
                            {
                                "statement": (
                                    "FastAPI supports automatic API documentation "
                                    "for backend teams building REST services."
                                ),
                                "citations": [
                                    {
                                        "url": "https://fastapi.tiangolo.com/features/",
                                        "quote": "Automatic docs",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["consensus"][0]["cross_llm_judge"] is not None
    assert (
        "Respond in exactly one word"
        in body["consensus"][0]["cross_llm_judge"]["prompt"]
    )
