import pytest
from pydantic import ValidationError

from internal.service.router import MockResponseRouter
from internal.service.schemas import TrustRequest


def _request() -> TrustRequest:
    """Build a valid two-model request for router and schema tests."""

    return TrustRequest(
        prompt="Compare claims.",
        responses=[
            {"model": "alpha", "claims": [{"statement": "Claim one."}]},
            {"model": "beta", "claims": [{"statement": "Claim two."}]},
        ],
    )


def test_mock_router_passes_through_validated_responses() -> None:
    """Verify the mock router does not mutate caller-supplied responses."""

    request = _request()

    routed = MockResponseRouter().route(request)

    assert routed == request.responses


def test_schema_rejects_duplicate_model_names() -> None:
    """Verify responses are compared across distinct model names."""

    with pytest.raises(ValidationError, match="distinct model names"):
        TrustRequest(
            prompt="Compare claims.",
            responses=[
                {"model": "alpha", "claims": [{"statement": "Claim one."}]},
                {"model": "alpha", "claims": [{"statement": "Claim two."}]},
            ],
        )
