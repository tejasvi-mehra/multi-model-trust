"""FastAPI entrypoint for multi-model trust orchestration."""

from fastapi import FastAPI, HTTPException

from config import build_consensus_engine, load_settings
from internal.framework.logger import build_logger, log_json
from internal.service.router import MockResponseRouter, ResponseRouter
from internal.service.schemas import Metrics, TrustRequest, TrustResponse
from internal.service.validator import (
    CitationValidator,
    HttpxUrlFetcher,
    average_citation_support_score,
    citation_precision,
    unreachable_citation_count,
)

# Load defaults once for process-level objects; request handler refreshes settings.
settings = load_settings()
logger = build_logger(settings.service_name, settings.log_level)
router: ResponseRouter = MockResponseRouter()
validator = CitationValidator(
    fetcher=HttpxUrlFetcher(
        timeout_seconds=settings.citation_timeout_seconds,
        user_agent=settings.citation_user_agent,
    ),
    concurrency=settings.max_citation_concurrency,
)

app = FastAPI(
    title="Multi-Model Trust",
    version="0.1.0",
    description="Routes mock model responses, validates citations, and detects agreement.",
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return a lightweight health check for local runs and containers."""

    return {"status": "ok", "service": settings.service_name}


@app.post("/v1/trust-orchestrate", response_model=TrustResponse)
async def trust_orchestrate(request: TrustRequest) -> TrustResponse:
    """Validate model claims, compare them, and return a trust report."""

    runtime_settings = load_settings()
    consensus_engine = build_consensus_engine(runtime_settings)

    if len(request.responses) < runtime_settings.min_model_responses:
        raise HTTPException(
            status_code=422,
            detail=(
                f"at least {runtime_settings.min_model_responses} "
                "model responses are required"
            ),
        )

    log_json(
        logger,
        "orchestration.start",
        {
            "prompt": request.prompt,
            "response_models": [response.model for response in request.responses],
            "consensus_config": {
                "similarity_threshold": runtime_settings.consensus_similarity_threshold,
                "cross_llm_judge_enabled": runtime_settings.cross_llm_judge_enabled,
            },
        },
    )

    responses = router.route(request)
    validations = await validator.validate(responses)
    consensus = consensus_engine.build(
        responses,
        validations,
        prompt=request.prompt,
        run_logger=logger,
    )
    precision, valid_citations, attempted_citations = citation_precision(validations)

    response = TrustResponse(
        prompt=request.prompt,
        validations=validations,
        consensus=consensus,
        metrics=Metrics(
            citation_precision=precision,
            validated_citations=valid_citations,
            attempted_citations=attempted_citations,
            unreachable_citations=unreachable_citation_count(validations),
            average_citation_support_score=average_citation_support_score(
                validations
            ),
            agreement_groups=sum(1 for group in consensus if group.label == "AGREE"),
            disagreement_groups=sum(
                1 for group in consensus if group.label == "DISAGREE"
            ),
            partial_groups=sum(1 for group in consensus if group.label == "PARTIAL"),
            invalid_response_groups=sum(
                1 for group in consensus if group.label == "INVALID"
            ),
        ),
    )

    log_json(
        logger,
        "orchestration.complete",
        {
            "prompt": request.prompt,
            "metrics": response.metrics.model_dump(),
            "consensus_labels": [group.label for group in consensus],
        },
    )
    return response
