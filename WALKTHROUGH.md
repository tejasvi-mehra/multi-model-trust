# Walkthrough

## What This Service Is

Multi-Model Trust compares answers from two or more models on the same prompt. The caller supplies model responses; the service validates citations and reports agreement, disagreement, partial overlap, or invalid evidence.

## Repository Layout

```text
config.py                         runtime settings + engine factory
main.py                           FastAPI entrypoint
internal/framework/logger.py      logging + JSON trace helper
internal/framework/runner.py      bounded async helper
internal/service/schemas.py       Pydantic API and domain schemas
internal/service/router.py        response router
internal/service/validator.py     citation fetching and quote checks
internal/service/consensus.py     similarity, judge prompt, final labels
tests/fixtures/pages/             prefetched HTML used by tests
```

## Install And Run

```bash
python -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env
make run
```

Open `http://localhost:8000/docs`.

## Try Sample Requests

Agreement:

```bash
curl -X POST http://localhost:8000/v1/trust-orchestrate \
  -H "Content-Type: application/json" \
  --data @samples/agreement_request.json
```

Disagreement:

```bash
curl -X POST http://localhost:8000/v1/trust-orchestrate \
  -H "Content-Type: application/json" \
  --data @samples/disagreement_request.json
```

Partial invalid citations (valid models prioritized):

```bash
curl -X POST http://localhost:8000/v1/trust-orchestrate \
  -H "Content-Type: application/json" \
  --data @samples/partial_invalid_citations_request.json
```

## Cross-LLM Judge Sample

Set in `.env`:

```text
CROSS_LLM_JUDGE=true
CONSENSUS_SIMILARITY_THRESHOLD=0.9
```

Restart the server, then:

```bash
curl -X POST http://localhost:8000/v1/trust-orchestrate \
  -H "Content-Type: application/json" \
  --data @samples/llm_judge_request.json
```

Inspect `consensus[0].cross_llm_judge.prompt` in the JSON response.

## Tests

```bash
make test
make test-unit
make test-eval
```

Evaluation tests read prefetched pages from `tests/fixtures/pages` via `PrefetchedPageFetcher`, so they validate the same URLs and quotes as the samples without live network calls.
