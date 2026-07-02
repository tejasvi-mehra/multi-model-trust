# Sample citation sources

Sample payloads cite public documentation URLs. When running the service locally, those pages are fetched live with a configurable `User-Agent`.

## Live URLs used in samples

| URL | Typical quotes |
| --- | --- |
| `https://fastapi.tiangolo.com/` | `Interactive API docs`, `automatic interactive documentation` |
| `https://fastapi.tiangolo.com/features/` | `Automatic docs`, `based on open standards` |
| `https://www.python.org/doc/essays/blurb/` | Python executive summary paragraphs |
| `https://docs.python.org/3/library/asyncio.html` | asyncio overview sentences |

`partial_invalid_citations_request.json` also references `https://www.python.org/doc/essays/trust-mvp-unreachable-page/` to produce an unreachable citation.

## Offline tests

The same URLs are mapped to HTML snapshots under `tests/fixtures/pages/`. Evaluation and unit tests load those files through `tests/fixtures/prefetched_fetcher.py` instead of calling the network.

## Sample scenarios

| File | Scenario |
| --- | --- |
| `agreement_request.json` | Four models, clear agreement |
| `disagreement_request.json` | Four models, clear disagreement |
| `mixed_opinion_request.json` | Two models recommend FastAPI, two oppose |
| `llm_judge_request.json` | Borderline valid quotes with judge enabled |
| `all_invalid_citations_request.json` | All quotes invalid |
| `partial_invalid_citations_request.json` | Valid models prioritized over bad evidence |
| `large_mixed_request.json` | Multiple claims with mixed outcomes |

Expected labels and metrics are listed in `expected_response_summary.json`.
