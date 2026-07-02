"""Pass-through router for caller-supplied model responses."""

from typing import Protocol

from internal.service.schemas import ModelResponse, TrustRequest


class ResponseRouter(Protocol):
    """Interface for routing prompts to model responses."""

    def route(self, request: TrustRequest) -> list[ModelResponse]:
        """Return normalized model responses for downstream validation."""


class MockResponseRouter:
    """Router implementation for caller-supplied mock model responses."""

    def route(self, request: TrustRequest) -> list[ModelResponse]:
        """Pass through validated mock responses without live model calls."""

        return request.responses


# Example adapter sketch for a future OpenAI-backed router implementation.
#
# class OpenAIResponseRouter:
#     """Call OpenAI and normalize output to ModelResponse objects."""
#
#     def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
#         from openai import OpenAI
#
#         self.client = OpenAI(api_key=api_key)
#         self.model = model
#
#     def route(self, request: TrustRequest) -> list[ModelResponse]:
#         """Send the prompt to OpenAI and map structured claims into our schema."""
#
#         completion = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "Return JSON with claims and citations for the user prompt."
#                     ),
#                 },
#                 {"role": "user", "content": request.prompt},
#             ],
#             response_format={"type": "json_object"},
#         )
#         payload = json.loads(completion.choices[0].message.content or "{}")
#         return [
#             ModelResponse(
#                 model=self.model,
#                 claims=payload.get("claims", []),
#             )
#         ]
#
# In main.py, swap MockResponseRouter() for:
#   OpenAIResponseRouter(api_key=os.environ["OPENAI_API_KEY"])
