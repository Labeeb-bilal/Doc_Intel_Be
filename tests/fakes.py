"""Fake adapters shared across the test suite so it runs fully offline."""
from __future__ import annotations

from pydantic import BaseModel


class FakeLLMClient:
    """Implements the LLMClient protocol with canned, inspectable responses.

    Tests can either set `response` up front, or provide `responses` — a
    list consumed one call at a time (useful for a conversation with two
    turns needing two different canned answers).
    """

    def __init__(
        self,
        *,
        response: str = "This is a fake answer.",
        responses: list[str] | None = None,
        structured_response: BaseModel | None = None,
        structured_responses: list[BaseModel] | None = None,
    ):
        self._responses = list(responses) if responses is not None else None
        self._default_response = response
        self._structured_responses = list(structured_responses) if structured_responses is not None else None
        self._default_structured_response = structured_response
        self.calls: list[dict] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"kind": "complete", "system": system, "user": user})
        if self._responses:
            return self._responses.pop(0)
        return self._default_response

    async def complete_structured(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls.append({"kind": "structured", "system": system, "user": user, "schema": schema})
        if self._structured_responses:
            return self._structured_responses.pop(0)
        if self._default_structured_response is not None:
            return self._default_structured_response
        raise NotImplementedError("FakeLLMClient.complete_structured has no canned response configured")
