"""Groq adapter (OpenAI-compatible chat completions REST API). Every HTTP
call to Groq lives here.

Four guards wrap every call so nothing else in the codebase has to think
about LLM failure modes: a concurrency cap, a requests-per-minute limiter,
retry-with-backoff on 429/5xx, and a hard per-call timeout. A caller only
ever sees two outcomes: a result, or LLMUnavailableError.

Uses httpx directly rather than an SDK — httpx is already a project
dependency (for tests), and Groq's API is a plain OpenAI-shaped REST
endpoint, so a real SDK would add a dependency for no real benefit.
"""
from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Protocol, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.config import get_settings

log = structlog.get_logger("llm")

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMClient(Protocol):
    async def complete(self, *, system: str, user: str) -> str: ...
    async def complete_structured(self, *, system: str, user: str, schema: type[SchemaT]) -> SchemaT: ...


class LLMUnavailableError(Exception):
    """The one exception this adapter raises outward — auth failure,
    timeout, retries exhausted on a transient error, or a structured
    response that didn't validate against the requested schema all land
    here so callers have a single thing to catch.

    `rate_limited` lets a caller distinguish "the provider is rate-
    limiting us" (-> 429 + Retry-After is more honest to the client) from
    every other failure (-> 503).
    """

    def __init__(self, message: str, *, rate_limited: bool = False):
        super().__init__(message)
        self.message = message
        self.rate_limited = rate_limited


class _RetryableHTTPError(Exception):
    """429 or 5xx — worth another attempt."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code


class _NonRetryableHTTPError(Exception):
    """Any other 4xx (bad request, bad API key, ...) — fails identically
    on retry, so don't waste attempts on it."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableHTTPError)


class _RequestsPerMinuteLimiter:
    """Spaces calls evenly across the minute rather than bursting then
    stalling — good enough for a single-process app; not a general-purpose
    distributed rate limiter."""

    def __init__(self, rate_per_minute: int) -> None:
        self._interval = 60.0 / rate_per_minute
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


class GroqClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_concurrency: int,
        rpm: int,
        timeout_seconds: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._limiter = _RequestsPerMinuteLimiter(rpm)
        self._timeout = timeout_seconds
        self._http = httpx.AsyncClient()

    async def complete(self, *, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **_NO_REASONING_PARAMS,
        }
        data = await self._call(payload)
        return data["choices"][0]["message"]["content"] or ""

    async def complete_structured(self, *, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **_NO_REASONING_PARAMS,
            # Native constrained decoding, not prompt-and-parse: Groq's
            # OpenAI-compatible json_schema mode guarantees syntactically
            # valid JSON matching this schema.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
        }
        data = await self._call(payload)
        content = data["choices"][0]["message"]["content"]
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise LLMUnavailableError(f"Groq returned output that didn't match the schema: {exc}") from exc

    async def _call(self, payload: dict) -> dict:
        attempts = {"count": 0}
        overall_start = time.perf_counter()

        def _before_sleep(retry_state) -> None:
            wait_s = round(retry_state.next_action.sleep, 2) if retry_state.next_action else None
            log.warning(
                "llm_retry",
                model=self._model,
                attempt=retry_state.attempt_number,
                wait_s=wait_s,
                error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
            )

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=20),
            before_sleep=_before_sleep,
            reraise=True,
        )
        async def _attempt() -> dict:
            attempts["count"] += 1
            async with self._semaphore:
                await self._limiter.acquire()
                response = await self._http.post(
                    _GROQ_CHAT_COMPLETIONS_URL,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self._timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableHTTPError(response.status_code, response.text)
                if response.status_code >= 400:
                    raise _NonRetryableHTTPError(response.status_code, response.text)
                return response.json()

        try:
            result = await _attempt()
        except Exception as exc:
            log.error(
                "llm_call_failed",
                model=self._model,
                attempts=attempts["count"],
                duration_ms=round((time.perf_counter() - overall_start) * 1000, 1),
                error=str(exc),
            )
            rate_limited = isinstance(exc, _RetryableHTTPError) and exc.status_code == 429
            raise LLMUnavailableError(
                f"Groq call failed after {attempts['count']} attempt(s): {exc}", rate_limited=rate_limited
            ) from exc

        usage = result.get("usage") or {}
        log.info(
            "llm_call",
            model=self._model,
            attempts=attempts["count"],
            duration_ms=round((time.perf_counter() - overall_start) * 1000, 1),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        return result


# No reasoning tokens, ever: explainability here comes from measured
# pipeline telemetry (scores, ranks, timings), not model narration.
# reasoning_format=hidden drops any reasoning text from the response
# entirely (not just from what we read); reasoning_effort=low additionally
# cuts the tokens actually spent generating it.
_NO_REASONING_PARAMS = {"reasoning_effort": "low", "reasoning_format": "hidden"}


@lru_cache
def get_llm_client() -> GroqClient:
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMUnavailableError("GROQ_API_KEY is not configured.")
    return GroqClient(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        max_concurrency=settings.llm_max_concurrency,
        rpm=settings.llm_rpm,
        timeout_seconds=settings.llm_timeout_seconds,
    )
