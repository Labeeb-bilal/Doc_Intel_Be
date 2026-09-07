"""Stage 1 checkpoint: GroqClient retries on 429/5xx with visible backoff,
does not retry on non-retryable errors, and surfaces one clean exception
type (LLMUnavailableError) regardless of underlying cause."""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from app.adapters.llm import GroqClient, LLMUnavailableError


def _make_client(**overrides) -> GroqClient:
    defaults = dict(api_key="fake", model="openai/gpt-oss-20b", max_concurrency=4, rpm=6000, timeout_seconds=5)
    defaults.update(overrides)
    return GroqClient(**defaults)


def _fake_response(status_code: int = 200, *, content: str = "hello", prompt_tokens: int = 10, completion_tokens: int = 3) -> httpx.Response:
    """A real httpx.Response, not a mock — exercises the actual
    .status_code / .text / .json() call sites GroqClient relies on."""
    return httpx.Response(
        status_code=status_code,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


def _error_response(status_code: int, message: str = "error") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json={"error": {"message": message}},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


async def test_complete_returns_text_on_success():
    client = _make_client()
    client._http.post = AsyncMock(return_value=_fake_response(content="hi there"))

    result = await client.complete(system="be terse", user="say hi")

    assert result == "hi there"
    client._http.post.assert_awaited_once()


async def test_retries_on_429_then_succeeds(capsys):
    client = _make_client()
    rate_limited = _error_response(429, "rate limited")
    client._http.post = AsyncMock(side_effect=[rate_limited, rate_limited, _fake_response(content="finally")])

    result = await client.complete(system="s", user="u")

    assert result == "finally"
    assert client._http.post.await_count == 3
    # structlog prints via PrintLoggerFactory (stdout), not stdlib logging,
    # so capsys — not caplog — is what sees it.
    out = capsys.readouterr().out
    assert out.count("llm_retry") == 2  # visible backoff before attempts 2 and 3
    assert "wait_s=" in out


async def test_retries_on_5xx():
    client = _make_client()
    server_error = _error_response(503, "unavailable")
    client._http.post = AsyncMock(side_effect=[server_error, _fake_response(content="ok")])

    result = await client.complete(system="s", user="u")

    assert result == "ok"
    assert client._http.post.await_count == 2


async def test_exhausted_retries_raise_llm_unavailable_error():
    client = _make_client()
    rate_limited = _error_response(429, "rate limited")
    client._http.post = AsyncMock(side_effect=[rate_limited, rate_limited, rate_limited])

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete(system="s", user="u")

    assert exc_info.value.rate_limited is True
    assert client._http.post.await_count == 3  # stop_after_attempt(3), no 4th try


async def test_non_retryable_client_error_fails_immediately():
    client = _make_client()
    auth_error = _error_response(401, "bad key")
    client._http.post = AsyncMock(side_effect=[auth_error, _fake_response(content="unreachable")])

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete(system="s", user="u")

    # a 401 is not retryable — must fail on the very first attempt
    assert client._http.post.await_count == 1
    assert exc_info.value.rate_limited is False


async def test_complete_structured_parses_schema():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: str

    client = _make_client()
    client._http.post = AsyncMock(return_value=_fake_response(content='{"value": "ok"}'))

    result = await client.complete_structured(system="s", user="u", schema=Answer)

    assert isinstance(result, Answer)
    assert result.value == "ok"


async def test_complete_structured_raises_llm_unavailable_on_invalid_json():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: str

    client = _make_client()
    client._http.post = AsyncMock(return_value=_fake_response(content="not valid json at all"))

    with pytest.raises(LLMUnavailableError):
        await client.complete_structured(system="s", user="u", schema=Answer)


async def test_no_reasoning_params_are_sent_on_every_call():
    """Reasoning tokens are never requested: reasoning_format=hidden drops
    them from the response entirely, reasoning_effort=low minimises the
    tokens spent generating them in the first place."""
    client = _make_client()
    client._http.post = AsyncMock(return_value=_fake_response())

    await client.complete(system="s", user="u")

    _, kwargs = client._http.post.call_args
    assert kwargs["json"]["reasoning_effort"] == "low"
    assert kwargs["json"]["reasoning_format"] == "hidden"
