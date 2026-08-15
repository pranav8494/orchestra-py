"""Tests for the provider port and the Anthropic adapter (CONVENTIONS.md §6, §12).

No request leaves the process: the SDK client is replaced with a stub, so what is under
test is the translation and the error mapping — the two things the adapter exists for.
"""

from typing import Any

import anthropic
import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from orchestra.core.errors import ExitCode, ProviderError
from orchestra.providers.anthropic import AnthropicProvider
from orchestra.providers.base import MessageRole, Provider, ProviderMessage, create_provider


class Answer(BaseModel):
    text: str


class StubMessages:
    """Stands in for `client.messages`, recording the kwargs it was called with."""

    def __init__(self, result: object) -> None:
        self._result = result
        self.kwargs: dict[str, Any] = {}

    async def parse(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class StubParsed:
    """The one attribute of `ParsedMessage` the adapter reads."""

    def __init__(self, parsed_output: Answer | None) -> None:
        self.parsed_output = parsed_output


def _provider(
    result: object, monkeypatch: pytest.MonkeyPatch
) -> tuple[AnthropicProvider, StubMessages]:
    """Build a provider whose SDK client answers with `result`.

    Replaces the private `_client` deliberately: taking the client as a constructor
    argument would put an SDK type in the signature, and the point of the adapter is
    that nothing outside it knows that type exists.
    """
    provider = AnthropicProvider(api_key=SecretStr("test-key"), model="claude-opus-5")
    stub = StubMessages(result)
    monkeypatch.setattr(provider, "_client", type("StubClient", (), {"messages": stub})())
    return provider, stub


def _schema_failure() -> ValidationError:
    """A real pydantic failure, as the SDK raises when the reply is the wrong shape."""
    try:
        Answer.model_validate({"wrong": 1})
    except ValidationError as exc:
        return exc
    raise AssertionError("Answer.model_validate({'wrong': 1}) should not validate")


@pytest.mark.asyncio
async def test_parse_structured_translates_our_messages_and_returns_parsed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, stub = _provider(StubParsed(Answer(text="ok")), monkeypatch)

    result = await provider.parse_structured(
        system="be brief",
        messages=[ProviderMessage(role=MessageRole.USER, content="hello")],
        output_format=Answer,
    )

    assert result == Answer(text="ok")
    assert stub.kwargs["model"] == "claude-opus-5"
    assert stub.kwargs["system"] == "be brief"
    assert stub.kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert stub.kwargs["output_format"] is Answer
    # Thinking is on by default and shares the budget with the output, so a plan-sized
    # answer needs room for both.
    assert stub.kwargs["max_tokens"] >= 8_000
    # The default model answers all four of these with a 400.
    assert not {"temperature", "top_p", "top_k", "thinking"} & stub.kwargs.keys()


@pytest.mark.asyncio
async def test_parse_structured_returns_none_when_nothing_was_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal or a truncated reply — the caller's cue to reformat, not an error."""
    provider, _ = _provider(StubParsed(None), monkeypatch)

    result = await provider.parse_structured(system="s", messages=[], output_format=Answer)

    assert result is None


@pytest.mark.asyncio
async def test_parse_structured_returns_none_when_the_reply_fails_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK validates the text block itself, so a wrong-shaped reply raises there.
    To the caller that is the same condition as an unparsed reply."""
    provider, _ = _provider(_schema_failure(), monkeypatch)

    result = await provider.parse_structured(system="s", messages=[], output_format=Answer)

    assert result is None


@pytest.mark.asyncio
async def test_parse_structured_maps_an_sdk_failure_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SDK exception type may escape `providers/` (§6, §8)."""
    failure = anthropic.AnthropicError("connection reset")
    provider, _ = _provider(failure, monkeypatch)

    with pytest.raises(ProviderError) as exc_info:
        await provider.parse_structured(system="s", messages=[], output_format=Answer)

    assert exc_info.value.exit_code == ExitCode.PROVIDER
    assert "claude-opus-5" in str(exc_info.value)
    assert exc_info.value.__cause__ is failure


def test_every_sdk_error_the_adapter_can_meet_derives_from_the_class_it_catches() -> None:
    """One `except` covers the SDK's failures only while this holds; if a release breaks
    it, an auth error would surface as an exit-1 bug instead of exit 4."""
    for error in (
        anthropic.APIError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.AuthenticationError,
        anthropic.RateLimitError,
    ):
        assert issubclass(error, anthropic.AnthropicError)


@pytest.mark.asyncio
async def test_aclose_closes_the_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unclosed, the SDK's pooled sockets outlive the process that made them (#23)."""
    provider = AnthropicProvider(api_key=SecretStr("test-key"), model="claude-opus-5")
    closed = False

    async def _close() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(provider._client, "close", _close)

    await provider.aclose()

    assert closed


@pytest.mark.asyncio
async def test_aclose_propagates_an_sdk_failure_as_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing is I/O too, and no SDK exception type may escape `providers/` (§6, §8)."""
    provider = AnthropicProvider(api_key=SecretStr("test-key"), model="claude-opus-5")

    async def _close() -> None:
        raise anthropic.AnthropicError("connection already gone")

    monkeypatch.setattr(provider._client, "close", _close)

    with pytest.raises(ProviderError) as exc_info:
        await provider.aclose()

    assert exc_info.value.exit_code == ExitCode.PROVIDER


def test_create_provider_returns_an_anthropic_provider() -> None:
    provider: Provider = create_provider(api_key=SecretStr("test-key"), model="claude-opus-5")

    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-opus-5"
