"""Tests for the provider port and the Anthropic adapter (§6).

No request leaves the process: the SDK client is stubbed, so what is under test is the
translation and the error mapping — the two things the adapter exists for.
"""

from typing import Any

import anthropic
import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from orchestra.core.errors import ExitCode, ProviderError
from orchestra.providers.anthropic import AnthropicProvider
from orchestra.providers.base import (
    MessageRole,
    Provider,
    ProviderMessage,
    ToolResult,
    create_provider,
)
from orchestra.tools.base import ToolCall, ToolSpec


class Answer(BaseModel):
    text: str


class StubMessages:
    """Stands in for `client.messages`, recording the kwargs it was called with.

    One stub for both entry points: `parse_structured` and `send` are two calls on the same
    SDK object, so a second stub would be a second mocking style to keep in step.
    """

    def __init__(self, result: object) -> None:
        self._result = result
        self.kwargs: dict[str, Any] = {}

    async def parse(self, **kwargs: Any) -> object:
        return self._answer(kwargs)

    async def create(self, **kwargs: Any) -> object:
        return self._answer(kwargs)

    def _answer(self, kwargs: dict[str, Any]) -> object:
        self.kwargs = kwargs
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class StubParsed:
    """The one attribute of `ParsedMessage` the adapter reads."""

    def __init__(self, parsed_output: Answer | None) -> None:
        self.parsed_output = parsed_output


class StubBlock:
    """One content block of a reply: whatever attributes its `type` implies."""

    def __init__(self, **attributes: Any) -> None:
        self.__dict__.update(attributes)


class StubUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class StubMessage:
    """The attributes of `Message` the adapter reads."""

    def __init__(
        self,
        content: list[StubBlock],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop_reason: str | None = "end_turn",
    ) -> None:
        self.content = content
        self.usage = StubUsage(input_tokens, output_tokens)
        self.stop_reason = stop_reason


def _provider(
    result: object, monkeypatch: pytest.MonkeyPatch
) -> tuple[AnthropicProvider, StubMessages]:
    """Build a provider whose SDK client answers with `result`.

    Replaces the private `_client` deliberately: a constructor argument would put an SDK
    type in the signature, and the adapter exists so nothing outside knows that type.
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
    # Thinking is on by default and shares the budget with the output.
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
    """The SDK validates the text block itself; to the caller a raise there is the same
    condition as an unparsed reply."""
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


@pytest.mark.asyncio
async def test_send_decodes_a_plain_reply_with_no_tool_calls_and_summed_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tool calls is how the agent loop learns the model is finished."""
    reply = StubMessage(
        [StubBlock(type="text", text="the answer")], input_tokens=120, output_tokens=30
    )
    provider, stub = _provider(reply, monkeypatch)

    turn = await provider.send(
        system="be brief",
        messages=[ProviderMessage(role=MessageRole.USER, content="hello")],
    )

    assert turn.text == "the answer"
    assert turn.tool_calls == ()
    # Both halves are paid for out of the loop's one budget (§10).
    assert turn.usage_tokens == 150
    assert stub.kwargs["model"] == "claude-opus-5"
    assert stub.kwargs["system"] == "be brief"
    assert stub.kwargs["messages"] == [{"role": "user", "content": "hello"}]
    # Thinking is on by default and shares the budget with the output.
    assert stub.kwargs["max_tokens"] >= 8_000
    assert not {"temperature", "top_p", "top_k", "thinking"} & stub.kwargs.keys()


@pytest.mark.asyncio
async def test_send_omits_the_tools_argument_when_none_are_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API rejects `tools=[]`, and a final tool-free turn takes this path."""
    provider, stub = _provider(StubMessage([]), monkeypatch)

    await provider.send(system="s", messages=[], tools=())

    assert "tools" not in stub.kwargs


@pytest.mark.asyncio
async def test_send_translates_tool_specs_into_the_sdk_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, stub = _provider(StubMessage([]), monkeypatch)
    spec = ToolSpec(
        name="query_csv",
        description="Run one SQL-ish query over the sales table.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    await provider.send(system="s", messages=[], tools=[spec])

    assert stub.kwargs["tools"] == [
        {
            "name": "query_csv",
            "description": "Run one SQL-ish query over the sales table.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]


@pytest.mark.asyncio
async def test_send_decodes_tool_use_blocks_into_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ids must survive: they are what pairs an answer with the call that asked for it."""
    reply = StubMessage(
        [
            StubBlock(type="thinking", thinking="ignored"),
            StubBlock(type="text", text="looking that up"),
            StubBlock(
                type="tool_use", id="toolu_01", name="query_csv", input={"query": "SELECT 1"}
            ),
            StubBlock(type="tool_use", id="toolu_02", name="search", input={"q": "margin"}),
        ],
        input_tokens=1,
        output_tokens=2,
    )
    provider, _ = _provider(reply, monkeypatch)

    turn = await provider.send(system="s", messages=[])

    assert turn.text == "looking that up"
    assert turn.tool_calls == (
        ToolCall(id="toolu_01", name="query_csv", arguments={"query": "SELECT 1"}),
        ToolCall(id="toolu_02", name="search", arguments={"q": "margin"}),
    )
    assert turn.usage_tokens == 3


@pytest.mark.asyncio
async def test_send_replays_an_assistant_turn_as_text_and_tool_use_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that cannot see the call it made cannot read the answer to it."""
    provider, stub = _provider(StubMessage([]), monkeypatch)

    await provider.send(
        system="s",
        messages=[
            ProviderMessage(role=MessageRole.USER, content="what were sales?"),
            ProviderMessage(
                role=MessageRole.ASSISTANT,
                content="looking that up",
                tool_calls=(
                    ToolCall(id="toolu_01", name="query_csv", arguments={"query": "SELECT 1"}),
                ),
            ),
        ],
    )

    assert stub.kwargs["messages"] == [
        {"role": "user", "content": "what were sales?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "looking that up"},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "query_csv",
                    "input": {"query": "SELECT 1"},
                },
            ],
        },
    ]


@pytest.mark.asyncio
async def test_send_omits_the_text_block_when_the_assistant_narrated_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API rejects an empty text block, and a tool call without narration is common."""
    provider, stub = _provider(StubMessage([]), monkeypatch)

    await provider.send(
        system="s",
        messages=[
            ProviderMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(ToolCall(id="toolu_01", name="search", arguments={}),),
            )
        ],
    )

    assert stub.kwargs["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_01", "name": "search", "input": {}}],
        }
    ]


@pytest.mark.asyncio
async def test_send_sends_tool_results_as_tool_result_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`is_error` rides along so the model reads a failure and retries (§6)."""
    provider, stub = _provider(StubMessage([]), monkeypatch)

    await provider.send(
        system="s",
        messages=[
            ProviderMessage(
                role=MessageRole.USER,
                tool_results=(
                    ToolResult(call_id="toolu_01", content="42"),
                    ToolResult(call_id="toolu_02", content="no column `margin`", is_error=True),
                ),
            )
        ],
    )

    assert stub.kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": "42",
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_02",
                    "content": "no column `margin`",
                    "is_error": True,
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_send_maps_an_sdk_failure_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SDK exception type may escape `providers/` (§6, §8)."""
    failure = anthropic.AnthropicError("rate limited")
    provider, _ = _provider(failure, monkeypatch)

    with pytest.raises(ProviderError) as exc_info:
        await provider.send(system="s", messages=[])

    assert exc_info.value.exit_code == ExitCode.PROVIDER
    assert "claude-opus-5" in str(exc_info.value)
    assert exc_info.value.__cause__ is failure


def test_every_sdk_error_the_adapter_can_meet_derives_from_the_class_it_catches() -> None:
    """One `except` covers the SDK's failures only while this holds; if a release breaks it,
    an auth error surfaces as an exit-1 bug instead of exit 4."""
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
    provider: Provider = create_provider(
        api_key=SecretStr("test-key"), model="claude-opus-5", max_tokens=1234
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-opus-5"
    # The factory's one job beyond choosing the vendor: the caller's cap must reach the
    # client, since only a live request would otherwise reveal that it did not.
    assert provider._max_tokens == 1234


# --------------------------------------------------------------------------
# Replaying a turn whole. Regressions from the review of PR #27.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_keeps_the_whole_reply_for_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decoded fields summarise the turn but do not replace it: the API requires the
    model's reasoning back alongside the call, so undecoded blocks have to be kept."""
    blocks = [
        StubBlock(type="thinking"),
        StubBlock(type="tool_use", id="toolu_01", name="q", input={}),
    ]
    provider, _ = _provider(StubMessage(blocks), monkeypatch)

    turn = await provider.send(system="s", messages=[])

    assert turn.raw_content is blocks


@pytest.mark.asyncio
async def test_send_replays_raw_content_instead_of_rebuilding_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstructing from fields is what drops the undecoded blocks and gets the next
    request rejected."""
    blocks = [{"type": "thinking", "thinking": "", "signature": "abc"}]
    provider, stub = _provider(StubMessage([]), monkeypatch)

    await provider.send(
        system="s",
        messages=[
            ProviderMessage(
                role=MessageRole.ASSISTANT,
                content="this text must not be what is sent",
                tool_calls=(ToolCall(id="toolu_01", name="query_csv", arguments={}),),
                raw_content=blocks,
            )
        ],
    )

    assert stub.kwargs["messages"] == [{"role": "assistant", "content": blocks}]


@pytest.mark.asyncio
async def test_send_reports_why_the_model_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated reply looks exactly like a finished one without this."""
    provider, _ = _provider(
        StubMessage([StubBlock(type="text", text="cut off mid-")], stop_reason="max_tokens"),
        monkeypatch,
    )

    assert (await provider.send(system="s", messages=[])).stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_send_joins_multiple_text_blocks_on_a_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare concatenation runs the last word of one block into the first of the next."""
    blocks = [
        StubBlock(type="text", text="Let me check"),
        StubBlock(type="text", text="the numbers"),
    ]
    provider, _ = _provider(StubMessage(blocks), monkeypatch)

    assert (await provider.send(system="s", messages=[])).text == "Let me check\nthe numbers"
