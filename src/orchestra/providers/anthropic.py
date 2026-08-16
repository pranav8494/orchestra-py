"""The Anthropic adapter — the only module that imports `anthropic`.

Two things happen here and nowhere else: SDK failures become `ProviderError` (exit 4,
§8), and the SDK's message shape becomes ours. `import anthropic` below is absolute and
resolves to the vendor package, not to this module of the same name.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, assert_never, cast

import anthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from pydantic import SecretStr, ValidationError

from orchestra.core.errors import ProviderError
from orchestra.providers.base import AssistantTurn, MessageRole, ProviderMessage, StructuredT
from orchestra.tools.base import ToolCall, ToolSpec

# claude-opus-5 thinks by default and `max_tokens` caps thinking *and* output together,
# so a budget sized for the answer alone truncates the JSON and yields no parsed output.
DEFAULT_MAX_TOKENS = 16_000


class AnthropicProvider:
    """Talks to the Anthropic Messages API. Constructed by `create_provider` (§3.3).

    Sends no `temperature`, `top_p` or `top_k`: the default model 400s on all three. No
    `thinking` block either — this model thinks by default, and turning it off lets it
    write a tool call into visible text where the call silently never runs.
    """

    def __init__(
        self, *, api_key: SecretStr, model: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        """Create the client. `api_key` is unwrapped here and nowhere above (§9)."""
        self._client = anthropic.AsyncAnthropic(api_key=api_key.get_secret_value())
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        """The model identifier requests are sent to."""
        return self._model

    async def parse_structured(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        output_format: type[StructuredT],
    ) -> StructuredT | None:
        """Send one structured-output request. See `Provider.parse_structured`."""
        with _as_provider_error(f"Anthropic request to {self._model}"):
            try:
                response = await self._client.messages.parse(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=[_to_sdk_message(message) for message in messages],
                    output_format=output_format,
                )
            except ValidationError:
                # JSON, but not *this* schema. Same condition as a refusal for the caller —
                # no usable structured output — so it gets the same answer, not a second
                # error path meaning the same thing.
                return None
        return response.parsed_output

    async def send(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AssistantTurn:
        """Send one conversational turn. See `Provider.send`."""
        # `tools` omitted rather than passed empty: the API rejects `tools=[]`, which is
        # also the shape of a final, tool-free turn. One `max_tokens` for both calls — a
        # tool-use turn is shorter than a plan, so a second constant is one more to tune.
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = [_to_sdk_tool(tool) for tool in tools]

        with _as_provider_error(f"Anthropic request to {self._model}"):
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[_to_sdk_message(message) for message in messages],
                **extra,
            )
        return _from_sdk_response(response)

    async def aclose(self) -> None:
        """Close the SDK client's connection pool. See `Provider.aclose`.

        Closing is I/O, so it fails like any other call and is mapped here — no SDK
        exception type leaves this module (§6, §8).
        """
        with _as_provider_error("Closing the Anthropic client"):
            await self._client.close()


@contextmanager
def _as_provider_error(action: str) -> Iterator[None]:
    """Turn an SDK failure into the taxonomy's `ProviderError` (§8), as `<action> failed`.

    `anthropic.AnthropicError` is the SDK's base class — transport, status and retryable
    failures — so no bare `except` and no SDK type leaves this module (§6).
    """
    try:
        yield
    except anthropic.AnthropicError as exc:
        raise ProviderError(f"{action} failed: {exc}") from exc


def _to_sdk_message(message: ProviderMessage) -> MessageParam:
    """Translate one turn into the SDK's shape. Shared by both requests (§2.2), so a
    transcript reads the same whichever call resends it."""
    # `assert_never` rather than an else-branch: a new `MessageRole` member must fail
    # under mypy here, not be silently relabelled `user` at runtime.
    role: Literal["user", "assistant"]
    match message.role:
        case MessageRole.USER:
            role = "user"
        case MessageRole.ASSISTANT:
            role = "assistant"
        case unhandled:
            assert_never(unhandled)
    return MessageParam(role=role, content=_to_sdk_content(message))


def _to_sdk_content(message: ProviderMessage) -> str | list[Any]:
    """The body of one turn: a plain string, or the block list a tool turn needs."""
    # A replayed assistant turn goes back byte-for-byte: rebuilding it would drop the
    # blocks this adapter never decodes, including the thinking the API requires
    # alongside a tool call (see `AssistantTurn.raw_content`).
    if message.raw_content is not None:
        return cast("list[Any]", message.raw_content)

    if not message.tool_calls and not message.tool_results:
        return message.content

    blocks: list[Any] = []
    # Results first: the API requires a user turn's `tool_result` blocks to lead it.
    blocks += [
        ToolResultBlockParam(
            type="tool_result",
            tool_use_id=result.call_id,
            content=result.content,
            is_error=result.is_error,
        )
        for result in message.tool_results
    ]
    # Only when non-empty: the API rejects a text block carrying "", and a tool call
    # with no narration is the common case.
    if message.content:
        blocks.append(TextBlockParam(type="text", text=message.content))
    blocks += [
        ToolUseBlockParam(
            type="tool_use",
            id=call.id,
            name=call.name,
            # `dict(...)`: the SDK types this as `Dict`, our side is a `Mapping` (§7).
            input=dict(call.arguments),
        )
        for call in message.tool_calls
    ]
    return blocks


def _to_sdk_tool(tool: ToolSpec) -> ToolParam:
    """Translate what a tool publishes into what the API offers the model.

    Field-by-field, not `asdict`: the names line up today, and a rename on either side
    should break here rather than at the API.
    """
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=dict(tool.input_schema),
    )


def _from_sdk_response(response: Any) -> AssistantTurn:
    """Decode one reply into our types — the point past which the SDK does not exist (§6).

    Only `text` and `tool_use` are decoded; the rest of the turn is kept in
    `raw_content`, not dropped, so the next request can replay it whole. The API
    requires a turn's thinking blocks back with its tool calls.
    """
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    return AssistantTurn(
        # Newline-joined: a reply can arrive as several text blocks, and joining them
        # bare runs the last word of one into the first of the next.
        text="\n".join(texts),
        tool_calls=tuple(calls),
        # Summed here so every agent isn't re-deriving the loop's budget (§10).
        usage_tokens=response.usage.input_tokens + response.usage.output_tokens,
        stop_reason=response.stop_reason or "",
        raw_content=response.content,
    )
