"""The Anthropic adapter — the only module in the codebase that imports `anthropic`.

Two things happen at this boundary and nowhere else: SDK failures become `ProviderError`
(exit 4, §8), and the SDK's message shape becomes ours. Note the module is
`orchestra.providers.anthropic`; `import anthropic` below is absolute and resolves to
the vendor package, not to this file.
"""

from collections.abc import Sequence
from typing import Any, Literal, assert_never

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

    Deliberately sends no `temperature`, `top_p`, `top_k` or `thinking` block: the
    default model rejects all four with a 400.
    """

    def __init__(
        self, *, api_key: SecretStr, model: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        """Create the client.

        Args:
            api_key: the credential, unwrapped here and nowhere above (§9).
            model: model identifier, from `Config.anthropic_model`.
            max_tokens: combined thinking + output budget for one request.
        """
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
        try:
            response = await self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[_to_sdk_message(message) for message in messages],
                output_format=output_format,
            )
        except anthropic.AnthropicError as exc:
            # The SDK's own base class, so transport, status and retryable failures are
            # all covered without a bare `except` (§8). Cancellation is a BaseException
            # and passes straight through (§10).
            raise ProviderError(f"Anthropic request to {self._model} failed: {exc}") from exc
        except ValidationError:
            # The SDK validates the text block against `output_format`, so a reply that
            # is JSON but not *this* schema lands here. For the caller that is the same
            # condition as a refusal — no usable structured output — so it gets the same
            # answer rather than a second error path that means the same thing.
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
        # `tools` is omitted rather than passed empty: the API rejects `tools=[]`, and
        # this is also the shape of a final, tool-free turn. `max_tokens` is the same
        # DEFAULT_MAX_TOKENS budget as `parse_structured` — thinking is on by default and
        # shares that budget with the output (see the constant), and a tool-use turn is
        # shorter than a plan, so a second constant would only be a second thing to tune.
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = [_to_sdk_tool(tool) for tool in tools]

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[_to_sdk_message(message) for message in messages],
                **extra,
            )
        except anthropic.AnthropicError as exc:
            # Same mapping as `parse_structured`: the SDK's base class, so no bare
            # `except` (§8), and cancellation is a BaseException that passes through (§10).
            raise ProviderError(f"Anthropic request to {self._model} failed: {exc}") from exc
        return _from_sdk_response(response)

    async def aclose(self) -> None:
        """Close the SDK client's connection pool. See `Provider.aclose`.

        Closing is I/O, so it fails like any other call — and mapping it here keeps the
        rule that no SDK exception type leaves this module (§6, §8).
        """
        try:
            await self._client.close()
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"Closing the Anthropic client failed: {exc}") from exc


def _to_sdk_message(message: ProviderMessage) -> MessageParam:
    """Translate one turn into the SDK's shape. Shared by both requests (§2.2), so a
    transcript reads the same whichever call resends it.

    No SDK type crosses back out: replies are decoded by `_from_sdk_response` (§6).
    """
    # Exhaustive rather than an else-branch: the SDK types `role` as a literal, and a
    # fourth `MessageRole` member must fail under mypy here, not be silently relabelled
    # `user` at runtime. `assert_never` is what makes that a compile-time guarantee.
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
    """The body of one turn: a plain string, or the block list a tool turn needs.

    A plain string for a plain turn — it is what the API documents, what every existing
    `parse_structured` transcript already sends, and one fewer thing to read in a test.
    """
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
    # Only when non-empty: the API rejects a text block carrying "". A model that asks
    # for a tool without narrating it is the common case, not an edge one.
    if message.content:
        blocks.append(TextBlockParam(type="text", text=message.content))
    blocks += [
        ToolUseBlockParam(
            type="tool_use",
            id=call.id,
            name=call.name,
            # `dict(...)` because the SDK mutates nothing but types this as `Dict`, and
            # our side is a read-only `Mapping` (§7).
            input=dict(call.arguments),
        )
        for call in message.tool_calls
    ]
    return blocks


def _to_sdk_tool(tool: ToolSpec) -> ToolParam:
    """Translate what a tool publishes into what the API offers the model.

    Field-by-field rather than by `asdict`: the names happen to line up today, and a
    silent rename on either side should break here rather than at the API.
    """
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=dict(tool.input_schema),
    )


def _from_sdk_response(response: Any) -> AssistantTurn:
    """Decode one reply into our types — the point past which the SDK does not exist (§6).

    Block types other than `text` and `tool_use` are dropped, not an error: the model
    thinks by default, and a `thinking` block is not part of the conversation the agent
    loop reasons about.
    """
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    return AssistantTurn(
        text="".join(texts),
        tool_calls=tuple(calls),
        # Summed, because the loop's budget (§10) pays for both halves and a caller that
        # had to add them up would be re-deriving the same number in every agent.
        usage_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )
