"""The provider port and its factory (CONVENTIONS.md §3.3, §6).

Callers name a model and a message list; what a vendor calls those, and which
exceptions it raises, stops here. A provider returns our types or raises from the
taxonomy in `core/errors.py` — never an SDK type.

**Two calls, deliberately.** `parse_structured` fills one schema in one shot — the
planner and the aggregator want an answer, not a conversation. `send` is the other shape:
a free-text turn that may come back asking for tools, which is the loop the worker agents
(#5-#7) run. Neither expresses the other — collapsing them would make every caller decode
a union — so both live on the one port rather than in two (§1.5).

`stream()`, which §6 also sketches, still does not exist: no caller needs a token stream,
and declaring it would force every adapter and every fake to implement dead code. The port
grows when the first caller does.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import BaseModel, SecretStr

# Imported from `tools/`, not redefined here. The provider transports tool calls and a
# tool answers them, so both speak one vocabulary; a provider-side copy of `ToolCall` or
# `ToolSpec` would drift from the one the tools validate against (§1.5). The edge points
# `providers/` -> `tools/`, which §3.2 allows; nothing in `tools/` knows this exists.
from orchestra.tools.base import ToolCall, ToolSpec

# Bound to BaseModel because structured output is a trust boundary: the schema the
# model fills in and the validation of what comes back are the same object (§7).
StructuredT = TypeVar("StructuredT", bound=BaseModel)


class MessageRole(StrEnum):
    """Who authored a turn. A closed set, so an enum rather than bare strings (§7)."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One tool's answer, on its way back to the model.

    Not the same thing as `tools.base.ToolResponse`, and deliberately not merged with it
    (§2.3): a `ToolResponse` is what a tool produced, knowing nothing about the request;
    this is that content addressed to one `ToolCall`. The `call_id` is the whole
    difference and it is the agent loop, not the tool, that can supply it — a model
    running two calls in a turn cannot otherwise tell which one was answered.
    """

    call_id: str  # correlates with `ToolCall.id`
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    """One conversation turn — an internal value object, not a trust boundary (§7).

    A turn carries text, tool calls, or tool results, and the role says which to expect:
    an assistant turn replays the calls the model asked for, and the user turn answering
    it carries the results. Both are needed because the transcript is resent whole on
    every iteration of the loop — a model that cannot see the call it made cannot read
    the answer to it.

    One class rather than a role-specific subclass per shape: the fields are empty by
    default, so a plain text turn still reads as `ProviderMessage(role, content)`, and
    the adapter is the only code that has to know which combinations a vendor accepts.
    """

    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant turns replay what they asked for
    tool_results: tuple[ToolResult, ...] = ()  # user turns carry the answers
    # An assistant turn replayed verbatim. See `AssistantTurn.raw_content`; when set, the
    # adapter sends this instead of rebuilding the turn from `content` and `tool_calls`.
    raw_content: object = None


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """One reply: what the model said, what it wants run, what it cost.

    `tool_calls` empty is how the agent loop knows the model is finished; anything else
    is another lap. `usage_tokens` is reported per turn rather than accumulated here
    because the budget belongs to the loop, not to the provider — a provider instance is
    shared across agents and would otherwise count someone else's spend.

    **`raw_content` is an opaque replay handle, and the loop is required to pass it
    back.** A model that reasons before calling a tool returns that reasoning as part of
    the turn, and the API requires the turn to be replayed *whole* on the next request —
    a reconstruction carrying only the text and the tool calls is rejected, because the
    blocks it dropped are the ones that must come back unchanged. So the adapter keeps
    the vendor's own turn here and the worker hands it straight back on
    `ProviderMessage.raw_content`.

    Typed `object`, not the SDK's type: this is a token to carry, not a value to read.
    Nothing outside `providers/` may inspect it, which is what keeps the SDK quarantined
    (§6) while still letting the transcript survive a round trip.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage_tokens: int = 0  # input + output; the agent loop's token budget (§10) counts these
    # Why the model stopped. `"max_tokens"` means the reply was cut off mid-generation:
    # the blocks that arrived are real, but the turn is unfinished, and a truncated reply
    # with no tool call in it is indistinguishable from a finished one unless the loop
    # checks this. Vendor-neutral strings, not an enum — the set is the vendor's, and a
    # value we have not seen must reach the caller rather than fail parsing.
    stop_reason: str = ""
    raw_content: object = None


class Provider(Protocol):
    """A model provider. Implemented once per vendor, plus `FakeProvider` in tests."""

    @property
    def model(self) -> str:
        """The model identifier requests are sent to."""
        ...

    async def parse_structured(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        output_format: type[StructuredT],
    ) -> StructuredT | None:
        """Ask the model for a response matching `output_format`.

        Args:
            system: the system prompt, from `orchestra.prompts`.
            messages: the conversation so far, oldest first.
            output_format: the pydantic model the response must populate.

        Returns:
            The validated model, or `None` when the response carried no usable
            structured output — a refusal, or a reply truncated before the JSON
            closed. Callers decide whether that is worth another attempt; it is not
            an error the provider can fix.

        Raises:
            ProviderError: the request itself failed (auth, transport, rate limit).
        """
        ...

    async def send(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AssistantTurn:
        """Send one conversational turn and return what the model replied.

        One request, not a loop: running the tools and appending their results is the
        agent's job, because only the agent knows its iteration cap, its token budget
        (§10) and which tools it is willing to run. A provider that looped would hide
        both from the caller that has to bound them.

        Args:
            system: the system prompt, from `orchestra.prompts`.
            messages: the conversation so far, oldest first — including every earlier
                assistant turn and the tool results answering it.
            tools: the tools the model may call this turn. Empty means a plain reply is
                the only thing it can do.

        Returns:
            The reply. `tool_calls` empty means the model is done; otherwise each call
            must be run and answered with a `ToolResult` carrying its `call_id`.

        Raises:
            ProviderError: the request itself failed (auth, transport, rate limit).
        """
        ...

    async def aclose(self) -> None:
        """Release the connections the provider holds. Idempotent.

        On the port because the sockets are the vendor's, so only the adapter can close
        them and no caller can reach past `Provider` to do it. A process that exits drops
        them anyway; a long-lived one, or a test suite that turns `ResourceWarning` into
        an error, does not. Callers should use `contextlib.aclosing`.
        """
        ...


def create_provider(*, api_key: SecretStr, model: str) -> Provider:
    """Build the provider for `model` — the one place a vendor is chosen (§3.3).

    Args:
        api_key: the vendor credential, kept in a `SecretStr` so it cannot leak
            through a repr or a log record until the adapter unwraps it (§9).
        model: the model identifier, from `Config.anthropic_model`.

    Returns:
        A `Provider`. Adding a vendor means one module and one branch here.
    """
    # Imported here rather than at module scope so the port has no import edge to the
    # SDK: `agents/` importing `Provider` must not pull `anthropic` into the process.
    from orchestra.providers.anthropic import AnthropicProvider

    return AnthropicProvider(api_key=api_key, model=model)
