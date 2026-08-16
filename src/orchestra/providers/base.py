"""The provider port and its factory (CONVENTIONS.md §3.3, §6).

Vendor naming and vendor exceptions stop here: a provider returns our types or raises
from `core/errors.py`.

Two calls, deliberately. `parse_structured` fills one schema in one shot (planner,
aggregator); `send` is a free-text turn that may come back asking for tools (the worker
loop, #5-#7). Collapsing them would make every caller decode a union.

`stream()`, which §6 sketches, has no caller yet — declaring it would force every
adapter and fake to implement dead code. The port grows when the first caller does.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import BaseModel, SecretStr

# Imported, not redefined: a provider-side copy would drift from what tools validate
# against (§1.5). `providers/` -> `tools/` is an edge §3.2 allows.
from orchestra.tools.base import ToolCall, ToolSpec

# Bound to BaseModel: structured output is a trust boundary, so the schema the model
# fills in and the validation of what comes back are the same object (§7).
StructuredT = TypeVar("StructuredT", bound=BaseModel)


class MessageRole(StrEnum):
    """Who authored a turn. A closed set, so an enum rather than bare strings (§7)."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One tool's answer, on its way back to the model.

    Not merged with `tools.base.ToolResponse` (§2.3): that knows nothing about the
    request. `call_id` is the difference, and only the agent loop can supply it.
    """

    call_id: str  # correlates with `ToolCall.id`
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    """One conversation turn — an internal value object, not a trust boundary (§7).

    The transcript is resent whole on every lap of the loop, so an assistant turn
    replays the calls it asked for and the user turn answering it carries the results.

    One class rather than a subclass per shape: the fields default empty, so a plain
    turn still reads as `ProviderMessage(role, content)` and only the adapter has to
    know which combinations a vendor accepts.
    """

    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant turns replay what they asked for
    tool_results: tuple[ToolResult, ...] = ()  # user turns carry the answers
    # When set, the adapter sends this instead of rebuilding the turn. See
    # `AssistantTurn.raw_content`.
    raw_content: object = None


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """One reply: what the model said, what it wants run, what it cost.

    Empty `tool_calls` is how the agent loop knows the model is finished. `usage_tokens`
    is per turn, not accumulated: one provider instance is shared across agents, so
    totalling here would count someone else's spend.

    `raw_content` is an opaque replay handle the loop must pass back. The API wants a
    turn replayed *whole* — a rebuild from `text` and `tool_calls` drops the reasoning
    blocks and is rejected. Typed `object` so the SDK stays quarantined (§6).
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage_tokens: int = 0  # input + output; the agent loop's token budget (§10) counts these
    # Why the model stopped. `"max_tokens"` is a reply cut off mid-generation, which
    # otherwise looks exactly like a finished one. A string, not an enum: the set is the
    # vendor's, and an unseen value must reach the caller rather than fail parsing.
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

        Returns the validated model, or `None` when the reply carried no usable
        structured output — a refusal, or JSON truncated before it closed. Retrying is
        the caller's call; the provider cannot fix it.

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

        One request, not a loop: only the agent knows its iteration cap, its token
        budget (§10) and which tools it will run, so running them and appending the
        results is its job.

        `messages` is oldest first and includes every earlier assistant turn with the
        results answering it. Empty `tool_calls` in the reply means the model is done;
        otherwise each call must be answered with a `ToolResult` carrying its `call_id`.

        Raises:
            ProviderError: the request itself failed (auth, transport, rate limit).
        """
        ...

    async def aclose(self) -> None:
        """Release the connections the provider holds. Idempotent.

        On the port because only the adapter can close the vendor's sockets. An exiting
        process drops them anyway; a long-lived one, or a suite that turns
        `ResourceWarning` into an error, does not.
        """
        ...


def create_provider(*, api_key: SecretStr, model: str, max_tokens: int) -> Provider:
    """Build the provider for `model` — the one place a vendor is chosen (§3.3).

    Adding a vendor means one module and one branch here. `max_tokens` has no default
    here on purpose: the caller holds the configured value, and a default would let it
    go unpassed and unnoticed.
    """
    # Imported here, not at module scope, so the port has no import edge to the SDK:
    # `agents/` importing `Provider` must not pull `anthropic` into the process.
    from orchestra.providers.anthropic import AnthropicProvider

    return AnthropicProvider(api_key=api_key, model=model, max_tokens=max_tokens)
