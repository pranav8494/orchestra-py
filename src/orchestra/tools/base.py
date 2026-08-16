"""The tool contract: what a model is offered, what it asks for, what comes back.

**Failures are data.** `run` returns `ToolResponse(is_error=True)` rather than raising.
An exception unwinds the agent's loop and denies the model the one thing it is good at
here — reading "no column named `margin`; the columns are …" and asking again. Raise
only for a programmer error (§6).

**Arguments are untrusted.** `ToolCall.arguments` is whatever the model emitted, so it
is `Mapping[str, object]` rather than a typed shape. Each tool declares a pydantic
params model, publishes `model_json_schema()` as its `input_schema`, and validates
`arguments` through it on entry — the schema the model is shown and the check applied to
its answer are then the same object (§7).

**Why these types live here and not in `providers/`.** The provider transports tool
calls and the tool answers them, so both layers need the same vocabulary. One definition
imported by both beats a provider-side copy that drifts from this one (§1.5). The edge
points `providers/` -> `tools/`; nothing here knows a provider exists.

No `metadata` field, and no `ctx` argument to `run` — §6 sketches both. Nothing needs
either yet: a tool's structured detail is currently its `content`, and everything
run-scoped a tool needs is injected when `agents/toolsets.py` constructs it. They get
added for every implementer in one PR when the first caller wants them, not before.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What the model is told about a tool: its name, when to use it, and its arguments.

    The description is a prompt, not a docstring (§6) — it says when to reach for the
    tool, when not to, and what it will refuse. It is the only thing standing between a
    two-tool agent and a one-tool agent that ignores the second.
    """

    name: str
    description: str
    # JSON Schema, from the tool's params model. `object` rather than `Any`: this
    # crosses into `providers/` and mypy is strict outside the SDK adapter (§7).
    input_schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One invocation the model asked for, as the provider decoded it.

    `id` is the provider's correlation id: the result must be sent back carrying it, or
    the model cannot tell which of two concurrent calls was answered.
    """

    id: str
    name: str
    arguments: Mapping[str, object]  # untrusted — see the module docstring


@dataclass(frozen=True, slots=True)
class ToolResponse:
    """What a tool hands back to the model.

    `content` is read by the model, so it is written for one: an error says what was
    wrong *and* what would work, because the next turn is the retry.

    **Three outcomes, not two.** A lookup that ran correctly and matched nothing is
    neither a failure nor a result: flagging it `is_error` invites a pointless retry,
    and reporting it as content lets an agent record "nothing matched" as if it were
    something retrieved. `is_empty` is the third state — the call worked, the answer is
    that there is no answer. Two booleans rather than a tri-state enum because
    `is_error` is also the wire flag the provider sends back to the model, and one
    concept should not have two spellings (§1.5).
    """

    content: str
    is_error: bool = False
    is_empty: bool = False


class BaseTool(Protocol):
    """A tool an agent can offer its model. Implemented once per tool (§6)."""

    def info(self) -> ToolSpec:
        """Describe this tool to the model. Pure and cheap — called on every turn."""
        ...

    async def run(self, call: ToolCall) -> ToolResponse:
        """Execute one call.

        Args:
            call: the invocation, with arguments this method must validate.

        Returns:
            The answer, or the failure as data — a bad argument, a missing file and an
            empty result are all `ToolResponse`s, never exceptions (§6).

        Raises:
            asyncio.CancelledError: the run was cancelled; propagated, never swallowed
                (§10). It is the only thing that should leave this method.
        """
        ...
