"""The tool contract: what a model is offered, what it asks for, what comes back.

**Failures are data.** `run` returns `ToolResponse(is_error=True)` rather than raising —
an exception unwinds the agent's loop and denies the model its retry (§6).

**Arguments are untrusted.** Each tool declares a pydantic params model, publishes its
schema as `input_schema`, and validates `arguments` through it on entry, so the schema
shown and the check applied are one object (§7).

**Here, not in `providers/`.** The provider transports tool calls and the tool answers
them; one definition imported by both beats a copy that drifts (§1.5). The edge points
`providers/` -> `tools/`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What the model is told about a tool: its name, when to use it, and its arguments.

    The description is a prompt, not a docstring (§6): when to reach for the tool, when
    not to, and what it will refuse.
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

    `content` is read by the model, so an error says what was wrong *and* what would
    work — the next turn is the retry.

    **Three outcomes, not two.** A lookup that ran and matched nothing is neither a
    failure nor a result: `is_error` invites a pointless retry, plain content lets a
    caller record "nothing matched" as something retrieved. `is_empty` is that third
    state. Booleans rather than an enum because `is_error` is also the wire flag sent
    back to the model (§1.5).
    """

    content: str
    is_error: bool = False
    is_empty: bool = False
    # Succeeded, but not as intended — a backend fell back, an output degraded.
    # Structured, not left for the caller to grep out of `content`: prose-matching stops
    # working the day the wording changes. `content` says it too, for the model.
    warning: str = ""


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


def format_validation_error(exc: ValidationError) -> str:
    """Flatten a pydantic error into one line the model can act on.

    Here rather than in each tool: `query_csv`, `search` and `run_python` all render a
    rejected `arguments` mapping the same way, and the third copy is where §2.2 says the
    shape has stopped being a coincidence. Beside `BaseTool` because that is the contract
    it serves — a validation failure is the commonest thing a tool returns as data (§6).

    Not `str(exc)`: pydantic's own rendering echoes the rejected input, which is model
    output of unbounded size, and adds a docs URL the model cannot follow.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors()
    )
