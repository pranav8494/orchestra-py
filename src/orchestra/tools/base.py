"""The tool contract: what a model is offered, what it asks for, what comes back.

- **Failures are data** — `run` returns `ToolResponse(is_error=True)`; raising unwinds the
  agent loop and denies the model its retry (§6).
- **Arguments are untrusted** — one pydantic params model per tool serves as both
  `input_schema` and entry validation, so the schema shown is the check applied (§7).
- **Here, not in `providers/`** — one definition imported by both beats a copy that
  drifts (§1.5). The edge points `providers/` -> `tools/`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What the model is told about a tool. `description` is a prompt, not a docstring (§6)."""

    name: str
    description: str
    # `object` rather than `Any`: this crosses into `providers/` and mypy is strict
    # outside the SDK adapter (§7).
    input_schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One invocation the model asked for, as the provider decoded it."""

    # The provider's correlation id: the result must carry it back, or the model cannot
    # tell which of two concurrent calls was answered.
    id: str
    name: str
    arguments: Mapping[str, object]  # untrusted — see the module docstring


@dataclass(frozen=True, slots=True)
class ToolResponse:
    """What a tool hands back to the model.

    Three outcomes, not two: a lookup that ran and matched nothing is neither a failure
    (`is_error` invites a pointless retry) nor a result (a caller would count it as
    something retrieved). Booleans rather than an enum because `is_error` is also the wire
    flag sent back to the model (§1.5).
    """

    # Read by the model, so an error says what was wrong *and* what would work.
    content: str
    is_error: bool = False
    is_empty: bool = False
    # Succeeded, but not as intended — a backend fell back, an output degraded. Structured
    # rather than grepped out of `content`, which stops working when the wording changes.
    warning: str = ""


class BaseTool(Protocol):
    """A tool an agent can offer its model. Implemented once per tool (§6)."""

    def info(self) -> ToolSpec:
        """Describe this tool to the model. Pure and cheap — called on every turn."""
        ...

    async def run(self, call: ToolCall) -> ToolResponse:
        """Execute one call, validating `call.arguments` first.

        A bad argument, a missing file and an empty result are all `ToolResponse`s, never
        exceptions (§6). Only `asyncio.CancelledError` may leave this method (§10).
        """
        ...


def format_validation_error(exc: ValidationError) -> str:
    """Flatten a pydantic error into one line the model can act on.

    Shared by all three tools rather than copied — the third copy is where §2.2 says the
    shape stopped being a coincidence. Not `str(exc)`: pydantic echoes the rejected input,
    unbounded model output, and adds a docs URL the model cannot follow.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors()
    )
