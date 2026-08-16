"""Shared test fixtures and doubles.

`FakeProvider` and `FakeTool` keep the suite off the network (§12). Both live here, not in
the module that first needed them: `tests/` is not a package, so a cross-module import of a
double would not resolve (§3.1).
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import BaseModel

from orchestra.artifacts import ArtifactStore
from orchestra.providers.base import AssistantTurn, Provider, ProviderMessage, StructuredT
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse, ToolSpec

# Every setting `Config` reads, listed once so a new field cannot leak the developer's
# shell into the suite. `TAVILY_API_KEY` is the one that reaches the network: exported, it
# puts the `search` tool on its live path (§12).
_SETTING_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MAX_TOKENS",
    "ANTHROPIC_MODEL",
    "ARTIFACT_DIR",
    "DATA_DIR",
    "MAX_CONCURRENCY",
    "TAVILY_API_KEY",
    "WORKER_MAX_TURNS",
    "WORKER_TOKEN_BUDGET",
)

# One second in total, matching the ceiling the polling callers used before they shared
# this helper.
_WAIT_INTERVAL = 0.001
_WAIT_POLLS = 1000


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cut every test off from the ambient environment and any real `.env` (§9).

    `Config` resolves `.env` against the cwd, so both halves are needed for a developer
    with an exported key to match CI. Autouse because the hazard is suite-wide.
    """
    for var in _SETTING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """The run's artifact store, in its own directory under `tmp_path`.

    A subdirectory rather than `tmp_path` itself, as `app.py` builds it from
    `ARTIFACT_DIR`: a test writing a fixture file beside it then cannot be mistaken for
    a stored artifact.
    """
    return ArtifactStore(tmp_path / "artifacts")


def tool_call(name: str, call_id: str = "call-1", **arguments: object) -> ToolCall:
    """One tool call as a provider would decode it."""
    return ToolCall(id=call_id, name=name, arguments=arguments)


async def _wait_until(predicate: Callable[[], bool], *, what: str) -> None:
    """Yield to the loop until `predicate` holds; bounded, so a condition that never
    arrives fails rather than hangs the suite.

    A timed sleep rather than `sleep(0)`: some conditions are set from a worker thread the
    test has no handle on, and only a yield the loop can poll on reliably picks that up.
    """
    for _ in range(_WAIT_POLLS):
        if predicate():
            return
        await asyncio.sleep(_WAIT_INTERVAL)
    raise AssertionError(f"timed out waiting for {what}")


@dataclass(frozen=True, slots=True)
class ParseCall:
    """One recorded `parse_structured` call."""

    system: str
    messages: tuple[ProviderMessage, ...]
    output_format: type[BaseModel]


@dataclass(frozen=True, slots=True)
class SendCall:
    """One recorded `send` call."""

    system: str
    messages: tuple[ProviderMessage, ...]
    tools: tuple[ToolSpec, ...]


@dataclass
class FakeProvider:
    """A `Provider` that answers from a queue and never opens a socket (§12).

    Entries are returned in order: a model is the parsed output, `None` a refusal or
    truncated reply, an exception is raised. `send` keeps a separate queue so a scripted
    tool-use loop is not coupled to the planner's structured calls.
    """

    responses: list[BaseModel | BaseException | None] = field(default_factory=list)
    turns: list[AssistantTurn | BaseException] = field(default_factory=list)
    model: str = "fake-model"
    # Holds every call open until the test releases it, so a cancellation test has a
    # request in flight to cancel.
    blocker: asyncio.Event | None = None
    calls: list[ParseCall] = field(default_factory=list)
    send_calls: list[SendCall] = field(default_factory=list)
    closed: bool = False

    async def parse_structured(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        output_format: type[StructuredT],
    ) -> StructuredT | None:
        self.calls.append(ParseCall(system, tuple(messages), output_format))
        if self.blocker is not None:
            await self.blocker.wait()
        if not self.responses:
            raise AssertionError(f"FakeProvider has no queued response for call {len(self.calls)}")
        answer = self.responses.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        # Heterogeneous by design; the test decides what the call returns.
        return cast("StructuredT | None", answer)

    async def send(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AssistantTurn:
        self.send_calls.append(SendCall(system, tuple(messages), tuple(tools)))
        if self.blocker is not None:
            await self.blocker.wait()
        if not self.turns:
            # Loud rather than repeating the last turn: a loop running one lap more than
            # scripted is the bug this fake exists to catch.
            raise AssertionError(f"FakeProvider has no queued turn for send {len(self.send_calls)}")
        answer = self.turns.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def aclose(self) -> None:
        """Nothing to release; the flag lets a test assert the caller closed it."""
        self.closed = True


class FakeTool:
    """A `BaseTool` that answers from a queue and records what it was called with."""

    def __init__(self, name: str, responses: list[ToolResponse]) -> None:
        self._name = name
        self._responses = responses
        self.calls: list[ToolCall] = []

    def info(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"fake {self._name}",
            input_schema={"type": "object", "properties": {}},
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        self.calls.append(call)
        if not self._responses:
            raise AssertionError(f"FakeTool {self._name!r} has no queued response")
        return self._responses.pop(0)


if TYPE_CHECKING:
    # Conformance checked by mypy, not `isinstance`: `runtime_checkable` compares attribute
    # names only, so it would pass a fake whose signature had drifted.
    _PROTOCOL_CHECK: Provider = FakeProvider()
    _TOOL_PROTOCOL_CHECK: BaseTool = FakeTool("x", [])
