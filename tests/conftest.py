"""Shared test fixtures and doubles.

`FakeProvider` is the substitute that lets the whole suite run without touching the
network — the payoff for keeping vendor SDKs behind the provider port. It answers from
a queue and records what it was asked, so a test asserts on the conversation as well as
on the result. `FakeTool` is the same idea at the other port: every worker's loop takes
tools, and a test that also ran a real one could not say which half broke.

Both live here rather than in the module that first needed them (§3.1): a double imported
across test modules only resolves because `tests/` is not a package, and the copy that
avoids that import is the duplication §2 is about.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import BaseModel

from orchestra.providers.base import AssistantTurn, Provider, ProviderMessage, StructuredT
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse, ToolSpec

# Every setting `Config` reads. Listed once so a new field cannot silently start
# leaking the developer's shell into the suite.
_SETTING_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ARTIFACT_DIR",
    "DATA_DIR",
    # The one that would actually reach the network: with this exported, the `search`
    # tool takes its live path and the suite starts making real requests (§12).
    "TAVILY_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cut every test off from the ambient environment and any real `.env` (§9).

    `Config` reads the environment and a `.env` resolved against the *current working
    directory*, so without both halves of this a developer with an exported key gets a
    different result from CI. Autouse because that hazard is suite-wide, not per-test.
    """
    for var in _SETTING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


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

    Queue entries are returned in order: a pydantic model is handed back as the parsed
    output, `None` stands for a refusal or a truncated reply, and an exception is
    raised. Running the queue dry is a test-authoring bug, so it fails loudly rather
    than repeating the last answer.

    `send` has its own queue and its own record, because a test that scripts an agent's
    tool-use loop cares about the order of *its* turns; interleaving them with the
    planner's structured calls would couple two unrelated scripts. Both share `blocker`
    — a cancellation test holds whichever call is in flight.
    """

    responses: list[BaseModel | BaseException | None] = field(default_factory=list)
    turns: list[AssistantTurn | BaseException] = field(default_factory=list)
    model: str = "fake-model"
    # Set to hold every call open until the test releases it — how a cancellation test
    # gets a request in flight to cancel.
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
        """Record the call and answer from the queue. See `Provider.parse_structured`."""
        self.calls.append(ParseCall(system, tuple(messages), output_format))
        if self.blocker is not None:
            await self.blocker.wait()
        if not self.responses:
            raise AssertionError(f"FakeProvider has no queued response for call {len(self.calls)}")
        answer = self.responses.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        # The queue is heterogeneous by design; the test decides what the call returns.
        return cast("StructuredT | None", answer)

    async def send(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AssistantTurn:
        """Record the call and answer from the `turns` queue. See `Provider.send`."""
        self.send_calls.append(SendCall(system, tuple(messages), tuple(tools)))
        if self.blocker is not None:
            await self.blocker.wait()
        if not self.turns:
            # Loud rather than repeating the last turn: a loop that ran one lap more than
            # the test scripted is exactly the bug this fake is here to catch.
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
    # Conformance is checked by mypy, not by `isinstance`: `runtime_checkable` compares
    # attribute names only, so it would pass a fake whose signature had drifted.
    _PROTOCOL_CHECK: Provider = FakeProvider()
    _TOOL_PROTOCOL_CHECK: BaseTool = FakeTool("x", [])
