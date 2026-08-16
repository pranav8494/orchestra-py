"""Shared test fixtures and doubles.

`FakeProvider` and `FakeTool` keep the suite off the network (§12). Any double a second
module needs moves here rather than staying in the one that first needed it: one owner, and
no test module reaching into another's internals (§3.1).
"""

import asyncio
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import BaseModel

from orchestra.agents.planner import Planner
from orchestra.agents.workers.base import Worker
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.interrupt import Chat
from orchestra.core.question import Asker, Question
from orchestra.core.state import SubtaskContext, TaskState
from orchestra.providers.base import AssistantTurn, Provider, ProviderMessage, StructuredT
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse, ToolSpec
from scenarios import Scenario

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
    "SUBTASK_ATTEMPTS",
    "TAVILY_API_KEY",
    "WORKER_MAX_TURNS",
    "WORKER_TOKEN_BUDGET",
)

# Colour forcing, which Click and Rich both read. Exported (Claude Code sets
# `FORCE_COLOR=3`) a `CliRunner` pipe is treated as a terminal: help text arrives wrapped
# in ANSI, and the report gets the `Panel` §5 reserves for a real terminal (#38).
_COLOUR_ENV_VARS = ("CLICOLOR", "CLICOLOR_FORCE", "FORCE_COLOR", "NO_COLOR")

# One second in total, matching the ceiling the polling callers used before they shared
# this helper.
_WAIT_INTERVAL = 0.001
_WAIT_POLLS = 1000


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cut every test off from the ambient environment and any real `.env` (§9).

    `Config` resolves `.env` against the cwd, so both halves are needed for a developer
    with an exported key to match CI. Autouse because the hazard is suite-wide.

    The colour variables go with them: not configuration, but they decide which shape the
    CLI writes, so ambient ones make the §5 stream tests depend on the developer's shell.
    """
    for var in (*_SETTING_ENV_VARS, *_COLOUR_ENV_VARS):
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


async def wait_until(predicate: Callable[[], bool], *, what: str) -> None:
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


@dataclass
class ScriptedAsker:
    """An `Asker` that answers from a queue and keeps what it was asked (§12).

    Questions are kept as objects, not strings: what a test checks is that the typed
    question reached the renderer intact.
    """

    answers: list[str] = field(default_factory=list)
    asked: list[Question] = field(default_factory=list)
    # As `FakeProvider`'s: holds the question open so a cancellation test has one in flight.
    blocker: asyncio.Event | None = None

    async def ask(self, question: Question) -> str:
        self.asked.append(question)
        if self.blocker is not None:
            await self.blocker.wait()
        if not self.answers:
            raise AssertionError(f"ScriptedAsker has no answer for question {len(self.asked)}")
        return self.answers.pop(0)


@dataclass
class ScriptedChat:
    """A `Chat` that asks to interrupt N times and talks from a queue (§12).

    `messages` is one pause's worth: the queue is emptied by the first session, and every
    later `next_message` returns "" — the user resuming — so a second pause cannot silently
    consume another pause's script.
    """

    messages: list[str] = field(default_factory=list)
    # How many pauses to ask for. The engine consumes one per `requested`, as a terminal
    # would consume one keypress.
    requests: int = 1
    # When set, the key is only "pressed" once this says so — the seam for choosing *when*
    # in the run the interrupt lands, which the engine's own behaviour turns on.
    armed: Callable[[], bool] | None = None
    # Called as the pause opens, for observing what the run was doing at that moment.
    on_session: Callable[[], None] | None = None
    said: list[str] = field(default_factory=list)
    sessions: int = 0
    # As `FakeProvider`'s: holds the prompt open so a cancellation test has one in flight.
    blocker: asyncio.Event | None = None

    def requested(self) -> bool:
        if self.requests <= 0 or (self.armed is not None and not self.armed()):
            return False
        self.requests -= 1
        return True

    @contextmanager
    def session(self) -> Iterator[None]:
        self.sessions += 1
        if self.on_session is not None:
            self.on_session()
        yield

    async def next_message(self) -> str:
        if self.blocker is not None:
            await self.blocker.wait()
        return self.messages.pop(0) if self.messages else ""

    def say(self, text: str) -> None:
        self.said.append(text)


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


@dataclass
class ScriptedWorker:
    """A `Worker` that records its context and does what the test scripted.

    `peak_concurrency` is sampled inside the worker, so "these two ran at once" measures
    the engine's dispatch rather than the test's timing.
    """

    fail_ids: frozenset[str] = frozenset()
    # Fails only its first dispatch, so a test can watch the engine's retry succeed.
    fail_once_ids: frozenset[str] = frozenset()
    # Raised instead of the default `TaskFailure` when a scripted failure trips.
    failure: Exception | None = None
    # Held open until the gate is set. Empty means every subtask waits on it.
    gate: asyncio.Event | None = None
    gate_ids: frozenset[str] = frozenset()
    pointer_override: str | None = None
    contexts: list[SubtaskContext] = field(default_factory=list)
    running: int = 0
    peak_concurrency: int = 0

    async def run(self, context: SubtaskContext) -> str:
        self.contexts.append(context)
        attempt = dispatches(self, context.subtask.id)
        self.running += 1
        self.peak_concurrency = max(self.peak_concurrency, self.running)
        try:
            if self.gate is not None and (not self.gate_ids or context.subtask.id in self.gate_ids):
                await self.gate.wait()
            else:
                # Yield once, so a sibling dispatched in the same pass is observable
                # running alongside this one.
                await asyncio.sleep(0)
            if context.subtask.id in self.fail_ids or (
                context.subtask.id in self.fail_once_ids and attempt == 1
            ):
                raise self.failure or TaskFailure(f"{context.subtask.id} could not be completed")
            return self.pointer_override or f"artifact:{context.subtask.id}.txt"
        finally:
            self.running -= 1


def dispatches(worker: ScriptedWorker, subtask_id: str) -> int:
    """How many times `subtask_id` reached the worker — one per attempt."""
    return sum(1 for context in worker.contexts if context.subtask.id == subtask_id)


async def planned(scenario: Scenario) -> TaskState:
    """A ledger holding the scenario's plan, built through the real planner."""
    state = TaskState(user_request=scenario.prompt)
    await Planner(FakeProvider(responses=[scenario.draft()])).create_plan(state)
    return state


if TYPE_CHECKING:
    # Conformance checked by mypy, not `isinstance`: `runtime_checkable` compares attribute
    # names only, so it would pass a fake whose signature had drifted.
    _PROTOCOL_CHECK: Provider = FakeProvider()
    _TOOL_PROTOCOL_CHECK: BaseTool = FakeTool("x", [])
    _ASKER_PROTOCOL_CHECK: Asker = ScriptedAsker()
    _WORKER_PROTOCOL_CHECK: Worker = ScriptedWorker()
    _CHAT_PROTOCOL_CHECK: Chat = ScriptedChat()
