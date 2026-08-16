"""Tests for the composition root (CONVENTIONS.md §3.1, §12).

The end-to-end check the walking skeleton exists for: a request goes in, the planner
plans it, the engine walks the DAG, stub workers write artifacts, the aggregator turns
them into a report, and the ledger comes back with pointers to every one of them — over
`FakeProvider`, so no network (§12).

Two structured calls per run, in order: the plan, then the report. `_responses()` queues
both, so a test that forgets one gets `FakeProvider`'s "no queued response" rather than a
hang.

The tests that build their own `Orchestra` still use `EchoWorker` throughout — this file
is about the composition root, not about any one agent, and a stub keeps a wiring failure
distinguishable from an agent failure. The two that go through `build_orchestra` get the
real Data Retrieval agent (#5), so they also queue `_turns()`: the tool-use conversation
that agent holds, on `FakeProvider`'s separate `turns` queue.

`run_once` is tested through its own wiring: only the vendor adapter is substituted, at
the provider port, so what the observer contract (#11) is asserted against is the real
composition root and not a stand-in for it.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr

from conftest import FakeProvider
from orchestra import app as app_module
from orchestra.agents.aggregator import Aggregator, FigureDraft, ReportDraft
from orchestra.agents.engine import DEFAULT_STEP_CAP, ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.toolsets import QUERY_CSV_TOOL
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.app import Orchestra, build_orchestra, run_once
from orchestra.artifacts import ArtifactStore
from orchestra.config import Config
from orchestra.core.errors import ProviderError
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, SubtaskStatus, TaskEvent
from orchestra.providers.base import AssistantTurn, Provider
from orchestra.tools.base import ToolCall
from orchestra.tools.python_exec import TOOL_NAME as RUN_PYTHON_TOOL
from scenarios import LINEAR

SUMMARY = "Revenue grew in each of the last three quarters."
# LINEAR's first step, and the pointer `EchoWorker` mints for it. Taken from the scenario
# rather than spelled out, so a renamed step fails on the name and not on a stale literal.
FIRST_STEP = LINEAR.draft().subtasks[0].id
FIRST_POINTER = f"artifact:{FIRST_STEP}.txt"


def _responses(*, figure_source: str | None = None) -> list[BaseModel | BaseException | None]:
    """The two answers a run consumes: the plan, then the report draft."""
    figures = (
        [FigureDraft(label="Q3 revenue", value="145", source=figure_source)]
        if figure_source is not None
        else []
    )
    return [LINEAR.draft(), ReportDraft(executive_summary=SUMMARY, key_figures=figures)]


# The analysis step's script, run for real in a subprocess. Stdlib only and no pandas:
# what this run proves is that the pointer the retrieval step minted resolves inside the
# executor, and an import that costs a second per run would prove it no better.
_ANALYSIS_CODE = """\
import json
data = json.load(open("fetch_quarterly_financials.json"))
rows = [row for table in data["datasets"] for row in table["csv"].splitlines()[1:] if row]
print("quarters analysed:", len(rows))
"""


def _turns() -> list[AssistantTurn | BaseException]:
    """Both real agents' conversations, in the order the plan runs them.

    Retrieval queries the CSV and summarises; analysis runs one script over the artifact
    retrieval wrote and summarises. Only the runs that go through `build_orchestra`
    consume these — everywhere else this file wires `EchoWorker` into every role. Both
    tool calls are real, so this is also the check that the shipped dataset is readable
    and that one worker's pointer reaches the next one's subprocess.
    """
    return [
        AssistantTurn(
            text="",
            tool_calls=(ToolCall(id="c1", name=QUERY_CSV_TOOL, arguments={"last_n": 3}),),
            usage_tokens=100,
        ),
        AssistantTurn(text="Retrieved the last three quarters.", usage_tokens=50),
        AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(
                    id="c2",
                    name=RUN_PYTHON_TOOL,
                    arguments={
                        "code": _ANALYSIS_CODE,
                        "inputs": ["artifact:fetch_quarterly_financials.json"],
                    },
                ),
            ),
            usage_tokens=100,
        ),
        AssistantTurn(text="Revenue rose in each quarter.", usage_tokens=50),
    ]


def _orchestra(
    tmp_path: Path, provider: FakeProvider, *, step_cap: int = DEFAULT_STEP_CAP
) -> Orchestra:
    """The real wiring, with the provider substituted — nothing else is faked."""
    broker: Broker[TaskEvent] = Broker()
    # One store, as `build_orchestra` builds it: the aggregator has to resolve the
    # pointers the workers minted.
    store = ArtifactStore(tmp_path)
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, EchoWorker(store))
    return Orchestra(
        planner=Planner(provider),
        engine=ExecutionEngine(workers=workers, broker=broker, step_cap=step_cap),
        aggregator=Aggregator(provider, store),
        provider=provider,
        broker=broker,
    )


@pytest.mark.asyncio
async def test_run_task_plans_executes_and_returns_a_ledger_of_pointers(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(responses=_responses(figure_source=FIRST_POINTER))
    store = ArtifactStore(tmp_path)

    state = await _orchestra(tmp_path, provider).run_task(LINEAR.prompt)

    assert state.plan is not None
    assert state.user_request == LINEAR.prompt
    assert [subtask.status for subtask in state.plan.subtasks] == [SubtaskStatus.DONE] * 3
    assert state.failed_subtasks == []
    for subtask in state.plan.subtasks:
        # Every step's output is readable through the pointer state carries.
        assert subtask.instruction in store.get_text(state.artifacts[subtask.id])
    assert state.events[0].kind is EventKind.PLAN_CREATED
    assert state.events[-1].kind is EventKind.RUN_FINISHED
    # The last link of the skeleton: the run hands back an answer, not just a ledger.
    assert state.final_result is not None
    assert state.final_result.executive_summary == SUMMARY
    assert [figure.source for figure in state.final_result.key_figures] == [FIRST_POINTER]


@pytest.mark.asyncio
async def test_run_task_subscribers_see_the_run_from_plan_to_finish(tmp_path: Path) -> None:
    """The dashboard (#11) attaches to `broker` before the run and needs both ends of it."""
    orchestra = _orchestra(tmp_path, FakeProvider(responses=_responses()))

    async with orchestra.broker.subscribe() as queue:
        await orchestra.run_task(LINEAR.prompt)
        kinds = [queue.get_nowait().kind for _ in range(queue.qsize())]

    assert kinds[0] is EventKind.PLAN_CREATED
    assert kinds[-1] is EventKind.RUN_FINISHED
    assert kinds.count(EventKind.SUBTASK_COMPLETED) == 3


@pytest.mark.asyncio
async def test_run_task_reports_what_finished_when_the_engine_ends_the_run(
    tmp_path: Path,
) -> None:
    """A step cap of 1 against a three-step plan: the engine raises, and `run_task` still
    returns a ledger with a report. Exit 5 with an empty stdout would leave the artifacts
    on disk unmentioned, which is the same to the user as never producing them (#8)."""
    provider = FakeProvider(responses=_responses(figure_source=FIRST_POINTER))

    state = await _orchestra(tmp_path, provider, step_cap=1).run_task(LINEAR.prompt)

    assert state.failure_reason is not None
    assert "Step cap of 1 exceeded" in state.failure_reason
    assert state.failed
    assert state.failed_subtasks == []  # nothing failed; the run was stopped
    assert state.artifacts == {FIRST_STEP: FIRST_POINTER}  # only the first step ran
    assert state.final_result is not None
    # The report was written over what did finish, not over an empty run.
    assert [figure.source for figure in state.final_result.key_figures] == [FIRST_POINTER]


@pytest.mark.asyncio
async def test_run_task_closes_the_provider(tmp_path: Path) -> None:
    provider = FakeProvider(responses=_responses())
    orchestra = _orchestra(tmp_path, provider)

    await orchestra.run_task(LINEAR.prompt)
    await orchestra.aclose()

    assert provider.closed


@pytest.mark.asyncio
async def test_build_orchestra_wires_the_app_from_config_without_touching_the_network(
    tmp_path: Path,
) -> None:
    """Construction is offline and eager: the artifact directory exists before step one (§9)."""
    artifacts = tmp_path / "artifacts"
    config = Config(anthropic_api_key=SecretStr("test-key"), artifact_dir=artifacts)

    orchestra = build_orchestra(config)
    try:
        assert artifacts.is_dir()
        assert orchestra.broker.subscriber_count == 0
    finally:
        await orchestra.aclose()


# --------------------------------------------------------------------------
# `run_once` and its observer: the seam the dashboard attaches to (#11).
# --------------------------------------------------------------------------


@dataclass
class RecordingObserver:
    """A `RunObserver` standing in for `cli/render.py`: subscribes on enter, keeps what
    came through, and counts both edges so a test can assert the run happened between
    them. Nothing here draws — §12 asserts on the data handed to the renderer."""

    entered: int = 0
    exited: int = 0
    events: list[TaskEvent] = field(default_factory=list)

    @asynccontextmanager
    async def __call__(self, broker: Broker[TaskEvent]) -> AsyncIterator[None]:
        """Stay subscribed for the run. See `orchestra.app.RunObserver`."""
        async with broker.subscribe() as queue:
            self.entered += 1
            try:
                yield
            finally:
                # Drained on the way out rather than by a reader task: the queue outlives
                # the run either way, and what is being asserted is which events reached
                # a subscriber attached this early — one attached late is simply missing
                # the first. Also runs on cancellation, which is the point (§10).
                self.events.extend(queue.get_nowait() for _ in range(queue.qsize()))
                self.exited += 1


def _offline_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: FakeProvider) -> None:
    """Let `run_once` load its own config and wire its own services, with the vendor
    adapter swapped at the provider port — the one seam, so no network (§12)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))

    def _create_provider(*, api_key: SecretStr, model: str) -> Provider:
        return provider

    monkeypatch.setattr(app_module, "create_provider", _create_provider)


async def _wait_until(predicate: Callable[[], bool], *, what: str) -> None:
    """Yield to the loop until `predicate` holds. Bounded, so a run that never gets
    there fails the test instead of hanging it.

    Deliberately a copy of `test_engine._wait_until` (§2.3): hoisting a two-caller
    helper into `conftest.py` couples the two modules' timing conventions before there
    is a third to show what varies.
    """
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
async def test_run_once_keeps_the_observer_attached_from_the_first_event_to_the_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dashboard has to be subscribed *before* the run publishes anything: the plan
    rides on the first event, and an observer entered late never learns the pending rows."""
    provider = FakeProvider(responses=_responses(), turns=_turns())
    _offline_run(monkeypatch, tmp_path, provider)
    observer = RecordingObserver()

    state = await run_once(LINEAR.prompt, observer=observer)

    assert (observer.entered, observer.exited) == (1, 1)
    kinds = [event.kind for event in observer.events]
    assert kinds[0] is EventKind.PLAN_CREATED  # nothing was published before it attached
    assert observer.events[0].plan is not None
    assert kinds[-1] is EventKind.RUN_FINISHED  # and it was still attached at the end
    assert state.final_result is not None
    assert provider.closed


@pytest.mark.asyncio
async def test_run_once_without_an_observer_runs_headless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default path `cli/app.py` takes today: no subscriber, same ledger."""
    provider = FakeProvider(responses=_responses(), turns=_turns())
    _offline_run(monkeypatch, tmp_path, provider)

    state = await run_once(LINEAR.prompt)

    assert state.final_result is not None
    assert not state.failed
    assert provider.closed


@pytest.mark.asyncio
async def test_run_once_exits_the_observer_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A provider failure while planning: `Live` must not be left owning the terminal (§5)."""
    provider = FakeProvider(responses=[ProviderError("The provider is unavailable.")])
    _offline_run(monkeypatch, tmp_path, provider)
    observer = RecordingObserver()

    with pytest.raises(ProviderError):
        await run_once(LINEAR.prompt, observer=observer)

    assert (observer.entered, observer.exited) == (1, 1)
    assert provider.closed


@pytest.mark.asyncio
async def test_run_once_cancellation_exits_the_observer_and_closes_the_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl-C's path (§10): the observer is torn down, the provider released, and the
    cancellation re-raised rather than swallowed into a ledger nobody asked for."""
    provider = FakeProvider(responses=_responses(), blocker=asyncio.Event())  # never set
    _offline_run(monkeypatch, tmp_path, provider)
    observer = RecordingObserver()

    run = asyncio.create_task(run_once(LINEAR.prompt, observer=observer))
    await _wait_until(lambda: bool(provider.calls), what="the run to reach the provider")

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert (observer.entered, observer.exited) == (1, 1)
    assert provider.closed
