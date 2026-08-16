"""Tests for the composition root (CONVENTIONS.md §3.1, §12).

The end-to-end check the walking skeleton exists for: a request goes in, the planner
plans it, the engine walks the DAG, stub workers write artifacts, the aggregator turns
them into a report, and the ledger comes back with pointers to every one of them — over
`FakeProvider`, so no network (§12).

Two model calls per run, in order: the plan, then the report. `_responses()` queues both,
so a test that forgets one gets `FakeProvider`'s "no queued response" rather than a hang.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr

from conftest import FakeProvider
from orchestra.agents.aggregator import Aggregator, FigureDraft, ReportDraft
from orchestra.agents.engine import DEFAULT_STEP_CAP, ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.app import Orchestra, build_orchestra
from orchestra.artifacts import ArtifactStore
from orchestra.config import Config
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, SubtaskStatus, TaskEvent
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
