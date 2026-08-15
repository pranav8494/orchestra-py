"""Tests for the composition root (CONVENTIONS.md §3.1, §12).

The end-to-end check the walking skeleton exists for: a request goes in, the planner
plans it, the engine walks the DAG, stub workers write artifacts, and the ledger comes
back with pointers to every one of them — over `FakeProvider`, so no network (§12).
"""

from pathlib import Path

import pytest
from pydantic import SecretStr

from conftest import FakeProvider
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.app import Orchestra, build_orchestra
from orchestra.artifacts import ArtifactStore
from orchestra.config import Config
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, SubtaskStatus, TaskEvent
from scenarios import LINEAR


def _orchestra(tmp_path: Path, provider: FakeProvider) -> Orchestra:
    """The real wiring, with the provider substituted — nothing else is faked."""
    broker: Broker[TaskEvent] = Broker()
    echo = EchoWorker(ArtifactStore(tmp_path))
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, echo)
    return Orchestra(
        planner=Planner(provider),
        engine=ExecutionEngine(workers=workers, broker=broker),
        provider=provider,
        broker=broker,
    )


@pytest.mark.asyncio
async def test_run_task_plans_executes_and_returns_a_ledger_of_pointers(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(responses=[LINEAR.draft()])
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


@pytest.mark.asyncio
async def test_run_task_subscribers_see_the_run_from_plan_to_finish(tmp_path: Path) -> None:
    """The dashboard (#11) attaches to `broker` before the run and needs both ends of it."""
    orchestra = _orchestra(tmp_path, FakeProvider(responses=[LINEAR.draft()]))

    async with orchestra.broker.subscribe() as queue:
        await orchestra.run_task(LINEAR.prompt)
        kinds = [queue.get_nowait().kind for _ in range(queue.qsize())]

    assert kinds[0] is EventKind.PLAN_CREATED
    assert kinds[-1] is EventKind.RUN_FINISHED
    assert kinds.count(EventKind.SUBTASK_COMPLETED) == 3


@pytest.mark.asyncio
async def test_run_task_closes_the_provider(tmp_path: Path) -> None:
    provider = FakeProvider(responses=[LINEAR.draft()])
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
