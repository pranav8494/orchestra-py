"""Tests for the execution engine (CONVENTIONS.md §10, §12).

The subject is orchestration, not work: `ScriptedWorker` does nothing but record what it
was handed and, when a test asks, block or fail. Plans come from `scenarios.py` through
the real planner, so the graph the engine walks is the one the planner emits — the two
cannot drift apart here.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pytest

from conftest import FakeProvider
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.workers.base import Worker
from orchestra.core.errors import ExitCode, TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import (
    AgentRole,
    EventKind,
    Plan,
    Subtask,
    SubtaskContext,
    SubtaskStatus,
    TaskEvent,
    TaskState,
)
from scenarios import FAN_OUT, LINEAR, ROLE_OMISSION, Scenario


@dataclass
class ScriptedWorker:
    """A `Worker` that records its context and does exactly what the test scripted.

    `peak_concurrency` is what makes "these two ran at once" an assertion rather than a
    hope: it is sampled inside the worker, so it measures the engine's dispatch rather
    than the test's timing.
    """

    fail_ids: frozenset[str] = frozenset()
    # Held open until the gate is set. Empty means every subtask waits on it.
    gate: asyncio.Event | None = None
    gate_ids: frozenset[str] = frozenset()
    pointer_override: str | None = None
    contexts: list[SubtaskContext] = field(default_factory=list)
    running: int = 0
    peak_concurrency: int = 0

    async def run(self, context: SubtaskContext) -> str:
        """See `Worker.run`."""
        self.contexts.append(context)
        self.running += 1
        self.peak_concurrency = max(self.peak_concurrency, self.running)
        try:
            if self.gate is not None and (not self.gate_ids or context.subtask.id in self.gate_ids):
                await self.gate.wait()
            else:
                # Yield once, so a sibling dispatched in the same pass can be observed
                # running alongside this one.
                await asyncio.sleep(0)
            if context.subtask.id in self.fail_ids:
                raise TaskFailure(f"{context.subtask.id} could not be completed")
            return self.pointer_override or f"artifact:{context.subtask.id}.txt"
        finally:
            self.running -= 1


def _broker() -> Broker[TaskEvent]:
    return Broker()


def _workers(worker: Worker) -> Mapping[AgentRole, Worker]:
    """The same worker for every role — which role does what is the planner's concern."""
    return dict.fromkeys(AgentRole, worker)


def _engine(
    worker: Worker,
    broker: Broker[TaskEvent] | None = None,
    *,
    max_concurrency: int = 4,
    step_cap: int = 15,
) -> ExecutionEngine:
    return ExecutionEngine(
        workers=_workers(worker),
        broker=broker if broker is not None else _broker(),
        max_concurrency=max_concurrency,
        step_cap=step_cap,
    )


async def _planned(scenario: Scenario) -> TaskState:
    """A ledger holding the scenario's plan, built through the real planner."""
    state = TaskState(user_request=scenario.prompt)
    await Planner(FakeProvider(responses=[scenario.draft()])).create_plan(state)
    return state


async def _wait_until(predicate: Callable[[], bool], *, what: str) -> None:
    """Yield to the loop until `predicate` holds, rather than sleeping a guessed interval.

    Bounded, so a scheduler that never gets there fails the test instead of hanging it.
    """
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


def _kinds(events: list[TaskEvent]) -> list[EventKind]:
    return [event.kind for event in events]


def _statuses(state: TaskState) -> dict[str, SubtaskStatus]:
    assert state.plan is not None
    return {subtask.id: subtask.status for subtask in state.plan.subtasks}


def _drain(queue: asyncio.Queue[TaskEvent]) -> list[TaskEvent]:
    return [queue.get_nowait() for _ in range(queue.qsize())]


# --------------------------------------------------------------------------
# The happy path: dependency order, pointer write-back, event contract.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_completes_every_subtask_and_records_its_pointer() -> None:
    state = await _planned(LINEAR)
    worker = ScriptedWorker()

    await _engine(worker).run(state)

    assert state.plan is not None
    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}
    for subtask in state.plan.subtasks:
        pointer = f"artifact:{subtask.id}.txt"
        assert subtask.output_pointer == pointer
        # Registered under the producing subtask's id, which is what `inputs` names.
        assert state.artifacts[subtask.id] == pointer
    assert state.current_step == len(state.plan.subtasks)


@pytest.mark.asyncio
async def test_run_dispatches_in_dependency_order() -> None:
    state = await _planned(LINEAR)
    worker = ScriptedWorker()

    await _engine(worker).run(state)

    assert [context.subtask.id for context in worker.contexts] == [
        "fetch_quarterly_financials",
        "analyse_trends",
        "chart_trends",
    ]


@pytest.mark.asyncio
async def test_run_passes_only_the_workers_state_slice() -> None:
    """§6: a worker sees its subtask and its declared inputs, never the whole ledger."""
    state = await _planned(LINEAR)
    worker = ScriptedWorker()

    await _engine(worker).run(state)

    by_id = {context.subtask.id: context for context in worker.contexts}
    assert by_id["fetch_quarterly_financials"].inputs == {}
    assert by_id["analyse_trends"].inputs == {
        "fetch_quarterly_financials": "artifact:fetch_quarterly_financials.txt"
    }
    assert set(SubtaskContext.model_fields) == {
        "user_request",
        "subtask",
        "inputs",
        "clarifications",
    }


@pytest.mark.asyncio
async def test_run_gives_the_worker_a_copy_the_ledger_does_not_share() -> None:
    """A worker writing to its slice must not rewrite the status the engine owns."""
    state = await _planned(LINEAR)
    worker = ScriptedWorker()

    await _engine(worker).run(state)
    worker.contexts[0].subtask.status = SubtaskStatus.PENDING

    assert _statuses(state)["fetch_quarterly_financials"] is SubtaskStatus.DONE


@pytest.mark.asyncio
async def test_run_emits_every_transition_to_the_ledger_and_the_broker() -> None:
    """A dropped completion strands a dashboard on a spinner, so both sides are asserted."""
    state = await _planned(ROLE_OMISSION)
    broker = _broker()

    async with broker.subscribe() as queue:
        await _engine(ScriptedWorker(), broker).run(state)
        published = _drain(queue)

    lifecycle = [
        EventKind.SUBTASK_STARTED,
        EventKind.SUBTASK_COMPLETED,
        EventKind.SUBTASK_STARTED,
        EventKind.SUBTASK_COMPLETED,
        EventKind.RUN_FINISHED,
    ]
    # The planner wrote plan_created to the ledger; the engine republished it to the
    # broker, which did not exist when the planner ran.
    assert _kinds(state.events) == [EventKind.PLAN_CREATED, *lifecycle]
    assert _kinds(published) == [EventKind.PLAN_CREATED, *lifecycle]
    assert [event.subtask_id for event in published[1:3]] == ["fetch_revenue_history"] * 2


# --------------------------------------------------------------------------
# Concurrency: the plan is a DAG, and the fan-out has to be visible as one.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_dispatches_independent_subtasks_concurrently() -> None:
    """The fan-out scenario's two retrievals need nothing from each other."""
    state = await _planned(FAN_OUT)
    worker = ScriptedWorker()

    await _engine(worker).run(state)

    assert worker.peak_concurrency == 2
    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}


@pytest.mark.asyncio
async def test_run_bounds_concurrency_with_the_semaphore() -> None:
    """§10: never unbounded fan-out. Same plan, same result, one at a time."""
    state = await _planned(FAN_OUT)
    worker = ScriptedWorker()

    await _engine(worker, max_concurrency=1).run(state)

    assert worker.peak_concurrency == 1
    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}


@pytest.mark.asyncio
async def test_run_starts_a_subtask_without_waiting_for_an_unrelated_slow_one() -> None:
    """Completion-driven, not wave-by-wave.

    `analyse` depends only on `fetch`, so it must start while the unrelated `slow_fetch`
    is still running. A scheduler that ran the graph level by level would hold it back.
    """
    state = TaskState(
        user_request=LINEAR.prompt,
        plan=Plan(
            subtasks=[
                Subtask(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Pull revenue"),
                Subtask(
                    id="analyse",
                    role=AgentRole.ANALYTICS,
                    instruction="Describe the trend",
                    inputs=["fetch"],
                    depends_on=["fetch"],
                ),
                Subtask(
                    id="slow_fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Pull last year"
                ),
            ]
        ),
    )
    gate = asyncio.Event()
    worker = ScriptedWorker(gate=gate, gate_ids=frozenset({"slow_fetch"}))

    run = asyncio.create_task(_engine(worker).run(state))
    await _wait_until(
        lambda: _statuses(state)["analyse"] is SubtaskStatus.DONE, what="analyse to finish"
    )

    assert _statuses(state)["slow_fetch"] is SubtaskStatus.RUNNING

    gate.set()
    await run
    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}


# --------------------------------------------------------------------------
# Failure: one subtask ends, the run does not.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_marks_a_failed_subtask_and_leaves_its_dependents_pending() -> None:
    state = await _planned(FAN_OUT)
    worker = ScriptedWorker(fail_ids=frozenset({"fetch_recent_quarters"}))
    broker = _broker()

    async with broker.subscribe() as queue:
        await _engine(worker, broker).run(state)
        published = _drain(queue)

    assert _statuses(state) == {
        "fetch_recent_quarters": SubtaskStatus.FAILED,
        # Independent of the failure, so it still runs.
        "fetch_prior_year_quarters": SubtaskStatus.DONE,
        "compare_quarters": SubtaskStatus.PENDING,
        "chart_comparison": SubtaskStatus.PENDING,
    }
    assert EventKind.SUBTASK_FAILED in _kinds(published)
    # The run ends rather than deadlocking on a dependency that will never arrive.
    assert _kinds(published)[-1] is EventKind.RUN_FINISHED
    assert [subtask.id for subtask in state.failed_subtasks] == ["fetch_recent_quarters"]


@pytest.mark.asyncio
async def test_run_fails_the_subtask_when_a_worker_returns_a_malformed_pointer() -> None:
    """Worker output is a trust boundary: a bad pointer must not reach `artifacts` (§7)."""
    state = await _planned(LINEAR)
    worker = ScriptedWorker(pointer_override="/etc/passwd")

    await _engine(worker).run(state)

    assert _statuses(state)["fetch_quarterly_financials"] is SubtaskStatus.FAILED
    assert state.artifacts == {}


# --------------------------------------------------------------------------
# Limits and wiring: the failures that end the whole run.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stops_when_the_step_cap_is_exceeded() -> None:
    """§10: a bad plan cannot hang the run. Exceeding the cap is a failure, not a retry."""
    state = await _planned(LINEAR)
    broker = _broker()

    async with broker.subscribe() as queue:
        with pytest.raises(TaskFailure, match="Step cap of 2 exceeded") as exc_info:
            await _engine(ScriptedWorker(), broker, step_cap=2).run(state)
        published = _drain(queue)

    assert exc_info.value.exit_code is ExitCode.TASK_FAILURE
    # Still emitted, so a subscriber learns the run stopped instead of spinning.
    assert _kinds(published)[-1] is EventKind.RUN_FINISHED


@pytest.mark.asyncio
async def test_run_fails_before_dispatching_when_a_role_has_no_worker() -> None:
    """§9: fail fast. Finding this out three steps in would waste the steps before it."""
    worker = ScriptedWorker()
    state = await _planned(LINEAR)
    engine = ExecutionEngine(workers={AgentRole.DATA_RETRIEVAL: worker}, broker=_broker())

    with pytest.raises(TaskFailure, match="No worker is registered"):
        await engine.run(state)

    assert worker.contexts == []


@pytest.mark.asyncio
async def test_run_without_a_plan_fails() -> None:
    state = TaskState(user_request=LINEAR.prompt)

    with pytest.raises(TaskFailure, match="no plan"):
        await _engine(ScriptedWorker()).run(state)


def test_engine_rejects_a_non_positive_concurrency_bound() -> None:
    """A wiring bug, so a `ValueError` — not part of the user-facing taxonomy (§8)."""
    with pytest.raises(ValueError, match="max_concurrency"):
        ExecutionEngine(workers={}, broker=_broker(), max_concurrency=0)


def test_engine_rejects_a_non_positive_step_cap() -> None:
    with pytest.raises(ValueError, match="step_cap"):
        ExecutionEngine(workers={}, broker=_broker(), step_cap=0)


# --------------------------------------------------------------------------
# Cancellation (§10).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cancellation_stops_in_flight_subtasks_and_propagates() -> None:
    """Ctrl-C's path: the TaskGroup unwinds, workers are cancelled, nothing is swallowed."""
    state = await _planned(FAN_OUT)
    worker = ScriptedWorker(gate=asyncio.Event())  # never set

    run = asyncio.create_task(_engine(worker).run(state))
    await _wait_until(lambda: worker.running == 2, what="both retrievals to be in flight")

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert worker.running == 0  # every in-flight worker unwound
    # Not marked failed: a cancelled subtask did not fail, and the run is over anyway.
    assert _statuses(state)["fetch_recent_quarters"] is SubtaskStatus.RUNNING
