"""Tests for the execution engine (§10).

The subject is orchestration, not work: `ScriptedWorker` records what it was handed and,
when a test asks, blocks or fails. Plans come from `scenarios.py` through the real planner,
so the graph the engine walks is the one the planner emits.
"""

import asyncio
from collections.abc import Mapping

import pytest

from conftest import ScriptedWorker, dispatches, planned, wait_until
from orchestra.agents.engine import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_STEP_CAP,
    DEFAULT_SUBTASK_ATTEMPTS,
    ExecutionEngine,
)
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
from scenarios import FAN_OUT, LINEAR, ROLE_OMISSION


def _broker() -> Broker[TaskEvent]:
    return Broker()


def _workers(worker: Worker) -> Mapping[AgentRole, Worker]:
    """The same worker for every role — which role does what is the planner's concern."""
    return dict.fromkeys(AgentRole, worker)


def _engine(
    worker: Worker,
    broker: Broker[TaskEvent] | None = None,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    step_cap: int = DEFAULT_STEP_CAP,
    subtask_attempts: int = DEFAULT_SUBTASK_ATTEMPTS,
) -> ExecutionEngine:
    return ExecutionEngine(
        workers=_workers(worker),
        broker=broker if broker is not None else _broker(),
        max_concurrency=max_concurrency,
        step_cap=step_cap,
        subtask_attempts=subtask_attempts,
    )


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
    state = await planned(LINEAR)
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
    state = await planned(LINEAR)
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
    state = await planned(LINEAR)
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
    state = await planned(LINEAR)
    worker = ScriptedWorker()

    await _engine(worker).run(state)
    worker.contexts[0].subtask.status = SubtaskStatus.PENDING

    assert _statuses(state)["fetch_quarterly_financials"] is SubtaskStatus.DONE


@pytest.mark.asyncio
async def test_run_emits_every_transition_to_the_ledger_and_the_broker() -> None:
    """A dropped completion strands a dashboard on a spinner, so both sides are checked."""
    state = await planned(ROLE_OMISSION)
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
    # The planner wrote plan_created to the ledger; the engine republishes it, since the
    # broker did not exist when the planner ran.
    assert _kinds(state.events) == [EventKind.PLAN_CREATED, *lifecycle]
    assert _kinds(published) == [EventKind.PLAN_CREATED, *lifecycle]
    assert [event.subtask_id for event in published[1:3]] == ["fetch_revenue_history"] * 2


@pytest.mark.asyncio
async def test_run_publishes_the_plan_on_plan_created_and_on_no_other_event() -> None:
    """A subscriber joins with no ledger of its own (#11), so the pending rows can only
    reach it here."""
    state = await planned(LINEAR)
    broker = _broker()

    async with broker.subscribe() as queue:
        await _engine(ScriptedWorker(), broker).run(state)
        published = _drain(queue)

    assert state.plan is not None
    assert published[0].kind is EventKind.PLAN_CREATED
    assert published[0].plan is not None
    assert [subtask.id for subtask in published[0].plan.subtasks] == [
        subtask.id for subtask in state.plan.subtasks
    ]
    assert all(event.plan is None for event in published[1:])
    assert all(event.plan is None for event in state.events)


@pytest.mark.asyncio
async def test_run_publishes_a_plan_copy_the_engine_does_not_go_on_mutating() -> None:
    """The event is history, not a live view: the engine writes `Subtask.status` in place,
    so a shared reference would show a subscriber `done` rows it was never told about."""
    state = await planned(LINEAR)
    broker = _broker()

    async with broker.subscribe() as queue:
        await _engine(ScriptedWorker(), broker).run(state)
        plan_created = _drain(queue)[0]

    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}  # the engine moved them
    published_plan = plan_created.plan
    assert published_plan is not None
    assert [subtask.status for subtask in published_plan.subtasks] == [SubtaskStatus.PENDING] * 3


# --------------------------------------------------------------------------
# Concurrency: the plan is a DAG, and the fan-out has to be visible as one.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_dispatches_independent_subtasks_concurrently() -> None:
    """The fan-out scenario's two retrievals need nothing from each other."""
    state = await planned(FAN_OUT)
    worker = ScriptedWorker()

    await _engine(worker).run(state)

    assert worker.peak_concurrency == 2
    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}


@pytest.mark.asyncio
async def test_run_starts_both_retrievals_before_either_completes() -> None:
    """The same overlap read off the event stream instead of sampled inside the worker
    (#17): both retrievals are `subtask_started` before the first `subtask_completed`.

    The gate holds every dispatch open until both starts are on the ledger, so the
    ordering asserted is the engine's and not the scheduler's.
    """
    state = await planned(FAN_OUT)
    gate = asyncio.Event()
    worker = ScriptedWorker(gate=gate)

    run = asyncio.create_task(_engine(worker).run(state))
    await wait_until(
        lambda: _kinds(state.events).count(EventKind.SUBTASK_STARTED) == 2,
        what="both retrievals to report started",
    )
    gate.set()
    await run

    first_completion = _kinds(state.events).index(EventKind.SUBTASK_COMPLETED)
    assert {
        event.subtask_id
        for event in state.events[:first_completion]
        if event.kind is EventKind.SUBTASK_STARTED
    } == {"fetch_our_growth", "fetch_industry_benchmarks"}
    # Overlapping events are not overlapping work: `subtask_started` published before the
    # semaphore would interleave under a serial engine too.
    assert worker.peak_concurrency == 2
    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}


@pytest.mark.asyncio
async def test_run_bounds_concurrency_with_the_semaphore() -> None:
    """§10: never unbounded fan-out. Same plan, same result, one at a time."""
    state = await planned(FAN_OUT)
    worker = ScriptedWorker()

    await _engine(worker, max_concurrency=1).run(state)

    assert worker.peak_concurrency == 1
    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}


@pytest.mark.asyncio
async def test_run_starts_a_subtask_without_waiting_for_an_unrelated_slow_one() -> None:
    """Completion-driven, not wave-by-wave: a scheduler running the graph level by level
    would hold `analyse` back behind the unrelated `slow_fetch`."""
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
    await wait_until(
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
    state = await planned(FAN_OUT)
    worker = ScriptedWorker(fail_ids=frozenset({"fetch_our_growth"}))
    broker = _broker()

    async with broker.subscribe() as queue:
        await _engine(worker, broker).run(state)
        published = _drain(queue)

    assert _statuses(state) == {
        "fetch_our_growth": SubtaskStatus.FAILED,
        # Independent of the failure, so it still runs.
        "fetch_industry_benchmarks": SubtaskStatus.DONE,
        "compare_against_benchmarks": SubtaskStatus.PENDING,
        "chart_growth_trend": SubtaskStatus.PENDING,
    }
    assert EventKind.SUBTASK_FAILED in _kinds(published)
    # Ends rather than deadlocking on a dependency that will never arrive.
    assert _kinds(published)[-1] is EventKind.RUN_FINISHED
    assert [subtask.id for subtask in state.failed_subtasks] == ["fetch_our_growth"]


@pytest.mark.asyncio
async def test_run_fails_the_subtask_when_a_worker_returns_a_malformed_pointer() -> None:
    """Worker output is a trust boundary: a bad pointer must not reach `artifacts` (§7)."""
    state = await planned(LINEAR)
    worker = ScriptedWorker(pointer_override="/etc/passwd")

    await _engine(worker).run(state)

    assert _statuses(state)["fetch_quarterly_financials"] is SubtaskStatus.FAILED
    assert state.artifacts == {}


# --------------------------------------------------------------------------
# Retries: a dispatch that raises is attempted again, up to the cap (#9).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_retries_a_failed_subtask_and_completes_it_on_the_next_attempt() -> None:
    state = await planned(LINEAR)
    worker = ScriptedWorker(fail_once_ids=frozenset({"analyse_trends"}))
    broker = _broker()

    async with broker.subscribe() as queue:
        await _engine(worker, broker).run(state)
        published = _drain(queue)

    assert set(_statuses(state).values()) == {SubtaskStatus.DONE}
    assert dispatches(worker, "analyse_trends") == 2
    warnings = [event for event in published if event.kind is EventKind.SUBTASK_WARNING]
    assert [event.subtask_id for event in warnings] == ["analyse_trends"]
    assert "Attempt 1 of 3" in warnings[0].message
    # A retried subtask is not a finished one: the dashboard and `failed_subtasks` read
    # `subtask_failed` as final.
    assert EventKind.SUBTASK_FAILED not in _kinds(published)
    assert state.failed_subtasks == []


@pytest.mark.asyncio
async def test_run_fails_a_subtask_once_its_attempts_are_spent() -> None:
    """The cap is what stops the retry loop; the run still finishes with partial results."""
    state = await planned(LINEAR)
    worker = ScriptedWorker(fail_ids=frozenset({"fetch_quarterly_financials"}))
    broker = _broker()

    async with broker.subscribe() as queue:
        await _engine(worker, broker).run(state)
        published = _drain(queue)

    assert dispatches(worker, "fetch_quarterly_financials") == DEFAULT_SUBTASK_ATTEMPTS
    assert _statuses(state)["fetch_quarterly_financials"] is SubtaskStatus.FAILED
    assert _kinds(published).count(EventKind.SUBTASK_FAILED) == 1
    assert _kinds(published)[-1] is EventKind.RUN_FINISHED
    assert [subtask.id for subtask in state.failed_subtasks] == ["fetch_quarterly_financials"]


@pytest.mark.asyncio
async def test_run_counts_every_retry_against_the_step_cap() -> None:
    """Retries spend the run's budget, so a failing subtask cannot loop past the cap.

    And the retry the cap took away has to be reconciled: without that, a subtask that
    failed twice was reported as pending and absent from `failed_subtasks` (#9).
    """
    state = await planned(LINEAR)
    worker = ScriptedWorker(fail_ids=frozenset({"fetch_quarterly_financials"}))
    broker = _broker()

    async with broker.subscribe() as queue:
        with pytest.raises(TaskFailure, match="Step cap of 2 exceeded"):
            await _engine(worker, broker, step_cap=2, subtask_attempts=5).run(state)
        published = _drain(queue)

    assert dispatches(worker, "fetch_quarterly_financials") == 2
    assert _statuses(state)["fetch_quarterly_financials"] is SubtaskStatus.FAILED
    assert [subtask.id for subtask in state.failed_subtasks] == ["fetch_quarterly_financials"]
    # Carrying the last attempt's error, and still before the run's verdict (§6).
    failures = [event for event in published if event.kind is EventKind.SUBTASK_FAILED]
    assert [(event.subtask_id, event.message) for event in failures] == [
        ("fetch_quarterly_financials", "fetch_quarterly_financials could not be completed")
    ]
    assert _kinds(published)[-1] is EventKind.RUN_FINISHED


@pytest.mark.parametrize(
    ("failure", "expected_dispatches"),
    [
        (TaskFailure("the plan's dependency order is wrong", retryable=False), 1),
        (RuntimeError("connection reset"), DEFAULT_SUBTASK_ATTEMPTS),
    ],
    ids=["deterministic", "unknown"],
)
@pytest.mark.asyncio
async def test_run_retries_only_a_failure_that_could_go_differently(
    failure: Exception, expected_dispatches: int
) -> None:
    """A bound the worker already hit is reached identically on a retry, and each attempt
    spends a step the rest of the plan needs. Anything not declared deterministic — every
    non-`OrchestraError` included — is still retried."""
    state = await planned(LINEAR)
    worker = ScriptedWorker(fail_ids=frozenset({"fetch_quarterly_financials"}), failure=failure)

    await _engine(worker).run(state)

    assert dispatches(worker, "fetch_quarterly_financials") == expected_dispatches
    assert _statuses(state)["fetch_quarterly_financials"] is SubtaskStatus.FAILED


def test_engine_rejects_a_non_positive_attempt_cap() -> None:
    with pytest.raises(ValueError, match="subtask_attempts"):
        ExecutionEngine(workers={}, broker=_broker(), subtask_attempts=0)


# --------------------------------------------------------------------------
# Limits and wiring: the failures that end the whole run.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stops_when_the_step_cap_is_exceeded() -> None:
    """§10: a bad plan cannot hang the run. Exceeding the cap is a failure, not a retry."""
    state = await planned(LINEAR)
    broker = _broker()

    async with broker.subscribe() as queue:
        with pytest.raises(TaskFailure, match="Step cap of 2 exceeded") as exc_info:
            await _engine(ScriptedWorker(), broker, step_cap=2).run(state)
        published = _drain(queue)

    assert exc_info.value.exit_code is ExitCode.TASK_FAILURE
    # This path raises, so nothing else reports the work that did finish.
    assert "2 of 3 subtasks finished before stopping" in str(exc_info.value)
    assert _statuses(state)["chart_trends"] is SubtaskStatus.PENDING
    # Still emitted, so a subscriber learns the run stopped instead of spinning.
    assert _kinds(published)[-1] is EventKind.RUN_FINISHED


@pytest.mark.asyncio
async def test_run_fails_before_dispatching_when_a_role_has_no_worker() -> None:
    """§9: fail fast — finding out three steps in would waste the steps before it."""
    worker = ScriptedWorker()
    state = await planned(LINEAR)
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
    state = await planned(FAN_OUT)
    worker = ScriptedWorker(gate=asyncio.Event())  # never set

    run = asyncio.create_task(_engine(worker).run(state))
    await wait_until(lambda: worker.running == 2, what="both retrievals to be in flight")

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert worker.running == 0  # every in-flight worker unwound
    # Not marked failed: a cancelled subtask did not fail.
    assert _statuses(state)["fetch_our_growth"] is SubtaskStatus.RUNNING


@pytest.mark.asyncio
async def test_run_cancellation_is_not_retried_as_a_failed_attempt() -> None:
    """`CancelledError` is not an attempt: retrying one would refuse the Ctrl-C."""
    state = await planned(FAN_OUT)
    worker = ScriptedWorker(gate=asyncio.Event())  # never set

    run = asyncio.create_task(_engine(worker).run(state))
    await wait_until(lambda: worker.running == 2, what="both retrievals to be in flight")

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert dispatches(worker, "fetch_our_growth") == 1
    assert EventKind.SUBTASK_WARNING not in _kinds(state.events)
