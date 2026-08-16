"""The supervisor loop: walk the plan's DAG, dispatch each subtask, record what it made.

**Completion-driven, not wave-by-wave.** The obvious loop runs one `TaskGroup` per
level of the graph, which makes every step in a level wait for the slowest one. Here a
single `TaskGroup` spans the run and the scheduler re-scans on every completion, so a
subtask starts the moment its dependencies are done. `asyncio.Semaphore` bounds how many
run at once (§10); `in_flight` — not `status` — is what keeps the scan from dispatching
the same subtask twice, because a subtask is only `RUNNING` once it holds the semaphore
and the dashboard should not show a queue of them as running.

**A failed subtask ends a step, not the run.** It is marked `failed`, its dependents are
never ready, and everything independent of it still completes. The run's verdict is the
caller's to draw from state — partial results beat no results (§8, and the report in #8).

**One thing does end the run**: the global step cap. It is the crude backstop against a
plan that would otherwise run forever; #9 replaces it with per-subtask attempt caps and
a token budget. Exceeding it is a `TaskFailure`, never a retry (§10).

**Settling `inputs`.** `Subtask.inputs` names upstream subtask ids, and an artifact is
registered in `TaskState.artifacts` under the id of the subtask that produced it. So the
two readings the planner flagged as unsettled — artifact keys, or subtask ids — are the
same set of strings by construction here.
"""

import asyncio
from collections.abc import Mapping

from orchestra.agents.workers.base import Worker
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import (
    AgentRole,
    EventKind,
    Plan,
    Subtask,
    SubtaskStatus,
    TaskEvent,
    TaskState,
)

DEFAULT_MAX_CONCURRENCY = 4
# The research doc's global step budget. Crude on purpose: it counts dispatches, so with
# no retries yet it only trips on an oversized plan (#9 makes it a real budget).
DEFAULT_STEP_CAP = 15


class ExecutionEngine:
    """Runs one plan. Built in `app.py` with its workers and broker."""

    def __init__(
        self,
        *,
        workers: Mapping[AgentRole, Worker],
        broker: Broker[TaskEvent],
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        step_cap: int = DEFAULT_STEP_CAP,
    ) -> None:
        """Wire the engine.

        Args:
            workers: the worker for each role, from `agents/toolsets.py`'s successor in
                `app.py`. A role the plan uses and this mapping lacks fails the run.
            broker: where every state transition is published.
            max_concurrency: how many subtasks may run at once.
            step_cap: how many dispatches the run may make in total.

        Raises:
            ValueError: a non-positive bound — a wiring bug, not a user-facing error.
        """
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be at least 1, got {max_concurrency}")
        if step_cap < 1:
            raise ValueError(f"step_cap must be at least 1, got {step_cap}")
        self._workers = workers
        self._broker = broker
        self._max_concurrency = max_concurrency
        self._step_cap = step_cap

    async def run(self, state: TaskState) -> None:
        """Execute `state.plan`, updating the ledger and emitting events as it goes.

        Args:
            state: the run's ledger. Subtask statuses, `artifacts`, `current_step` and
                `events` are all written here; nothing is returned.

        Raises:
            TaskFailure: there is no plan, a role has no worker, or the step cap was
                exceeded. All three end the run — a failed *subtask* does not.
            asyncio.CancelledError: the run was cancelled. In-flight subtasks are
                cancelled with it and the error is re-raised, never swallowed (§10).
        """
        plan = state.plan
        if plan is None:
            raise TaskFailure("There is no plan to execute. Plan the request first.")
        self._check_roles(plan)

        # Published, not appended: the planner already recorded `plan_created` in the
        # ledger, but no broker existed when it did, and the dashboard needs the event.
        # The plan rides along on this one event because a subscriber cannot draw the
        # pending rows from transitions it has not seen yet. Deep-copied because the
        # loop below mutates every `Subtask.status` in place, and a shared reference
        # would let a subscriber read live state through an event handed to it as
        # history. `_emit` deliberately has no `plan`, so ledger entries stay plan-free.
        await self._broker.publish_lifecycle(
            TaskEvent(
                kind=EventKind.PLAN_CREATED,
                message=f"Executing {len(plan.subtasks)} subtasks",
                plan=plan.model_copy(deep=True),
            )
        )

        semaphore = asyncio.Semaphore(self._max_concurrency)
        in_flight: set[str] = set()
        # Set by every finishing dispatch, so the scheduler wakes on completions rather
        # than polling. Cleared *before* the scan, so a completion that lands between
        # the scan and the wait is not missed.
        finished = asyncio.Event()
        dispatched = 0
        capped = False

        async with asyncio.TaskGroup() as group:
            while True:
                finished.clear()
                if not capped:
                    for subtask in _ready(plan, in_flight):
                        if dispatched >= self._step_cap:
                            # Stop dispatching, but let what is already running finish:
                            # the cap exists to end a runaway plan with partial results,
                            # and in-flight work is bounded by the semaphore anyway.
                            # Raised after the group closes — a `TaskGroup` re-raises a
                            # failure from its own body inside an `ExceptionGroup`.
                            capped = True
                            break
                        dispatched += 1
                        in_flight.add(subtask.id)
                        group.create_task(
                            self._dispatch(state, subtask, semaphore, in_flight, finished),
                            name=f"subtask:{subtask.id}",
                        )
                if not in_flight:
                    break  # nothing running and nothing left that could become ready
                await finished.wait()

        done = sum(1 for subtask in plan.subtasks if subtask.status is SubtaskStatus.DONE)

        if capped:
            # Says what finished rather than promising "partial results". Raising no
            # longer loses it: `app.py` records the message on `state.failure_reason`
            # and the report names the artifacts on disk (#8).
            message = (
                f"Step cap of {self._step_cap} exceeded; the plan is too large to run. "
                f"{done} of {len(plan.subtasks)} subtasks finished before stopping."
            )
            # Emitted before raising: a subscriber that never sees `run_finished` spins
            # forever, and "the run stopped" is exactly what it needs to hear.
            await self._emit(state, EventKind.RUN_FINISHED, message=message)
            raise TaskFailure(message)

        await self._emit(
            state,
            EventKind.RUN_FINISHED,
            message=f"{done} of {len(plan.subtasks)} subtasks completed",
        )

    async def _dispatch(
        self,
        state: TaskState,
        subtask: Subtask,
        semaphore: asyncio.Semaphore,
        in_flight: set[str],
        finished: asyncio.Event,
    ) -> None:
        """Run one subtask under the semaphore and record the outcome in `state`."""
        try:
            async with semaphore:
                subtask.status = SubtaskStatus.RUNNING
                state.current_step += 1
                await self._emit(
                    state,
                    EventKind.SUBTASK_STARTED,
                    subtask_id=subtask.id,
                    message=subtask.instruction,
                )
                pointer = await self._workers[subtask.role].run(state.state_slice(subtask))
                # Assigned to the model field first: `validate_assignment` checks the
                # pointer's shape, so a worker returning a malformed string fails here
                # rather than poisoning `artifacts`, which is not re-validated in place.
                subtask.output_pointer = pointer
                state.artifacts[subtask.id] = pointer
                subtask.status = SubtaskStatus.DONE
                await self._emit(
                    state,
                    EventKind.SUBTASK_COMPLETED,
                    subtask_id=subtask.id,
                    message=pointer,
                )
        except Exception as exc:
            # Not `BaseException`: `CancelledError` must propagate so the TaskGroup can
            # unwind, and a cancelled subtask is not a failed one (§10).
            subtask.status = SubtaskStatus.FAILED
            await self._emit(
                state, EventKind.SUBTASK_FAILED, subtask_id=subtask.id, message=str(exc)
            )
        finally:
            in_flight.discard(subtask.id)
            finished.set()

    async def _emit(
        self,
        state: TaskState,
        kind: EventKind,
        *,
        subtask_id: str | None = None,
        message: str = "",
    ) -> None:
        """Record a transition in the ledger and publish it must-deliver (§6)."""
        event = TaskEvent(kind=kind, message=message, subtask_id=subtask_id)
        state.events.append(event)
        await self._broker.publish_lifecycle(event)

    def _check_roles(self, plan: Plan) -> None:
        """Fail before any work starts if a role in the plan has no worker (§9).

        Raises:
            TaskFailure: the plan needs a role this engine cannot dispatch.
        """
        missing = sorted({subtask.role for subtask in plan.subtasks} - self._workers.keys())
        if missing:
            raise TaskFailure(f"No worker is registered for roles: {missing}")


def _ready(plan: Plan, in_flight: set[str]) -> list[Subtask]:
    """The subtasks that can start now: pending, not dispatched, dependencies done.

    A dependency that failed is never `done`, so its dependents are never ready and the
    scheduler drains to a stop instead of deadlocking on them.
    """
    done = {subtask.id for subtask in plan.subtasks if subtask.status is SubtaskStatus.DONE}
    return [
        subtask
        for subtask in plan.subtasks
        if subtask.status is SubtaskStatus.PENDING
        and subtask.id not in in_flight
        and set(subtask.depends_on) <= done
    ]
