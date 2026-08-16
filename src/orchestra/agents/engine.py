"""The supervisor loop: walk the plan's DAG, dispatch each subtask, record what it made.

**Completion-driven, not wave-by-wave.** A `TaskGroup` per graph level makes every step
wait for its level's slowest; here one `TaskGroup` spans the run and the scheduler
re-scans on every completion. `asyncio.Semaphore` bounds concurrency (§10); `in_flight` —
not `status` — is what stops a double dispatch, because a subtask is only `RUNNING` once
it holds the semaphore and a queued one should not show as running.

**A failed subtask ends a step, not the run.** Its dependents never become ready, and
everything independent of it still completes. The verdict is the caller's to draw from
state — partial results beat no results (§8, #8).

**One thing does end the run**: the global step cap, the crude backstop against a plan
that would run forever. #9 replaces it with per-subtask attempt caps and a token budget.
Exceeding it is a `TaskFailure`, never a retry (§10).

**Settling `inputs`.** `Subtask.inputs` names upstream subtask ids, and `state.artifacts`
is keyed by producing subtask id — so the two readings the planner flagged as unsettled
are the same set of strings here.
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
# Counts dispatches, so with no retries yet it only trips on an oversized plan; #9 makes
# it a real budget.
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
            workers: the worker for each role, wired in `app.py`. A role the plan uses and
                this mapping lacks fails the run.
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

        Subtask statuses, `artifacts`, `current_step` and `events` are written to `state`.

        Raises:
            TaskFailure: there is no plan, a role has no worker, or the step cap was
                exceeded. All three end the run — a failed *subtask* does not.
            asyncio.CancelledError: in-flight subtasks are cancelled with it and the error
                is re-raised, never swallowed (§10).
        """
        plan = state.plan
        if plan is None:
            raise TaskFailure("There is no plan to execute. Plan the request first.")
        self._check_roles(plan)

        # Published, not appended: the planner already recorded `plan_created`, but no
        # broker existed then. The plan rides on this one event because a subscriber
        # cannot draw pending rows from transitions it has not seen. Deep-copied because
        # the loop below mutates `Subtask.status` in place and a shared reference would
        # let a subscriber read live state through an event handed to it as history.
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
        # than polling. Cleared *before* the scan, so a completion landing between the
        # scan and the wait is not missed.
        finished = asyncio.Event()
        dispatched = 0
        capped = False

        async with asyncio.TaskGroup() as group:
            while True:
                finished.clear()
                if not capped:
                    for subtask in _ready(plan, in_flight):
                        if dispatched >= self._step_cap:
                            # Stop dispatching but let in-flight work finish: the cap ends
                            # a runaway plan with partial results. Raised after the group
                            # closes — raising inside its body gives an `ExceptionGroup`.
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
            # `app.py` records this message on `state.failure_reason` and the report names
            # the artifacts on disk (#8), so raising does not lose it.
            message = (
                f"Step cap of {self._step_cap} exceeded; the plan is too large to run. "
                f"{done} of {len(plan.subtasks)} subtasks finished before stopping."
            )
            # Emitted before raising: a subscriber that never sees `run_finished` spins
            # forever (§6).
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
                # pointer's shape, so a malformed one fails here rather than poisoning
                # `artifacts`, which is not re-validated in place.
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
        """Raise `TaskFailure` before any work starts if a role has no worker (§9)."""
        missing = sorted({subtask.role for subtask in plan.subtasks} - self._workers.keys())
        if missing:
            raise TaskFailure(f"No worker is registered for roles: {missing}")


def _ready(plan: Plan, in_flight: set[str]) -> list[Subtask]:
    """The subtasks that can start now: pending, not dispatched, dependencies done.

    A failed dependency is never `done`, so the scheduler drains to a stop rather than
    deadlocking on its dependents.
    """
    done = {subtask.id for subtask in plan.subtasks if subtask.status is SubtaskStatus.DONE}
    return [
        subtask
        for subtask in plan.subtasks
        if subtask.status is SubtaskStatus.PENDING
        and subtask.id not in in_flight
        and set(subtask.depends_on) <= done
    ]
