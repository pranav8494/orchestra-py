"""The shared typed ledger every agent reads and writes (CONVENTIONS.md §6, §3.3).

**Pointers, not blobs.** State is serialised into a prompt on every step, so an inlined
payload is re-sent to the model on every later call. Payloads go to
`orchestra.artifacts`; `ArtifactPointer` makes that a validated type rather than a
convention. This module never resolves a pointer — no I/O in `core/` (§3.2) — so a
pointer only means something against the store that minted it.

**Only its slice.** `state_slice()` gives a worker its subtask and declared inputs, not
the plan, the event log, or the other agents' artifacts.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from orchestra.core.errors import TaskFailure

ARTIFACT_PREFIX = "artifact:"

# Allow-list, not deny-list: names arrive from model output and become filenames, so
# separators, colons, leading dots and control characters are absent by construction.
# This is what makes `ArtifactStore` unable to escape its root.
ARTIFACT_NAME_PATTERN = r"[\w\- ][\w.\- ]*"

ArtifactPointer = Annotated[
    str, StringConstraints(pattern=rf"^{ARTIFACT_PREFIX}{ARTIFACT_NAME_PATTERN}$")
]


class AgentRole(StrEnum):
    """Worker roles a subtask can be assigned to."""

    DATA_RETRIEVAL = "data_retrieval"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"


class SubtaskStatus(StrEnum):
    """Driven by the engine: pending -> running -> done/failed."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class EventKind(StrEnum):
    """Lifecycle events, defined here so `core/events.py` publishes these rather than a
    second overlapping set (§1.5). Lossy progress events are not durable ledger entries."""

    PLAN_CREATED = "plan_created"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"
    SUBTASK_FAILED = "subtask_failed"
    RUN_FINISHED = "run_finished"


class Clarification(BaseModel):
    """An answered question. The pending, blocking request is `core/question.py`'s (#10)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    answer: str


class Subtask(BaseModel):
    """One unit of work, assigned to one role.

    Artifacts are registered in `TaskState.artifacts` under the producing subtask's id,
    so an entry in `inputs` names an upstream subtask.
    """

    # extra="forbid": these come from model output, so an unexpected field means the
    # schema drifted. validate_assignment: the engine writes `status` in place.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    role: AgentRole
    instruction: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)  # keys into TaskState.artifacts — data
    depends_on: list[str] = Field(default_factory=list)  # subtask ids — ordering
    status: SubtaskStatus = SubtaskStatus.PENDING
    output_pointer: ArtifactPointer | None = None


class Plan(BaseModel):
    """The decomposition: a DAG of subtasks, not a list.

    Validated on construction and assignment, because a cycle the model invented would
    otherwise deadlock the engine. In-place `subtasks` mutation is not re-validated, so
    replanning (#3) rebuilds rather than edits.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    subtasks: list[Subtask] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_dag(self) -> "Plan":
        ids = [subtask.id for subtask in self.subtasks]
        duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
        if duplicates:
            raise ValueError(f"duplicate subtask ids: {duplicates}")

        known = set(ids)
        for subtask in self.subtasks:
            unknown = sorted(set(subtask.depends_on) - known)
            if unknown:
                raise ValueError(f"subtask {subtask.id!r} depends on unknown subtasks: {unknown}")

        # Kahn's algorithm. If a pass frees nothing and work remains, every survivor is
        # waiting on another survivor.
        resolved: set[str] = set()
        remaining = list(self.subtasks)
        while remaining:
            ready = [subtask for subtask in remaining if set(subtask.depends_on) <= resolved]
            if not ready:
                stuck = sorted(subtask.id for subtask in remaining)
                raise ValueError(f"plan has a dependency cycle among: {stuck}")
            resolved.update(subtask.id for subtask in ready)
            remaining = [subtask for subtask in remaining if subtask.id not in resolved]
        return self


# Defined after `Plan` because `plan` below refers to it. Ordering, not layering.
class TaskEvent(BaseModel):
    """One entry in the event log. Frozen: history is not edited.

    `plan` is the exception to "pointers, not blobs": a dashboard subscribing to the
    stream has to draw the *pending* rows too, and it cannot learn them from a
    transition it has not seen yet. So the engine attaches the plan to the one
    `plan_created` event it publishes — a deep copy, because the engine then mutates
    every `Subtask.status` in place and a shared reference would let the renderer read
    live state through an event it was handed as history. Every other event leaves it
    `None`, and the ledger's own entries never carry one: `ExecutionEngine._emit`
    appends without it, so `TaskState.events` stays free of plan copies.

    The copy protects subscribers from the *engine*, not from each other: `Plan` and
    `Subtask` are mutable, and the broker fans one event object out to everyone, so a
    subscriber that wrote to `event.plan` would be writing to every other subscriber's
    copy. Nothing does — `cli/render.py` reads it into frozen rows — and with the
    dashboard as the only subscriber it is unreachable today. Worth knowing before
    adding a second one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: EventKind
    message: str = ""
    subtask_id: str | None = None
    plan: Plan | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubtaskContext(BaseModel):
    """Everything a worker gets, and nothing else. Built by `TaskState.state_slice()`."""

    # Frozen, over a deep-copied subtask: pydantic reuses nested instances, so otherwise
    # a worker writing `context.subtask.status` mutates the ledger the engine owns.
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_request: str
    subtask: Subtask
    inputs: dict[str, ArtifactPointer]
    clarifications: list[Clarification]


class KeyFigure(BaseModel):
    """One number the report states, and the artifact it was read from.

    `source` is a pointer, not free text: an unsourced figure is a claim nobody can
    check, and the type is what lets the aggregator drop one this run never produced.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    value: str  # str, not float: "12.4% QoQ" and "$1.2M" are both figures a report states
    source: ArtifactPointer


class FinalReport(BaseModel):
    """The run's answer: what happened, the numbers behind it, and the chart to open.

    Frozen, like `TaskEvent` — written once, at the end. A pointer for the chart, never
    the chart.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    executive_summary: str
    key_figures: list[KeyFigure] = Field(default_factory=list)
    chart: ArtifactPointer | None = None


class TaskState(BaseModel):
    """The one ledger for a run. Created by `app.py`, mutated by the engine."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    user_request: str
    plan: Plan | None = None
    # Display counter for "Step X of N". The plan is a DAG, so under concurrent dispatch
    # there is no single step the run is "at".
    current_step: int = Field(default=0, ge=0)
    artifacts: dict[str, ArtifactPointer] = Field(default_factory=dict)  # producer id -> pointer
    events: list[TaskEvent] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    final_result: FinalReport | None = None
    # Why the run stopped short, when it did. `app.py` records the engine's run-ending
    # failures here rather than letting them abort the command, so the report can still
    # name the artifacts already on disk.
    failure_reason: str | None = None

    @property
    def failed_subtasks(self) -> list[Subtask]:
        """The subtasks the engine marked failed — empty when the run produced everything.

        The ledger answers "did this run succeed?", so the CLI can map a result to an
        exit code without reimplementing the question (§4).
        """
        if self.plan is None:
            return []
        return [subtask for subtask in self.plan.subtasks if subtask.status is SubtaskStatus.FAILED]

    @property
    def failed(self) -> bool:
        """Did this run fall short — as a whole, or in one of its steps?

        The single question the CLI asks, so the command body maps a result to an exit
        code without holding a domain conditional of its own (§4).
        """
        return bool(self.failure_reason or self.failed_subtasks)

    def state_slice(self, subtask: Subtask) -> SubtaskContext:
        """Narrow the ledger to what one worker needs (§6).

        Takes the subtask rather than an id so replanning can slice before committing a
        reshaped plan to state.

        Args:
            subtask: the subtask about to be dispatched.

        Returns:
            A frozen context carrying pointers, never payloads.

        Raises:
            TaskFailure: an input names an artifact nothing produced, so the plan's
                dependency order is wrong. Better than handing a worker an empty input
                and letting it invent the data.
        """
        missing = sorted(set(subtask.inputs) - self.artifacts.keys())
        if missing:
            raise TaskFailure(
                f"subtask {subtask.id!r} needs artifacts {missing}, which no step has produced"
            )
        return SubtaskContext(
            user_request=self.user_request,
            subtask=subtask.model_copy(deep=True),
            inputs={name: self.artifacts[name] for name in subtask.inputs},
            clarifications=list(self.clarifications),  # entries are frozen
        )
