"""The shared typed ledger every agent reads and writes (CONVENTIONS.md §6, §3.3).

Pointers, not blobs: state is serialised into a prompt on every step, so an inlined
payload is re-sent on every later call. `ArtifactPointer` makes that a validated type;
resolving one is `orchestra.artifacts`' job, since `core/` does no I/O (§3.2).

`state_slice()` gives a worker its subtask and declared inputs — not the plan, the
event log, or the other agents' artifacts.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from orchestra.core.errors import TaskFailure

ARTIFACT_PREFIX = "artifact:"

# Allow-list, not deny-list: names come from model output and become filenames. No
# separator, colon, leading dot or control character, so the store cannot escape its root.
ARTIFACT_NAME_PATTERN = r"[\w\- ][\w.\- ]*"

ArtifactPointer = Annotated[
    str, StringConstraints(pattern=rf"^{ARTIFACT_PREFIX}{ARTIFACT_NAME_PATTERN}$")
]


def artifact_path(root: Path, pointer: ArtifactPointer) -> Path:
    """The path `pointer` names inside `root`.

    Pure — whether it exists is `artifacts.py`'s question. Here beside the constants it
    is built from, so the store that writes the file and the CLI that prints its path
    read the pointer the same way.
    """
    return root / pointer.removeprefix(ARTIFACT_PREFIX)


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
    second overlapping set (§1.5). Lossy progress events are not ledger entries."""

    PLAN_CREATED = "plan_created"
    SUBTASK_STARTED = "subtask_started"
    # A note on a step, not a transition: it degraded (a backend fell back), or an attempt
    # failed and will be retried. Never a status, so a consumer keeps its own.
    SUBTASK_WARNING = "subtask_warning"
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

    # forbid: model output, so an unexpected field means the schema drifted.
    # validate_assignment: the engine writes `status` in place.
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

    Validated on construction and assignment — a cycle the model invented would deadlock
    the engine. In-place `subtasks` mutation is not re-validated, so replanning (#3)
    rebuilds rather than edits.
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

        # Kahn's algorithm: a pass that frees nothing leaves only survivors waiting on
        # each other.
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


class TaskEvent(BaseModel):
    """One entry in the event log. Frozen: history is not edited.

    `plan` is the exception to "pointers, not blobs" — a dashboard has to draw the
    *pending* rows, which no transition it has seen can tell it. The engine attaches it
    to the one `plan_created` event, deep-copied, because it mutates `Subtask.status` in
    place afterwards and a shared reference would leak live state into history.

    The copy guards against the engine only: the broker still fans one mutable `Plan`
    out to every subscriber. Nothing writes to it today — worth knowing before a second
    subscriber lands.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: EventKind
    message: str = ""
    subtask_id: str | None = None
    plan: Plan | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubtaskContext(BaseModel):
    """Everything a worker gets, and nothing else. Built by `TaskState.state_slice()`."""

    # Frozen, over a deep-copied subtask: pydantic reuses nested instances, so a worker
    # writing `context.subtask.status` would otherwise mutate the engine's ledger.
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_request: str
    subtask: Subtask
    inputs: dict[str, ArtifactPointer]
    clarifications: list[Clarification]


class KeyFigure(BaseModel):
    """One number, and the upstream artifact the step that computed it was given.

    Not the file the script read — a step may open data staged beside its input, and what
    a figure cites has to be a pointer some step produced or `backed_figures` drops it.

    `source` is a pointer, not free text: it lets the aggregator drop a figure this run
    never produced. One type for both ends — a worker records the number as it computed it,
    the aggregator states it (§1.5).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Empty as a worker records it: the label is the report's wording, added by the
    # aggregator. `FigureDraft.label` keeps `min_length=1`, so a reported figure has one.
    label: str = ""
    value: str  # str, not float: "12.4% QoQ" and "$1.2M" are both figures a report states
    source: ArtifactPointer


class FinalReport(BaseModel):
    """The run's answer: what happened, the numbers behind it, and the chart to open.

    Frozen — written once, at the end. A pointer for the chart, never the chart.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    executive_summary: str
    key_figures: list[KeyFigure] = Field(default_factory=list)
    chart: ArtifactPointer | None = None
    # Inline, unlike `chart`: `cli/format.py` does no I/O, so text that is printed must
    # already be here — and this report is written once at the end, never re-sent to a model.
    chart_ascii: str | None = None


class TaskState(BaseModel):
    """The one ledger for a run. Created by `app.py`, mutated by the engine."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    user_request: str
    # Where this run's pointers resolve, set by `app.py`. A path, not a payload, so the
    # ledger stays pointers-not-blobs (§6) while being resolvable by whoever holds it.
    artifact_dir: Path | None = None
    plan: Plan | None = None
    # Display counter for "Step X of N" only: the plan is a DAG, so under concurrent
    # dispatch there is no single step the run is "at". Counts attempts, so a run with
    # retries in it goes past N.
    current_step: int = Field(default=0, ge=0)
    artifacts: dict[str, ArtifactPointer] = Field(default_factory=dict)  # producer id -> pointer
    events: list[TaskEvent] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    final_result: FinalReport | None = None
    # Why the run stopped short. `app.py` records run-ending failures here rather than
    # aborting, so the report can still name the artifacts already on disk.
    failure_reason: str | None = None

    @property
    def failed_subtasks(self) -> list[Subtask]:
        """The subtasks the engine marked failed."""
        if self.plan is None:
            return []
        return [subtask for subtask in self.plan.subtasks if subtask.status is SubtaskStatus.FAILED]

    @property
    def failed(self) -> bool:
        """Did this run fall short — as a whole, or in one of its steps?

        The ledger answers it so the CLI maps a result to an exit code without a domain
        conditional of its own (§4).
        """
        return bool(self.failure_reason or self.failed_subtasks)

    def backed_figures(self, figures: Iterable[KeyFigure]) -> list[KeyFigure]:
        """Keep only the figures sourced to a pointer this run produced.

        The ledger's rule, not the aggregator's: a figure citing an artifact nobody wrote
        is invention, whichever agent states it.
        """
        produced = set(self.artifacts.values())
        return [figure for figure in figures if figure.source in produced]

    def state_slice(self, subtask: Subtask) -> SubtaskContext:
        """Narrow the ledger to what one worker needs — pointers, never payloads (§6).

        Takes the subtask rather than an id so replanning can slice before committing a
        reshaped plan to state.

        Raises:
            TaskFailure: an input names an artifact nothing produced, so the plan's
                dependency order is wrong. Better than letting a worker invent the data.
        """
        missing = sorted(set(subtask.inputs) - self.artifacts.keys())
        if missing:
            # Not retryable: re-dispatching slices the same ledger to the same verdict.
            raise TaskFailure(
                f"subtask {subtask.id!r} needs artifacts {missing}, which no step has produced",
                retryable=False,
            )
        return SubtaskContext(
            user_request=self.user_request,
            subtask=subtask.model_copy(deep=True),
            inputs={name: self.artifacts[name] for name in subtask.inputs},
            clarifications=list(self.clarifications),  # entries are frozen
        )
