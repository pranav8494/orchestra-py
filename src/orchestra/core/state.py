"""The shared typed ledger every agent reads and writes (CONVENTIONS.md §6, §3.3).

**Pointers, not blobs.** This object is serialised into a prompt on every step, so a
DataFrame inlined here would be re-sent to the model on every subsequent call — the
context bloat the research doc's Centralized Memory Ledger exists to avoid. Large
payloads go to `orchestra.artifacts` and only the `artifact:<name>` string lands here.
`ArtifactPointer` makes that a validated type rather than a convention, because a rule
enforced only in a docstring is a rule the planner will break. This module never
resolves a pointer: it holds no I/O, which is what keeps `core/` portable behind
another front end (§3.2). Pointers are therefore only meaningful against the store that
minted them — a state file reloaded against a different `artifact_dir` resolves to
nothing.

**Only its slice.** `TaskState.state_slice()` hands a worker its subtask, the request,
and the pointers it declared — not the plan, not the event log, not the other agents'
artifacts. Selective context isolation, applied at the one place workers read state.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from orchestra.core.errors import TaskFailure

# The pointer format, defined here because state is what carries pointers around.
# `orchestra.artifacts` imports both rather than restating them — one definition of the
# vocabulary, and the dependency points inward (§3.2).
ARTIFACT_PREFIX = "artifact:"

# An allow-list, not a deny-list: artifact names arrive from model output and become
# filenames, so anything that could traverse (`/`, `\`, `:`, a leading `.`) or break an
# open() call (NUL, control characters) is absent by construction rather than blocked
# case by case. This is what makes `artifacts.ArtifactStore` provably unable to escape
# its root.
ARTIFACT_NAME_PATTERN = r"[\w\- ][\w.\- ]*"

ArtifactPointer = Annotated[
    str, StringConstraints(pattern=rf"^{ARTIFACT_PREFIX}{ARTIFACT_NAME_PATTERN}$")
]


class AgentRole(StrEnum):
    """The worker roles a subtask can be assigned to (§7: closed string set)."""

    DATA_RETRIEVAL = "data_retrieval"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"


class SubtaskStatus(StrEnum):
    """Lifecycle of one subtask. The engine (#4) drives pending -> running -> done/failed."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class EventKind(StrEnum):
    """What the ledger records: the lifecycle events §6 requires the broker to deliver.

    Defined here, with the log that persists them, so `core/events.py` publishes *these*
    rather than declaring a second overlapping set (§1.5, §6). Deliberately lifecycle
    only — §6's other publish mode is lossy progress, which by definition must not become
    a durable ledger entry.
    """

    PLAN_CREATED = "plan_created"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"
    SUBTASK_FAILED = "subtask_failed"
    RUN_FINISHED = "run_finished"


class TaskEvent(BaseModel):
    """One entry in the ledger's event log. Frozen: history is not edited."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: EventKind
    message: str = ""
    subtask_id: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Clarification(BaseModel):
    """A question put to the user and the answer given. Frozen: an answered pair is history.

    Answered pairs only — the *pending* request, which has no answer yet and blocks until
    it does, is `core/question.py`'s type (§3.3, #10). This is the record it leaves behind.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    answer: str


class Subtask(BaseModel):
    """One unit of work, assigned to one role.

    `inputs` and `depends_on` are deliberately separate: `depends_on` is ordering,
    `inputs` is data. A subtask can need another to finish without consuming its
    artifact, and can read an artifact that no subtask produced.

    Artifacts are registered in `TaskState.artifacts` under the **producing subtask's
    id**, so an entry in `inputs` names an upstream subtask. That convention is what
    links `output_pointer` on the producer to `inputs` on the consumer without a
    second naming scheme to keep in sync.
    """

    # extra="forbid": these arrive from model output, and an unexpected field means the
    # plan schema drifted — better a validation error than a silently ignored key (§7).
    # validate_assignment: the engine mutates `status` in place, so assignment is a
    # trust boundary too.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    role: AgentRole
    instruction: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)  # keys into TaskState.artifacts
    depends_on: list[str] = Field(default_factory=list)  # ids of subtasks that must finish first
    status: SubtaskStatus = SubtaskStatus.PENDING
    output_pointer: ArtifactPointer | None = None


class Plan(BaseModel):
    """The decomposition: a DAG of subtasks, not a list.

    Validated as a DAG on construction and on assignment, at the trust boundary, because
    the alternative is the execution engine deadlocking on a cycle the model invented
    (§7). In-place mutation of `subtasks` is *not* re-validated — pydantic cannot see a
    `list.append`, so a caller reshaping the plan (#3's replanning) must rebuild it.
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

        # Kahn's algorithm: peel off whatever is currently unblocked. If a pass frees
        # nothing and work remains, every survivor is waiting on another survivor.
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


class SubtaskContext(BaseModel):
    """Everything a worker gets, and nothing else. Built by `TaskState.state_slice()`."""

    # frozen, over a deep-copied subtask: isolation has to mean containment, not just
    # omission. Pydantic reuses nested model instances, so without the copy a worker
    # writing `context.subtask.status` would reach through and mutate the shared ledger
    # the engine owns.
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_request: str
    subtask: Subtask
    inputs: dict[str, ArtifactPointer]  # artifact name -> pointer, narrowed to this subtask's
    clarifications: list[Clarification]


class TaskState(BaseModel):
    """The one ledger for a run. Created by `app.py`, mutated by the engine."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    user_request: str
    plan: Plan | None = None
    # Completed steps, for the "Step X of N" line in the dashboard and in the replanning
    # prompt (research doc §4). A display counter: the plan is a DAG, so with concurrent
    # dispatch there is no single step the run is "at".
    current_step: int = Field(default=0, ge=0)
    artifacts: dict[str, ArtifactPointer] = Field(default_factory=dict)  # producer id -> pointer
    events: list[TaskEvent] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)

    def state_slice(self, subtask: Subtask) -> SubtaskContext:
        """Narrow the ledger to what one worker needs (§6: give each agent only its slice).

        The caller supplies the subtask rather than an id to look up, so this works
        during replanning, before a reshaped plan is committed to state.

        Args:
            subtask: the subtask about to be dispatched.

        Returns:
            A frozen context: the request, a copy of the subtask, the answered
            clarifications, and the pointers named in `subtask.inputs` — resolved to
            pointers, never to payloads.

        Raises:
            TaskFailure: an input names an artifact that does not exist, meaning the
                plan's dependency order is wrong. Failing here beats handing a worker a
                silently empty input and letting it invent the data.
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
            clarifications=list(self.clarifications),  # entries are frozen; the list is not shared
        )
