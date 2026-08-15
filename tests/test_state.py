"""Tests for the shared ledger and its slice (CONVENTIONS.md §6, §12).

`Plan` validation is exercised as a trust boundary, not as pydantic coverage: every
case here is one the planner (#3) can hand us from model output.
"""

import pytest
from pydantic import ValidationError

from orchestra.core.errors import ExitCode, TaskFailure
from orchestra.core.state import (
    AgentRole,
    Clarification,
    EventKind,
    Plan,
    Subtask,
    SubtaskStatus,
    TaskEvent,
    TaskState,
)

REQUEST = "Summarize the last 3 quarters' financial trends and create a chart"


def _fetch() -> Subtask:
    return Subtask(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Pull Q1-Q3 revenue")


def _analyse() -> Subtask:
    return Subtask(
        id="analyse",
        role=AgentRole.ANALYTICS,
        instruction="Compute the quarter-over-quarter trend",
        inputs=["revenue"],
        depends_on=["fetch"],
    )


def test_task_state_round_trip_preserves_plan_artifacts_and_log() -> None:
    """State is serialised into prompts and back out; nothing may be lost in the trip."""
    state = TaskState(
        user_request=REQUEST,
        plan=Plan(subtasks=[_fetch(), _analyse()]),
        current_step=1,
        artifacts={"revenue": "artifact:revenue.csv"},
        events=[TaskEvent(kind=EventKind.PLAN_CREATED, message="2 subtasks")],
        clarifications=[Clarification(question="Which currency?", answer="USD")],
    )

    restored = TaskState.model_validate_json(state.model_dump_json())

    assert restored == state


def test_task_state_round_trip_preserves_subtask_status_transitions() -> None:
    """Status lives on the subtask; the engine's transitions have to survive the trip."""
    plan = Plan(subtasks=[_fetch(), _analyse()])
    plan.subtasks[0].status = SubtaskStatus.DONE
    plan.subtasks[0].output_pointer = "artifact:revenue.csv"
    plan.subtasks[1].status = SubtaskStatus.RUNNING
    state = TaskState(user_request=REQUEST, plan=plan)

    restored = TaskState.model_validate_json(state.model_dump_json())

    assert restored.plan is not None
    assert [subtask.status for subtask in restored.plan.subtasks] == [
        SubtaskStatus.DONE,
        SubtaskStatus.RUNNING,
    ]
    assert restored.plan.subtasks[0].output_pointer == "artifact:revenue.csv"


def test_subtask_rejects_unknown_status() -> None:
    """`SubtaskStatus` is a closed set (§7) — a model inventing "in_progress" must not land."""
    with pytest.raises(ValidationError):
        Subtask(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Pull", status="in_progress")


@pytest.mark.parametrize(
    "blob",
    [
        "quarter,revenue\nQ1,120\n",  # the payload itself, inlined
        "/var/folders/revenue.csv",  # a bare path
        "artifact:../escaped.csv",  # a pointer that would traverse
    ],
)
def test_task_state_rejects_anything_that_is_not_a_pointer(blob: str) -> None:
    """Enforced by the type, not just by the store: state is what reaches the prompt."""
    with pytest.raises(ValidationError):
        TaskState(user_request=REQUEST, artifacts={"revenue": blob})


def test_subtask_rejects_an_output_pointer_that_is_not_a_pointer() -> None:
    with pytest.raises(ValidationError):
        Subtask(
            id="fetch",
            role=AgentRole.DATA_RETRIEVAL,
            instruction="Pull",
            output_pointer="/tmp/revenue.csv",
        )


def test_task_state_rejects_an_unexpected_field() -> None:
    """A key silently dropped on reload is data loss (§7)."""
    with pytest.raises(ValidationError):
        TaskState.model_validate({"user_request": REQUEST, "artifactss": {}})


def test_task_state_rejects_a_negative_step() -> None:
    with pytest.raises(ValidationError):
        TaskState(user_request=REQUEST, current_step=-1)


def test_plan_with_duplicate_subtask_ids_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate subtask ids"):
        Plan(subtasks=[_fetch(), _fetch()])


def test_plan_with_unknown_dependency_is_rejected() -> None:
    orphan = Subtask(
        id="chart",
        role=AgentRole.VISUALIZATION,
        instruction="Plot it",
        depends_on=["nonexistent"],
    )

    with pytest.raises(ValidationError, match="unknown subtasks"):
        Plan(subtasks=[orphan])


def test_plan_with_dependency_cycle_is_rejected() -> None:
    """A cycle would deadlock the execution engine; it dies here instead (§7)."""
    first = Subtask(id="a", role=AgentRole.ANALYTICS, instruction="Depends on b", depends_on=["b"])
    second = Subtask(id="b", role=AgentRole.ANALYTICS, instruction="Depends on a", depends_on=["a"])

    with pytest.raises(ValidationError, match="dependency cycle"):
        Plan(subtasks=[first, second])


def test_empty_plan_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Plan(subtasks=[])


def test_state_slice_gives_the_worker_only_its_declared_inputs() -> None:
    """Selective context isolation: the other agents' artifacts must not travel."""
    state = TaskState(
        user_request=REQUEST,
        plan=Plan(subtasks=[_fetch(), _analyse()]),
        artifacts={
            "revenue": "artifact:revenue.csv",
            "unrelated_headcount": "artifact:headcount.csv",
        },
        events=[TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id="fetch")],
    )

    context = state.state_slice(_analyse())

    assert context.inputs == {"revenue": "artifact:revenue.csv"}
    assert context.user_request == REQUEST
    assert context.subtask.id == "analyse"
    # The slice is what gets serialised into the worker's prompt, so absence is the assertion.
    serialised = context.model_dump_json()
    assert "headcount" not in serialised
    assert "subtask_started" not in serialised


def test_state_slice_does_not_hand_the_worker_a_write_handle_on_the_ledger() -> None:
    """Pydantic reuses nested instances, so without the copy a worker writing to its own
    context reaches through into the plan the engine owns."""
    plan = Plan(subtasks=[_fetch()])
    state = TaskState(user_request=REQUEST, plan=plan)

    context = state.state_slice(plan.subtasks[0])
    context.subtask.status = SubtaskStatus.DONE

    assert plan.subtasks[0].status is SubtaskStatus.PENDING


def test_state_slice_result_is_frozen() -> None:
    """mypy catches this statically — the ignore is the proof; this covers runtime."""
    context = TaskState(user_request=REQUEST).state_slice(_fetch())

    with pytest.raises(ValidationError):
        context.user_request = "something else"  # type: ignore[misc]


def test_state_slice_carries_answered_clarifications() -> None:
    """A worker needs what the user disambiguated, or it re-guesses what was already asked."""
    state = TaskState(
        user_request=REQUEST,
        clarifications=[Clarification(question="Which currency?", answer="USD")],
    )

    context = state.state_slice(_fetch())

    assert context.clarifications[0].answer == "USD"


def test_state_slice_with_missing_input_raises_task_failure() -> None:
    """Wrong dependency order — better than letting a worker invent the missing data."""
    state = TaskState(user_request=REQUEST)  # "revenue" was never produced

    with pytest.raises(TaskFailure) as exc_info:
        state.state_slice(_analyse())

    assert "revenue" in str(exc_info.value)
    assert exc_info.value.exit_code == ExitCode.TASK_FAILURE
