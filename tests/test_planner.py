"""Tests for the orchestrator.

Everything runs against `FakeProvider`. The assertions are about what the planner does
with model output — the conversation it sends, the validation it applies, what it writes
to the ledger — never about the model's judgement.
"""

import asyncio

import pytest

from conftest import FakeProvider
from orchestra.agents.planner import PlanDraft, Planner, SubtaskDraft
from orchestra.core.errors import ExitCode, ProviderError, TaskFailure
from orchestra.core.state import AgentRole, EventKind, SubtaskStatus, TaskState
from orchestra.prompts import PLANNER_SYSTEM_PROMPT
from scenarios import LINEAR

# Reusing the linear scenario as the good-plan fixture keeps the two suites from
# disagreeing about what a valid plan looks like (§2).
REQUEST = LINEAR.prompt
_financial_plan = LINEAR.draft


def _broken_plan() -> PlanDraft:
    """Shaped correctly, unrunnable: `depends_on` names a step outside the plan. Structured
    output cannot rule this out, which is why the reformat retry exists."""
    return PlanDraft(
        subtasks=[
            SubtaskDraft(
                id="chart_trends",
                role=AgentRole.VISUALIZATION,
                instruction="Plot the quarterly revenue trend.",
                depends_on=["analyse_trends"],
            )
        ]
    )


@pytest.mark.asyncio
async def test_create_plan_preserves_roles_and_dependency_ordering_from_the_draft() -> None:
    """Conversion loses no role and no edge; what a live model produces is
    `test_planner_scenarios_live.py`'s question."""
    provider = FakeProvider(responses=[_financial_plan()])
    state = TaskState(user_request=REQUEST)

    plan = await Planner(provider).create_plan(state)

    assert 3 <= len(plan.subtasks) <= 4
    assert [subtask.role for subtask in plan.subtasks] == [
        AgentRole.DATA_RETRIEVAL,
        AgentRole.ANALYTICS,
        AgentRole.VISUALIZATION,
    ]
    # Ordering is what the engine parallelises against.
    by_id = {subtask.id: subtask for subtask in plan.subtasks}
    assert by_id["fetch_quarterly_financials"].depends_on == []
    assert by_id["analyse_trends"].depends_on == ["fetch_quarterly_financials"]
    assert by_id["chart_trends"].depends_on == ["analyse_trends"]


@pytest.mark.asyncio
async def test_create_plan_leaves_engine_owned_fields_at_their_defaults() -> None:
    """The model fills in the draft; status and output pointers are the engine's."""
    provider = FakeProvider(responses=[_financial_plan()])

    plan = await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    assert all(subtask.status is SubtaskStatus.PENDING for subtask in plan.subtasks)
    assert all(subtask.output_pointer is None for subtask in plan.subtasks)


@pytest.mark.asyncio
async def test_create_plan_writes_the_plan_and_a_plan_created_event_to_state() -> None:
    provider = FakeProvider(responses=[_financial_plan()])
    state = TaskState(user_request=REQUEST)

    plan = await Planner(provider).create_plan(state)

    assert state.plan is plan
    assert [event.kind for event in state.events] == [EventKind.PLAN_CREATED]


@pytest.mark.asyncio
async def test_create_plan_sends_the_request_as_a_user_message_not_in_the_prompt() -> None:
    """§11: untrusted input goes in the user turn, never spliced into the instructions."""
    provider = FakeProvider(responses=[_financial_plan()])

    await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    call = provider.calls[0]
    assert call.system == PLANNER_SYSTEM_PROMPT
    assert REQUEST not in call.system
    assert [message.content for message in call.messages] == [REQUEST]
    assert call.output_format is PlanDraft


@pytest.mark.asyncio
async def test_create_plan_retries_once_when_the_plan_fails_validation() -> None:
    provider = FakeProvider(responses=[_broken_plan(), _financial_plan()])
    state = TaskState(user_request=REQUEST)

    plan = await Planner(provider).create_plan(state)

    assert len(plan.subtasks) == 3
    assert len(provider.calls) == 2
    # The retry has to carry the rejection, or the model re-sends the same plan.
    retry_messages = provider.calls[1].messages
    assert len(retry_messages) == 2
    assert "analyse_trends" in retry_messages[1].content


@pytest.mark.asyncio
async def test_create_plan_rejects_an_input_the_step_does_not_depend_on() -> None:
    """A data edge with no ordering edge is a race: the engine may start the consumer
    before the producer wrote anything. `Plan` checks `depends_on` only, so the planner
    owns this one."""
    unordered = PlanDraft(
        subtasks=[
            SubtaskDraft(
                id="fetch_quarterly_financials",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load revenue for the last three quarters.",
            ),
            SubtaskDraft(
                id="analyse_trends",
                role=AgentRole.ANALYTICS,
                instruction="Compute quarter-over-quarter growth.",
                inputs=["fetch_quarterly_financials"],  # consumed, but not depended on
            ),
        ]
    )
    provider = FakeProvider(responses=[unordered, _financial_plan()])
    state = TaskState(user_request=REQUEST)

    plan = await Planner(provider).create_plan(state)

    assert len(provider.calls) == 2
    assert "does not depend on them" in provider.calls[1].messages[1].content
    assert plan is state.plan


@pytest.mark.asyncio
async def test_create_plan_rejects_an_input_naming_a_step_outside_the_plan() -> None:
    ghost = PlanDraft(
        subtasks=[
            SubtaskDraft(
                id="chart_trends",
                role=AgentRole.VISUALIZATION,
                instruction="Plot the quarterly revenue trend.",
                inputs=["analyse_trends"],
                depends_on=["analyse_trends"],
            )
        ]
    )
    provider = FakeProvider(responses=[ghost, _financial_plan()])

    plan = await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    assert len(plan.subtasks) == 3
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_create_plan_retries_once_when_no_structured_output_comes_back() -> None:
    """`parsed_output is None` — a refusal, or a reply truncated before the JSON closed."""
    provider = FakeProvider(responses=[None, _financial_plan()])

    plan = await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    assert len(plan.subtasks) == 3
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_create_plan_accepts_a_third_attempt_after_two_rejections() -> None:
    """Two extra calls, each carrying the reason the last was rejected."""
    provider = FakeProvider(responses=[_broken_plan(), None, _financial_plan()])

    plan = await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    assert len(plan.subtasks) == 3
    assert len(provider.calls) == 3
    assert len(provider.calls[2].messages) == 3  # request, then one turn per rejection


@pytest.mark.asyncio
async def test_create_plan_raises_after_the_last_attempt_without_retrying_again() -> None:
    provider = FakeProvider(responses=[_broken_plan(), _broken_plan(), _broken_plan()])
    state = TaskState(user_request=REQUEST)

    # Exit 5, not 4: the provider answered every time; this run just has no plan.
    with pytest.raises(TaskFailure) as exc_info:
        await Planner(provider).create_plan(state)

    assert exc_info.value.exit_code == ExitCode.TASK_FAILURE
    assert len(provider.calls) == 3
    # A failed plan is not a plan: the ledger must not be left half-written.
    assert state.plan is None
    assert state.events == []


@pytest.mark.asyncio
async def test_create_plan_propagates_a_provider_failure() -> None:
    provider = FakeProvider(responses=[ProviderError("401 authentication_error")])

    with pytest.raises(ProviderError, match="authentication_error"):
        await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    # A transport failure is not invalid output; retrying here would double every
    # outage. That retry policy is #9's.
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_create_plan_is_cancellable() -> None:
    """§10: a run the user cannot stop is a defect, so cancellation must propagate."""
    provider = FakeProvider(responses=[_financial_plan()], blocker=asyncio.Event())
    state = TaskState(user_request=REQUEST)

    task = asyncio.create_task(Planner(provider).create_plan(state))
    await asyncio.sleep(0)  # let the task reach the provider call
    assert len(provider.calls) == 1  # in flight, blocked inside the provider
    task.cancel()

    # Bounded: a planner that swallowed the cancellation would hang on the blocker.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert state.plan is None


def test_planner_prompt_names_every_agent_role() -> None:
    """The prompt lists roles as literal text; this stops it drifting from `AgentRole`."""
    for role in AgentRole:
        assert role.value in PLANNER_SYSTEM_PROMPT
