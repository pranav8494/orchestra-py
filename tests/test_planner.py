"""Tests for the orchestrator (CONVENTIONS.md §12).

Everything runs against `FakeProvider`. The assertions are about what the planner does
with model output — the conversation it sends, the validation it applies, what it writes
to the ledger — never about the model's judgement, which is not ours to test.
"""

import asyncio

import pytest

from conftest import FakeProvider
from orchestra.agents.planner import PlanDraft, Planner, SubtaskDraft
from orchestra.core.errors import ExitCode, ProviderError
from orchestra.core.state import AgentRole, EventKind, SubtaskStatus, TaskState
from orchestra.prompts import PLANNER_SYSTEM_PROMPT

REQUEST = "Summarize the last 3 quarters financial trends and create a chart"


def _financial_plan() -> PlanDraft:
    """A plausible answer to `REQUEST`: fetch, then analyse, then chart and write up."""
    return PlanDraft(
        subtasks=[
            SubtaskDraft(
                id="fetch_quarterly_financials",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load revenue and margin for the last three quarters.",
            ),
            SubtaskDraft(
                id="analyse_trends",
                role=AgentRole.ANALYTICS,
                instruction="Compute quarter-over-quarter growth and describe the trend.",
                inputs=["fetch_quarterly_financials"],
                depends_on=["fetch_quarterly_financials"],
            ),
            SubtaskDraft(
                id="chart_trends",
                role=AgentRole.VISUALIZATION,
                instruction="Plot the quarterly revenue trend as a line chart.",
                inputs=["analyse_trends"],
                depends_on=["analyse_trends"],
            ),
        ]
    )


def _broken_plan() -> PlanDraft:
    """Shaped correctly, unrunnable: `depends_on` names a step that is not in the plan.
    Structured output cannot rule this out, which is why the reformat retry exists."""
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
async def test_create_plan_decomposes_the_financial_request_into_an_ordered_dag() -> None:
    provider = FakeProvider(responses=[_financial_plan()])
    state = TaskState(user_request=REQUEST)

    plan = await Planner(provider).create_plan(state)

    assert 3 <= len(plan.subtasks) <= 4
    assert [subtask.role for subtask in plan.subtasks] == [
        AgentRole.DATA_RETRIEVAL,
        AgentRole.ANALYTICS,
        AgentRole.VISUALIZATION,
    ]
    # Ordering is the contract the engine parallelises against: retrieval starts
    # immediately, and nothing plots before the numbers exist.
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
async def test_create_plan_retries_once_when_no_structured_output_comes_back() -> None:
    """`parsed_output is None` — a refusal, or a reply truncated before the JSON closed."""
    provider = FakeProvider(responses=[None, _financial_plan()])

    plan = await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    assert len(plan.subtasks) == 3
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_create_plan_raises_after_a_second_failure_without_retrying_again() -> None:
    provider = FakeProvider(responses=[_broken_plan(), _broken_plan()])
    state = TaskState(user_request=REQUEST)

    with pytest.raises(ProviderError) as exc_info:
        await Planner(provider).create_plan(state)

    assert exc_info.value.exit_code == ExitCode.PROVIDER
    assert len(provider.calls) == 2
    # A failed plan is not a plan: the ledger must not be left half-written.
    assert state.plan is None
    assert state.events == []


@pytest.mark.asyncio
async def test_create_plan_propagates_a_provider_failure() -> None:
    provider = FakeProvider(responses=[ProviderError("401 authentication_error")])

    with pytest.raises(ProviderError, match="authentication_error"):
        await Planner(provider).create_plan(TaskState(user_request=REQUEST))

    # A transport failure is not invalid output; retrying it here would double every
    # outage. Retry policy for that is #9's.
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

    # Bounded: a planner that swallowed the cancellation would sit on the blocker
    # forever, and a suite that hangs tells you nothing.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert state.plan is None


def test_planner_prompt_names_every_agent_role() -> None:
    """The prompt lists the roles as literal text; this is what stops it drifting from
    `AgentRole` when a fourth worker lands."""
    for role in AgentRole:
        assert role.value in PLANNER_SYSTEM_PROMPT
