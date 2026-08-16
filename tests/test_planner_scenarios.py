"""Offline half of the #17 scenario suite.

Pins down that a shape survives the planner: `_to_plan` invents no edge, drops no role,
reorders nothing. Whether a live model decomposes this way is
`test_planner_scenarios_live.py`'s question.
"""

import pytest

from conftest import FakeProvider
from orchestra.agents.planner import Planner
from orchestra.core.state import AgentRole, Plan, Subtask, TaskState
from scenarios import (
    FAN_OUT,
    LINEAR,
    ROLE_OMISSION,
    SCENARIOS,
    PlanShape,
    Scenario,
    assert_plan_shape,
)
from scenarios import scenario_id as _id


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_id)
async def test_create_plan_keeps_the_scenario_shape(scenario: Scenario) -> None:
    """Step count, role assignment, and every dependency edge, per scenario."""
    provider = FakeProvider(responses=[scenario.draft()])

    plan = await Planner(provider).create_plan(TaskState(user_request=scenario.prompt))

    assert_plan_shape(plan, scenario.shape)


@pytest.mark.asyncio
async def test_create_plan_leaves_the_two_fan_out_retrievals_unlinked() -> None:
    """The engine parallelises on `depends_on` and nothing else."""
    provider = FakeProvider(responses=[FAN_OUT.draft()])

    plan = await Planner(provider).create_plan(TaskState(user_request=FAN_OUT.prompt))

    retrievals = [s for s in plan.subtasks if s.role is AgentRole.DATA_RETRIEVAL]
    assert len(retrievals) == 2
    first, second = retrievals
    assert first.depends_on == []
    assert second.depends_on == []
    assert first.id not in second.inputs
    assert second.id not in first.inputs


@pytest.mark.asyncio
async def test_create_plan_omits_visualization_when_the_request_asks_for_no_chart() -> None:
    """Omission, not reordering: the role is absent from the plan entirely."""
    provider = FakeProvider(responses=[ROLE_OMISSION.draft()])

    plan = await Planner(provider).create_plan(TaskState(user_request=ROLE_OMISSION.prompt))

    assert AgentRole.VISUALIZATION not in {subtask.role for subtask in plan.subtasks}
    assert len(plan.subtasks) == 2


@pytest.mark.asyncio
async def test_assert_plan_shape_rejects_a_plan_of_the_wrong_shape() -> None:
    """Guards the checker itself: one that passes everything would let all three scenarios
    go green against a single fixed chain."""
    provider = FakeProvider(responses=[FAN_OUT.draft()])

    plan = await Planner(provider).create_plan(TaskState(user_request=FAN_OUT.prompt))

    with pytest.raises(AssertionError, match="expected 3 subtasks, got 4"):
        assert_plan_shape(plan, LINEAR.shape)


def test_assert_plan_shape_rejects_two_branches_that_never_meet() -> None:
    """The fan-out shape allows the comparison to be one step or two, so the count alone no
    longer proves the branches rejoin. Two retrievals feeding nothing in common is not a
    fan-out; it is two plans."""
    plan = Plan(
        subtasks=[
            Subtask(id="fetch_ours", role=AgentRole.DATA_RETRIEVAL, instruction="Load revenue"),
            Subtask(
                id="fetch_theirs", role=AgentRole.DATA_RETRIEVAL, instruction="Find benchmarks"
            ),
            Subtask(
                id="summarise",
                role=AgentRole.ANALYTICS,
                instruction="Describe our growth",
                inputs=["fetch_ours"],
                depends_on=["fetch_ours"],
            ),
            Subtask(
                id="chart",
                role=AgentRole.VISUALIZATION,
                instruction="Plot our growth",
                inputs=["summarise"],
                depends_on=["summarise"],
            ),
        ]
    )

    with pytest.raises(AssertionError, match="fan out and never meet"):
        assert_plan_shape(plan, FAN_OUT.shape)


def test_assert_plan_shape_rejects_a_count_outside_a_permitted_range() -> None:
    """The other direction of the widened counts: the fan-out shape admits a second
    analytics step, not a third. A range that accepted anything would retire the
    assertion."""
    plan = Plan(
        subtasks=[
            *(
                Subtask(id=f"fetch_{n}", role=AgentRole.DATA_RETRIEVAL, instruction="Load")
                for n in range(2)
            ),
            *(
                Subtask(
                    id=f"crunch_{n}",
                    role=AgentRole.ANALYTICS,
                    instruction="Compute",
                    inputs=["fetch_0", "fetch_1"],
                    depends_on=["fetch_0", "fetch_1"],
                )
                for n in range(3)
            ),
            Subtask(
                id="chart",
                role=AgentRole.VISUALIZATION,
                instruction="Plot",
                inputs=["crunch_0"],
                depends_on=["crunch_0"],
            ),
        ]
    )

    with pytest.raises(AssertionError, match="expected 4 to 5 subtasks, got 6"):
        assert_plan_shape(plan, FAN_OUT.shape)


def test_plan_shape_rejects_role_counts_that_cannot_reach_the_step_count() -> None:
    """A scenario whose counts disagree with its step count accepts plans it should reject,
    so the definition itself fails rather than the plans it is checked against."""
    with pytest.raises(ValueError, match="role counts allow"):
        PlanShape(
            steps=range(4, 6),
            role_counts={
                AgentRole.DATA_RETRIEVAL: 2,
                AgentRole.ANALYTICS: 1,
                AgentRole.VISUALIZATION: 1,
            },
            precedes=(),
        )
