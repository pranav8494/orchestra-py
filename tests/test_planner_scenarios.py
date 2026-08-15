"""Offline half of the #17 scenario suite (§12).

These pin down that a shape survives the planner: `_to_plan` invents no dependency edge,
drops no role, and reorders nothing. Whether a live model decomposes each prompt this way
is `test_planner_scenarios_live.py`'s question; a fake cannot answer it.

The shapes live in `scenarios.py`, so both suites assert the same contract.
"""

import pytest

from conftest import FakeProvider
from orchestra.agents.planner import Planner
from orchestra.core.state import AgentRole, TaskState
from scenarios import FAN_OUT, LINEAR, ROLE_OMISSION, SCENARIOS, Scenario, assert_plan_shape
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
    """The fan-out criterion stated directly: no edge in either direction between the
    retrievals. The engine parallelises on `depends_on` and nothing else."""
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
    """The assertion's own error path: a checker that passes everything would let all
    three scenarios go green against one fixed chain."""
    provider = FakeProvider(responses=[FAN_OUT.draft()])

    plan = await Planner(provider).create_plan(TaskState(user_request=FAN_OUT.prompt))

    with pytest.raises(AssertionError, match="expected 3 subtasks, got 4"):
        assert_plan_shape(plan, LINEAR.shape)
