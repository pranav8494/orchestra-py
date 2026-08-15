"""The three scenarios that prove the planner is dynamic (#17).

A three-role pipeline can look dynamic while always emitting the same DAG. Each scenario
below pairs a prompt with the plan shape that prompt must produce, taken verbatim from
the research doc's "Proving the Planner Is Dynamic" table:

| Scenario      | Required shape                                                    |
|---------------|-------------------------------------------------------------------|
| linear        | 3 steps, sequential                                               |
| fan_out       | 2 independent retrievals, then analytics, then visualization      |
| role_omission | 2 steps, no visualization                                         |

A shape is stated in roles and edges, never in ids: the same assertion has to hold for a
canned draft and for whatever a live model decides to name its steps.

Two suites consume this module. `test_planner_scenarios.py` runs it offline against
`FakeProvider` and answers "does the shape survive draft -> `Plan` conversion?".
`test_planner_scenarios_live.py` runs it against the real model and answers "does the
planner actually decompose this way?" — the question a fake cannot answer and the one
§12 will not let run by default. Counts are exact rather than tolerant, so a live
failure is a signal about the prompt, which is the whole reason to run it.
"""

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from orchestra.agents.planner import PlanDraft, SubtaskDraft
from orchestra.core.state import AgentRole, Plan, Subtask


@dataclass(frozen=True, slots=True)
class PlanShape:
    """What a plan must look like, in roles and edges — an internal value object (§7).

    Attributes:
        steps: how many subtasks the plan must have.
        role_counts: how many subtasks carry each role. Every `AgentRole` must appear,
            so an omitted role is stated as `0` rather than left implicit — that is the
            assertion the role-omission scenario turns on, and it must not go missing
            when a fourth role lands.
        precedes: `(earlier, later)` pairs. Every subtask of `later` must transitively
            depend on every subtask of `earlier`.
        concurrent: a role whose subtasks must be pairwise independent — neither
            reachable from the other, so the engine may run them at once. `None` where
            the role has a single subtask and the assertion would be vacuous.
    """

    steps: int
    role_counts: Mapping[AgentRole, int]
    precedes: tuple[tuple[AgentRole, AgentRole], ...]
    concurrent: AgentRole | None = None

    def __post_init__(self) -> None:
        """Check the scenario definition, not a plan.

        A shape whose role counts disagree with its step count would pass some plans it
        should reject, and the test would look green while asserting nothing.

        Raises:
            ValueError: the shape is not internally consistent.
        """
        if set(self.role_counts) != set(AgentRole):
            raise ValueError("every role needs a count, including the roles that must be absent")
        if sum(self.role_counts.values()) != self.steps:
            raise ValueError(
                f"role counts sum to {sum(self.role_counts.values())}, not {self.steps}"
            )


@dataclass(frozen=True, slots=True)
class Scenario:
    """One prompt, the shape it must produce, and a plausible model answer to it."""

    name: str
    prompt: str
    shape: PlanShape
    # A factory, not an instance: the offline suite hands this to `FakeProvider`, and a
    # draft shared between tests is a draft one test can leave modified for the next.
    draft: Callable[[], PlanDraft]


def linear_draft() -> PlanDraft:
    """Fetch, then analyse, then chart — the answer the linear prompt should get."""
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


def fan_out_draft() -> PlanDraft:
    """Two retrievals that need nothing from each other, then one comparison, then a
    chart. Neither retrieval names the other in `depends_on`: that missing edge is what
    lets the engine run both at once, and asserting on it is this scenario's point."""
    return PlanDraft(
        subtasks=[
            SubtaskDraft(
                id="fetch_recent_quarters",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load revenue for the last three quarters.",
            ),
            SubtaskDraft(
                id="fetch_prior_year_quarters",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load revenue for the same three quarters of last year.",
            ),
            SubtaskDraft(
                id="compare_quarters",
                role=AgentRole.ANALYTICS,
                instruction="Compute the year-over-year change for each of the three quarters.",
                inputs=["fetch_recent_quarters", "fetch_prior_year_quarters"],
                depends_on=["fetch_recent_quarters", "fetch_prior_year_quarters"],
            ),
            SubtaskDraft(
                id="chart_comparison",
                role=AgentRole.VISUALIZATION,
                instruction="Plot both years' quarterly revenue as a grouped bar chart.",
                inputs=["compare_quarters"],
                depends_on=["compare_quarters"],
            ),
        ]
    )


def role_omission_draft() -> PlanDraft:
    """A written summary and nothing to draw. The absent visualization subtask is the
    assertion: the planner drops a role it does not need, rather than reordering it."""
    return PlanDraft(
        subtasks=[
            SubtaskDraft(
                id="fetch_revenue_history",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load the revenue figures for the reporting period.",
            ),
            SubtaskDraft(
                id="summarise_revenue_trend",
                role=AgentRole.ANALYTICS,
                instruction="Describe the revenue trend in one paragraph.",
                inputs=["fetch_revenue_history"],
                depends_on=["fetch_revenue_history"],
            ),
        ]
    )


LINEAR = Scenario(
    name="linear",
    prompt="Summarize the last 3 quarters financial trends and create a chart",
    shape=PlanShape(
        steps=3,
        role_counts={
            AgentRole.DATA_RETRIEVAL: 1,
            AgentRole.ANALYTICS: 1,
            AgentRole.VISUALIZATION: 1,
        },
        precedes=(
            (AgentRole.DATA_RETRIEVAL, AgentRole.ANALYTICS),
            (AgentRole.ANALYTICS, AgentRole.VISUALIZATION),
        ),
    ),
    draft=linear_draft,
)

FAN_OUT = Scenario(
    name="fan_out",
    prompt="Compare the last 3 quarters against the same quarters last year and chart both",
    shape=PlanShape(
        steps=4,
        role_counts={
            AgentRole.DATA_RETRIEVAL: 2,
            AgentRole.ANALYTICS: 1,
            AgentRole.VISUALIZATION: 1,
        },
        precedes=(
            (AgentRole.DATA_RETRIEVAL, AgentRole.ANALYTICS),
            (AgentRole.ANALYTICS, AgentRole.VISUALIZATION),
        ),
        concurrent=AgentRole.DATA_RETRIEVAL,
    ),
    draft=fan_out_draft,
)

ROLE_OMISSION = Scenario(
    name="role_omission",
    prompt="Summarize the revenue trend in one paragraph",
    shape=PlanShape(
        steps=2,
        role_counts={
            AgentRole.DATA_RETRIEVAL: 1,
            AgentRole.ANALYTICS: 1,
            AgentRole.VISUALIZATION: 0,
        },
        precedes=((AgentRole.DATA_RETRIEVAL, AgentRole.ANALYTICS),),
    ),
    draft=role_omission_draft,
)

SCENARIOS = (LINEAR, FAN_OUT, ROLE_OMISSION)


def scenario_id(scenario: Scenario) -> str:
    """Name a parametrised case after its scenario rather than by index."""
    return scenario.name


def assert_plan_shape(plan: Plan, shape: PlanShape) -> None:
    """Assert `plan` matches `shape`, naming the offending plan when it does not.

    Args:
        plan: the plan the planner produced.
        shape: the shape the scenario's prompt must produce.

    Raises:
        AssertionError: the plan has the wrong step count, the wrong roles, a missing
            ordering edge, or a dependency between subtasks that must run concurrently.
    """
    rendered = _render(plan)

    assert len(plan.subtasks) == shape.steps, (
        f"expected {shape.steps} subtasks, got {len(plan.subtasks)}:\n{rendered}"
    )

    for role, expected in shape.role_counts.items():
        actual = len(_with_role(plan, role))
        assert actual == expected, f"expected {expected} {role} subtasks, got {actual}:\n{rendered}"

    ancestors = _ancestors(plan)
    for earlier, later in shape.precedes:
        for consumer in _with_role(plan, later):
            for producer in _with_role(plan, earlier):
                assert producer.id in ancestors[consumer.id], (
                    f"{consumer.id!r} ({later}) must run after {producer.id!r} ({earlier}), "
                    f"directly or transitively:\n{rendered}"
                )

    if shape.concurrent is not None:
        for first, second in itertools.combinations(_with_role(plan, shape.concurrent), 2):
            assert second.id not in ancestors[first.id] and first.id not in ancestors[second.id], (
                f"{first.id!r} and {second.id!r} are both {shape.concurrent} and need nothing "
                f"from each other, so neither may depend on the other:\n{rendered}"
            )


def _with_role(plan: Plan, role: AgentRole) -> list[Subtask]:
    """The plan's subtasks carrying `role`, in plan order."""
    return [subtask for subtask in plan.subtasks if subtask.role is role]


def _ancestors(plan: Plan) -> dict[str, set[str]]:
    """Every subtask id each subtask transitively depends on.

    `Plan` validates acyclicity on construction, so this walk needs no cycle guard —
    an unvalidated graph would recurse forever instead of failing an assertion.
    """
    by_id = {subtask.id: subtask for subtask in plan.subtasks}
    resolved: dict[str, set[str]] = {}

    def walk(subtask_id: str) -> set[str]:
        cached = resolved.get(subtask_id)
        if cached is not None:
            return cached
        found: set[str] = set()
        for parent in by_id[subtask_id].depends_on:
            found.add(parent)
            found |= walk(parent)
        resolved[subtask_id] = found
        return found

    return {subtask.id: walk(subtask.id) for subtask in plan.subtasks}


def _render(plan: Plan) -> str:
    """The plan as failure-message text: one line per subtask, with its role and edges.

    A shape assertion that fails prints "expected 4, got 3" and nothing else is a
    failure you have to re-run under a debugger to understand.
    """
    return "\n".join(
        f"  {subtask.id} [{subtask.role}] depends_on={subtask.depends_on}"
        for subtask in plan.subtasks
    )
