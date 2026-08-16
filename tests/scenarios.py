"""The three scenarios that prove the planner is dynamic (#17).

A three-role pipeline can look dynamic while always emitting the same DAG. Each scenario
pairs a prompt with the shape it must produce:

| Scenario      | Required shape                                               |
|---------------|--------------------------------------------------------------|
| linear        | 3 steps, sequential                                          |
| fan_out       | 2 independent retrievals, then analytics, then visualization |
| role_omission | 2 steps, no visualization                                    |

Shapes are stated in roles and edges, never ids, so the same assertion holds for a canned
draft and for whatever a live model names its steps.

`fan_out` asks for two subjects held in two different places, because the planner is told
what the team can retrieve (#10) and reads that roster honestly. Its earlier prompt —
three quarters against the same quarters a year earlier — is two ranges of one CSV, so
planning it as one step is correct and the split stopped being reliable.
"""

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from orchestra.agents.planner import PlannerAction, PlannerDraft, SubtaskDraft
from orchestra.core.state import AgentRole, Plan, Subtask

# A count a plan must hit exactly, or a `range` of counts it may land anywhere inside.
# A range is for the one thing a scenario does not get to dictate: how finely the model
# divides a role's work. Splitting "compute our growth, then compare it" into two steps is
# the same judgement that keeps two ranges of one CSV in one step, and a live plan did
# exactly that. What the scenario *does* dictate — which roles appear, and which edges do
# and do not exist — stays exact.
type Count = int | range


def _holds(actual: int, expected: Count) -> bool:
    """Does `actual` satisfy an exact count or a permitted range?"""
    return actual == expected if isinstance(expected, int) else actual in expected


def _describe(expected: Count) -> str:
    """A count as failure-message text: `"3"`, or `"4 to 5"` for a range."""
    if isinstance(expected, int):
        return str(expected)
    return f"{_lowest(expected)} to {_highest(expected)}"


def _lowest(expected: Count) -> int:
    return expected if isinstance(expected, int) else expected.start


def _highest(expected: Count) -> int:
    return expected if isinstance(expected, int) else expected.stop - 1


@dataclass(frozen=True, slots=True)
class PlanShape:
    """What a plan must look like, in roles and edges.

    Attributes:
        steps: how many subtasks the plan must have.
        role_counts: subtasks per role. Every `AgentRole` must appear; an absent role is
            stated as `0`, which is the assertion role_omission turns on.
        precedes: `(earlier, later)` pairs; every `later` subtask must transitively depend
            on at least one `earlier` one. "At least one" rather than "all": where a role
            has several subtasks they are separate pieces of work, and a chart drawn from
            one of them is not missing the others.
        concurrent: a role whose subtasks must be pairwise independent, and which must
            reconverge — some later subtask depending on all of them. Without the second
            half, two branches that never meet would also pass. `None` where the role has
            one subtask.
    """

    steps: Count
    role_counts: Mapping[AgentRole, Count]
    precedes: tuple[tuple[AgentRole, AgentRole], ...]
    concurrent: AgentRole | None = None

    def __post_init__(self) -> None:
        """Reject an inconsistent scenario definition: counts that disagree with the step
        count would accept plans they should reject."""
        if set(self.role_counts) != set(AgentRole):
            raise ValueError("every role needs a count, including the roles that must be absent")
        for bound in (_lowest, _highest):
            total = sum(bound(count) for count in self.role_counts.values())
            if total != bound(self.steps):
                raise ValueError(f"role counts allow {total} subtasks, not {bound(self.steps)}")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One prompt, the shape it must produce, and a plausible model answer to it."""

    name: str
    prompt: str
    shape: PlanShape
    # A factory, not an instance: a shared draft is one a test can leave mutated.
    draft: Callable[[], PlannerDraft]


def linear_draft() -> PlannerDraft:
    """Fetch, then analyse, then chart — the answer the linear prompt should get."""
    return PlannerDraft(
        action=PlannerAction.PLAN,
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
        ],
    )


def fan_out_draft() -> PlannerDraft:
    """Our figures and the industry's, then a comparison, then a chart. The missing edge
    between the retrievals is what lets the engine run them at once."""
    return PlannerDraft(
        action=PlannerAction.PLAN,
        subtasks=[
            SubtaskDraft(
                id="fetch_our_growth",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load revenue for the last three quarters from the financials.",
            ),
            SubtaskDraft(
                id="fetch_industry_benchmarks",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Search for published growth benchmarks for comparable companies.",
            ),
            SubtaskDraft(
                id="compare_against_benchmarks",
                role=AgentRole.ANALYTICS,
                instruction="Compute our quarterly growth and place it against the benchmarks.",
                inputs=["fetch_our_growth", "fetch_industry_benchmarks"],
                depends_on=["fetch_our_growth", "fetch_industry_benchmarks"],
            ),
            SubtaskDraft(
                id="chart_growth_trend",
                role=AgentRole.VISUALIZATION,
                instruction="Plot the quarterly growth trend against the benchmark range.",
                inputs=["compare_against_benchmarks"],
                depends_on=["compare_against_benchmarks"],
            ),
        ],
    )


def role_omission_draft() -> PlannerDraft:
    """A written summary and nothing to draw: visualization is dropped, not reordered."""
    return PlannerDraft(
        action=PlannerAction.PLAN,
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
        ],
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
    prompt="Compare our last 3 quarters of revenue growth against industry benchmarks "
    "and chart the trend",
    shape=PlanShape(
        # Two retrievals and one chart, exactly; the comparison may be one step or two.
        steps=range(4, 6),
        role_counts={
            AgentRole.DATA_RETRIEVAL: 2,
            AgentRole.ANALYTICS: range(1, 3),
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
    """Assert `plan` matches `shape`, rendering the offending plan when it does not."""
    rendered = _render(plan)

    assert _holds(len(plan.subtasks), shape.steps), (
        f"expected {_describe(shape.steps)} subtasks, got {len(plan.subtasks)}:\n{rendered}"
    )

    for role, expected in shape.role_counts.items():
        actual = len(with_role(plan, role))
        assert _holds(actual, expected), (
            f"expected {_describe(expected)} {role} subtasks, got {actual}:\n{rendered}"
        )

    ancestors = _ancestors(plan)
    for earlier, later in shape.precedes:
        producers = {producer.id for producer in with_role(plan, earlier)}
        for consumer in with_role(plan, later):
            assert ancestors[consumer.id] & producers, (
                f"{consumer.id!r} ({later}) must run after a {earlier} subtask, directly or "
                f"transitively:\n{rendered}"
            )

    if shape.concurrent is not None:
        branches = with_role(plan, shape.concurrent)
        for first, second in itertools.combinations(branches, 2):
            assert second.id not in ancestors[first.id] and first.id not in ancestors[second.id], (
                f"{first.id!r} and {second.id!r} are both {shape.concurrent} and need nothing "
                f"from each other, so neither may depend on the other:\n{rendered}"
            )
        branch_ids = {branch.id for branch in branches}
        assert any(branch_ids <= ancestors[subtask.id] for subtask in plan.subtasks), (
            f"the {shape.concurrent} subtasks fan out and never meet: no subtask depends on "
            f"all of {sorted(branch_ids)}:\n{rendered}"
        )


def with_role(plan: Plan, role: AgentRole) -> list[Subtask]:
    """The plan's subtasks carrying `role`, in plan order."""
    return [subtask for subtask in plan.subtasks if subtask.role is role]


def _ancestors(plan: Plan) -> dict[str, set[str]]:
    """Every subtask id each subtask transitively depends on.

    `Plan` validates acyclicity on construction, so the walk needs no cycle guard.
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
    """The plan as failure-message text: one line per subtask, with its role and edges."""
    return "\n".join(
        f"  {subtask.id} [{subtask.role}] depends_on={subtask.depends_on}"
        for subtask in plan.subtasks
    )
