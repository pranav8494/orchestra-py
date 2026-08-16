"""Live half of the #17 scenario suite, plus #10's clarification — deselected by default (§12).

```bash
uv run pytest -m live          # needs ANTHROPIC_API_KEY; costs six model calls
```

A failure here is a prompt problem: `prompts/planner.py` has to satisfy the shapes. Config
is read at import, before `conftest._isolated_env` cuts the environment off, and via
`load_config()` rather than `os.environ` (§6).

The planner is built as `app.py` builds it, roster and all. Without it the model is told
this run has no data sources, which is not the configuration any of these assertions are
about.
"""

from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

import pytest

from orchestra.agents.planner import Planner
from orchestra.agents.toolsets import data_retrieval_tools, retrievable_data
from orchestra.config import Config, load_config
from orchestra.core.errors import ConfigError
from orchestra.core.question import MAX_QUESTIONS, Question
from orchestra.core.state import AgentRole, TaskState
from orchestra.providers.base import Provider, create_provider
from orchestra.tools.question import AskUserTool
from scenarios import LINEAR, SCENARIOS, Scenario, assert_plan_shape
from scenarios import scenario_id as _id

try:
    CONFIG: Config | None = load_config()
except ConfigError:
    CONFIG = None

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(CONFIG is None, reason="live scenarios need ANTHROPIC_API_KEY"),
]

# What a request has to leave unstated to be worth a question at all (#10).
AMBIGUOUS = "Make a chart of performance"

# Measures no tool holds. A live run offered "stock price" as a metric, took the answer,
# and planned three steps with nowhere to get the data — the roster exists to stop that,
# so its absence from the questions is the assertion.
UNAVAILABLE = ("stock", "share price", "headcount", "traffic")


class AnswersAnything:
    """An `Asker` that answers whatever it is handed, and keeps the questions.

    Not `conftest.ScriptedAsker`: a live model's questions are its own, so there is no
    answer list to write in advance.
    """

    def __init__(self) -> None:
        self.asked: list[Question] = []

    async def ask(self, question: Question) -> str:
        self.asked.append(question)
        if question.choices:
            return question.choices[0]
        return "yes" if question.kind.value == "yes_no" else "the last 3 quarters"


@asynccontextmanager
async def _planner(asker: AnswersAnything | None = None) -> AsyncIterator[Planner]:
    """The planner as `build_orchestra` wires it, against the real provider.

    Unclosed, the SDK's pooled sockets outlive the test and `filterwarnings = ["error"]`
    fails teardown regardless of the assertions.
    """
    assert CONFIG is not None  # guaranteed by the skipif; narrows the type for mypy
    provider: Provider = create_provider(
        api_key=CONFIG.anthropic_api_key,
        model=CONFIG.anthropic_model,
        max_tokens=CONFIG.anthropic_max_tokens,
    )
    tools = data_retrieval_tools(CONFIG.data_dir, search_api_key=CONFIG.tavily_api_key)
    async with aclosing(provider):
        yield Planner(
            provider,
            ask_tool=None if asker is None else AskUserTool(asker),
            retrievable_data=retrievable_data(tools),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_id)
async def test_the_planner_shapes_a_real_plan_to_the_request(scenario: Scenario) -> None:
    async with _planner() as planner:
        plan = await planner.create_plan(TaskState(user_request=scenario.prompt))

    assert_plan_shape(plan, scenario.shape)


@pytest.mark.asyncio
async def test_the_planner_asks_only_about_data_the_team_holds() -> None:
    """The graded round, live. What it asks is the model's judgement; that it asks, stays
    inside the cap, and never offers a measure nothing can fetch is the prompt's."""
    asker = AnswersAnything()
    async with _planner(asker) as planner:
        state = TaskState(user_request=AMBIGUOUS)
        plan = await planner.create_plan(state)

    assert 1 <= len(asker.asked) <= MAX_QUESTIONS
    offered = " ".join(
        f"{question.text} {question.description} {' '.join(question.choices)}".lower()
        for question in asker.asked
    )
    for measure in UNAVAILABLE:
        assert measure not in offered, f"asked about {measure!r}, which no tool supplies"
    # The answers were used, not collected and dropped.
    assert len(state.clarifications) == len(asker.asked)
    assert any(subtask.role is AgentRole.DATA_RETRIEVAL for subtask in plan.subtasks)


@pytest.mark.asyncio
async def test_the_planner_asks_nothing_it_could_answer_itself() -> None:
    """The other half of the criterion: a request naming its subject and period is planned,
    not interrogated. One question here is a regression in the ambiguity check."""
    asker = AnswersAnything()
    async with _planner(asker) as planner:
        plan = await planner.create_plan(TaskState(user_request=LINEAR.prompt))

    assert asker.asked == []
    assert_plan_shape(plan, LINEAR.shape)
