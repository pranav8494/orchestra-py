"""Live half of the #17 scenario suite — deselected by default (§12).

```bash
uv run pytest -m live          # needs ANTHROPIC_API_KEY; costs three model calls
```

Only a real model can show that three differently shaped requests produce three
differently shaped plans. A failure here is a prompt problem: the shapes come from the
research doc's table, and `prompts/planner.py` has to satisfy them.

Config is read at import, before `conftest._isolated_env` cuts the environment off, and
through `load_config()` rather than `os.environ` (§6).
"""

from contextlib import aclosing

import pytest

from orchestra.agents.planner import Planner
from orchestra.config import Config, load_config
from orchestra.core.errors import ConfigError
from orchestra.core.state import TaskState
from orchestra.providers.base import create_provider
from scenarios import SCENARIOS, Scenario, assert_plan_shape
from scenarios import scenario_id as _id

try:
    CONFIG: Config | None = load_config()
except ConfigError:
    CONFIG = None

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(CONFIG is None, reason="live scenarios need ANTHROPIC_API_KEY"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_id)
async def test_the_planner_shapes_a_real_plan_to_the_request(scenario: Scenario) -> None:
    assert CONFIG is not None  # guaranteed by the skipif; narrows the type for mypy

    # Unclosed, the SDK's pooled sockets outlive the test and `filterwarnings = ["error"]`
    # fails teardown whatever the assertions did.
    async with aclosing(
        create_provider(api_key=CONFIG.anthropic_api_key, model=CONFIG.anthropic_model)
    ) as provider:
        plan = await Planner(provider).create_plan(TaskState(user_request=scenario.prompt))

    assert_plan_shape(plan, scenario.shape)
