"""Live re-verification of the run's report (#8) — deselected by default (§12).

```bash
uv run pytest -m live -k aggregation    # needs ANTHROPIC_API_KEY; costs one whole run
```

`test_app.py::test_run_once_reports_a_computed_figure_and_its_chart_in_both_output_shapes`
is the fake-provider twin; this proves the same contract against the real model, where the
figures are whatever it chose to state. Config is read at import, before
`conftest._isolated_env` cuts the environment off, and via `load_config()` rather than
`os.environ` (§6).
"""

import json
import re
from pathlib import Path

import pytest

from orchestra.app import run_once
from orchestra.artifacts import ArtifactStore
from orchestra.cli.format import OutputFormat, format_result
from orchestra.config import Config, load_config
from orchestra.core.errors import ConfigError
from orchestra.core.state import AgentRole, SubtaskStatus, artifact_path
from scenarios import LINEAR, assert_plan_shape, with_role

try:
    CONFIG: Config | None = load_config()
except ConfigError:
    CONFIG = None

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(CONFIG is None, reason="the live report run needs ANTHROPIC_API_KEY"),
]

# A model writes "$1.15M" for 1,153,000, so asserting the value as a substring would be
# flaky. Both sides are reduced to significant digits instead — separators, currency and
# scale suffixes drop out, and the digits themselves still have to come from the artifact.
# Grouped form first, and only in threes: a looser comma rule reads a CSV row as one
# number. A row the model itself printed, "Q2,576,441", still reads as one; the cost is a
# false failure on a paid run, not a figure let through.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?")

# Below this a run is too short to tell a reading from a coincidence — two digits
# prefix-match something in any artifact holding thirty numbers — so it is demanded whole
# instead. The price is that "$1.2M" fails against 1,153,000 where "$1.15M" passes: at two
# digits, rounding and accident are the same evidence, and this asserts against invention.
_MIN_SIGNIFICANT = 3


@pytest.mark.asyncio
async def test_the_report_cites_this_run_s_own_numbers_and_chart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#8's re-verify criterion against the real model: every figure is sourced to an
    artifact this run minted, its number is in that artifact, the chart pointer opens a
    file on disk, and `--output json` carries all of it.

    `run_once` releases the provider in its own `finally`, so there is no second provider
    to `aclosing` here as in the planner scenarios.
    """
    _export_settings(monkeypatch, tmp_path)

    state = await run_once(LINEAR.prompt)

    report = state.final_result
    assert report is not None, "the run produced no report"
    assert state.artifact_dir is not None
    assert report.executive_summary.strip()
    assert report.key_figures, f"the report states no number:\n{report.executive_summary}"

    store = ArtifactStore(state.artifact_dir)
    produced = set(state.artifacts.values())
    for figure in report.key_figures:
        # The anti-hallucination contract: a pointer of this run, and the number really in it.
        assert figure.source in produced, (
            f"{figure.label!r} cites {figure.source}, which this run never produced"
        )
        payload = store.get_text(figure.source)
        assert _traces_to(figure.value, payload), (
            f"{figure.label!r} states {figure.value!r}, which is in no number in "
            f"{figure.source}:\n{payload}"
        )

    # Before the chart guard, so a plan that omitted visualization still proves this half.
    document = json.loads(format_result(state, output=OutputFormat.JSON))
    assert document["report"]["key_figures"] == [
        {"label": figure.label, "value": figure.value, "source": figure.source}
        for figure in report.key_figures
    ]
    assert document["report"]["chart_ascii"] == report.chart_ascii

    # This prompt asks for a chart, so a plan without one is a planner defect, not a
    # variation to skip past — and the last step is the one the aggregator read.
    assert state.plan is not None
    assert_plan_shape(state.plan, LINEAR.shape)
    step = with_role(state.plan, AgentRole.VISUALIZATION)[-1]
    assert step.status is SubtaskStatus.DONE, f"{step.id!r} ended {step.status}, so it drew nothing"
    assert report.chart is not None, "the visualization step ran but the report names no chart"
    # `artifact_path`, not `store.path_for`: the pure composer, which reports the missing
    # file as a failed assertion rather than raising before the assertion is reached.
    assert "<html" in artifact_path(state.artifact_dir, report.chart).read_text()
    assert report.chart_ascii, "nothing for the terminal to print"


def _export_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Put the import-time config back in the environment for `run_once`'s own
    `load_config()`, which `_isolated_env` has just stripped.

    `TAVILY_API_KEY` stays unset: retrieval then reads the bundled corpus, so the run costs
    one provider's worth and no third-party call (§12). The run bounds are left at their
    defaults too — a `.env` that lowered them to cap cost would otherwise make this flaky.
    """
    assert CONFIG is not None  # guaranteed by the skipif; narrows the type for mypy
    monkeypatch.setenv("ANTHROPIC_API_KEY", CONFIG.anthropic_api_key.get_secret_value())
    monkeypatch.setenv("ANTHROPIC_MODEL", CONFIG.anthropic_model)
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", str(CONFIG.anthropic_max_tokens))
    monkeypatch.setenv("DATA_DIR", str(CONFIG.data_dir))
    # Under `tmp_path`, never the operator's `~/.orchestra`.
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))


def _traces_to(value: str, payload: str) -> bool:
    """Does `payload` hold every number `value` states?

    Every, not any: a figure pairing one real number with one invented one is exactly the
    hallucination this asserts against. Short runs are checked only where the figure states
    nothing longer, so "3 quarters" beside a real total does not fail the run — and where
    they are checked, a small count still meets a quarter label, which is the residual.
    """
    stated, found = _digit_runs(value), _digit_runs(payload)
    significant = [run for run in stated if len(run) >= _MIN_SIGNIFICANT]
    if not significant:
        return bool(stated) and all(run in found for run in stated)
    return all(any(_reads_as(item, run) for item in found) for run in significant)


def _digit_runs(text: str) -> list[str]:
    """Every number in `text` as its mantissa: "1,234.50" and "123450" both -> "12345".

    Scale goes with the separators, which is what lets "$5.76M" meet 5,760,000 — and the
    price is that it also meets 5.76, so this proves the digits, not the magnitude.
    """
    runs = (re.sub(r"\D", "", match.group()).strip("0") for match in _NUMBER.finditer(text))
    # A figure of zero states a number; without this it would state none and fail as if it
    # had cited nothing.
    return [run or "0" for run in runs]


def _reads_as(found: str, stated: str) -> bool:
    """Is `stated` how `found` reads at `stated`'s precision?"""
    if len(found) < len(stated):
        return False
    head, rest = found[: len(stated)], found[len(stated) :]
    # Second branch: 1,153,000 reported as "$1.2M" is rounded, not invented.
    return head == stated or (rest[:1] >= "5" and str(int(head) + 1).zfill(len(stated)) == stated)
