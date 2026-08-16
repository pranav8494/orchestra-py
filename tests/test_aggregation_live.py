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
from orchestra.core.state import AgentRole, Subtask, SubtaskStatus, TaskState, artifact_path
from scenarios import LINEAR

try:
    CONFIG: Config | None = load_config()
except ConfigError:
    CONFIG = None

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(CONFIG is None, reason="the live report run needs ANTHROPIC_API_KEY"),
]

# A model writes "$1.2M" for 1,153,000, so asserting the value as a substring would be
# flaky. Both sides are reduced to digit runs instead — separators, currency and scale
# suffixes drop out, and the digits themselves still have to come from the artifact.
# Grouped form first, and only in threes: a looser comma rule reads a CSV row as one number.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?")


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

    step = _visualization(state)
    if step is None:
        pytest.skip("this plan omitted visualization, so the run has no chart to check")
    assert step.status is SubtaskStatus.DONE, f"{step.id!r} ended {step.status}, so it drew nothing"
    assert report.chart is not None, "the visualization step ran but the report names no chart"
    # `artifact_path`, not `store.path_for`: the pure composer, which reports the missing
    # file as a failed assertion rather than raising before the assertion is reached.
    assert artifact_path(state.artifact_dir, report.chart).is_file()
    assert report.chart_ascii, "nothing for the terminal to print"


def _export_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Put the import-time config back in the environment for `run_once`'s own
    `load_config()`, which `_isolated_env` has just stripped.

    `TAVILY_API_KEY` stays unset: retrieval then reads the bundled corpus, so the run costs
    one provider's worth and no third-party call (§12).
    """
    assert CONFIG is not None  # guaranteed by the skipif; narrows the type for mypy
    monkeypatch.setenv("ANTHROPIC_API_KEY", CONFIG.anthropic_api_key.get_secret_value())
    monkeypatch.setenv("ANTHROPIC_MODEL", CONFIG.anthropic_model)
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", str(CONFIG.anthropic_max_tokens))
    monkeypatch.setenv("DATA_DIR", str(CONFIG.data_dir))
    # Under `tmp_path`, never the operator's `~/.orchestra`.
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))


def _visualization(state: TaskState) -> Subtask | None:
    """The plan's visualization step, or `None` where it planned none — a legitimate plan
    for some requests, so the chart assertions skip rather than fail on it."""
    subtasks = [] if state.plan is None else state.plan.subtasks
    return next((task for task in subtasks if task.role is AgentRole.VISUALIZATION), None)


def _traces_to(value: str, payload: str) -> bool:
    """Is a number `value` states one that `payload` holds?"""
    stated = _digit_runs(value)
    return bool(stated) and any(
        _reads_as(found, run) for run in stated for found in _digit_runs(payload)
    )


def _digit_runs(text: str) -> list[str]:
    """Every number in `text` as its significant digits: "1,234.56" -> "123456"."""
    runs = (re.sub(r"\D", "", match.group()).lstrip("0") for match in _NUMBER.finditer(text))
    return [run for run in runs if run]


def _reads_as(found: str, stated: str) -> bool:
    """Is `stated` how `found` reads at `stated`'s precision?"""
    if len(found) < len(stated):
        # Trailing zeros only: "1.20" is 1.2 written longer, "1200" is not 12.
        return stated.startswith(found) and set(stated[len(found) :]) <= {"0"}
    head, rest = found[: len(stated)], found[len(stated) :]
    # Second branch: 1,153,000 reported as "$1.2M" is rounded, not invented.
    return head == stated or (rest[:1] >= "5" and str(int(head) + 1).zfill(len(stated)) == stated)
