"""Live re-verification of the run's report (#8) — deselected by default (§12).

```bash
uv run pytest -m live -k aggregation    # needs ANTHROPIC_API_KEY; costs one whole run
```

`test_app.py::test_run_once_reports_a_computed_figure_and_its_chart_in_both_output_shapes`
is the fake-provider twin; here the figures are whatever the real model chose to state.
Config is read at import, before `conftest._isolated_env` cuts the environment off, and via
`load_config()` rather than `os.environ` (§6).
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

# A model writes "$1.2M" for 1,153,000, so a substring assertion would be flaky: both sides
# are read as numbers and compared at the precision the figure was written to. The leading
# guard drops a digit following a letter, so the "3" of "2024Q3" stays a label.
_NUMBER = re.compile(
    r"(?<![A-Za-z\d])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(%|bn|[KMB])?", re.IGNORECASE
)
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9}

# Points against a fraction, or the reverse — the only rescaling allowed. Without it 23.4%
# would not meet 0.2343; with anything looser "$5.76B" would meet 5,760,000.
_FACTORS = (1.0, 100.0, 0.01)


@pytest.mark.asyncio
async def test_the_report_cites_this_run_s_own_numbers_and_chart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#8's re-verify criterion against the real model: every figure is sourced to an
    artifact this run minted, its number is in that artifact, the chart pointer opens a
    file on disk, and `--output json` carries all of it.

    `run_once` releases the provider in its own `finally`, so nothing to `aclosing` here.
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
    # variation to skip past. The last visualization step is the one the aggregator read.
    assert state.plan is not None
    assert_plan_shape(state.plan, LINEAR.shape)
    step = with_role(state.plan, AgentRole.VISUALIZATION)[-1]
    assert step.status is SubtaskStatus.DONE, f"{step.id!r} ended {step.status}, so it drew nothing"
    assert report.chart is not None, "the visualization step ran but the report names no chart"
    # `artifact_path`, not `store.path_for`: the store raises on a missing file, so that
    # assertion could only ever raise, never fail.
    assert "<html" in artifact_path(state.artifact_dir, report.chart).read_text()
    assert report.chart_ascii, "nothing for the terminal to print"


def _export_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Put the import-time config back for `run_once`'s own `load_config()`, which
    `_isolated_env` has just stripped.

    `TAVILY_API_KEY` stays unset so retrieval reads the bundled corpus — one provider's
    worth, no third-party call (§12). Run bounds stay at their defaults: a `.env` that
    lowered them to cap cost would make this flaky.
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

    Every, not any: a figure pairing one real number with an invented one is the
    hallucination this asserts against.
    """
    stated = _numbers(value)
    found = [number for number, _ in _numbers(payload)]
    return bool(stated) and all(
        any(_reads_as(item, number, tolerance) for item in found) for number, tolerance in stated
    )


def _numbers(text: str) -> list[tuple[float, float]]:
    """Every number in `text` with the tolerance its own precision allows.

    "$1.2M" is 1,200,000 give or take 50,000, so it meets 1,153,000; "$1.15M" holds to
    5,000 and does not. Writing a figure more precisely claims more.
    """
    numbers = []
    for match in _NUMBER.finditer(text):
        digits, suffix = match.group(1), (match.group(2) or "").lower()
        scale = _SCALE.get(suffix, 1.0)
        decimals = len(digits.partition(".")[2])
        numbers.append((float(digits.replace(",", "")) * scale, 0.5 * 10**-decimals * scale))
    return numbers


def _reads_as(found: float, stated: float, tolerance: float) -> bool:
    """Is `stated` how `found` reads, to `tolerance`?"""
    return any(abs(found * factor - stated) <= tolerance for factor in _FACTORS)
