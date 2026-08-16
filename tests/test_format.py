"""Tests for the `text | json` switch (CONVENTIONS.md §5, §12).

No console and no CliRunner here: `format_result` returns a string, which is the whole
reason the switch is a module of its own (§3.3). What reaches which stream is
`test_cli.py`'s subject.

The JSON assertions go through `json.loads` rather than comparing text, because the
contract is the document — keys, types, and nulls — not pydantic's indentation.
"""

import json
from typing import Any

import pytest

from orchestra.cli.format import NO_REPORT, OutputFormat, format_result
from orchestra.core.state import (
    AgentRole,
    FinalReport,
    KeyFigure,
    Plan,
    Subtask,
    SubtaskStatus,
    TaskState,
)

REQUEST = "Summarize the last 3 quarters' financial trends and create a chart"
SUMMARY = "Revenue grew in each of the last three quarters."


def _state(*, report: FinalReport | None, failure_reason: str | None = None) -> TaskState:
    """A finished three-step run: two done, one failed, with `report` as its result."""
    plan = Plan(
        subtasks=[
            Subtask(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Load revenue."),
            Subtask(id="analyze", role=AgentRole.ANALYTICS, instruction="Compute growth."),
            Subtask(id="chart", role=AgentRole.VISUALIZATION, instruction="Plot the trend."),
        ]
    )
    for subtask, pointer in zip(plan.subtasks, ["artifact:fetch.txt", None, None], strict=True):
        subtask.status = SubtaskStatus.DONE if pointer else SubtaskStatus.FAILED
        subtask.output_pointer = pointer
    return TaskState(
        user_request=REQUEST,
        plan=plan,
        artifacts={"fetch": "artifact:fetch.txt"},
        final_result=report,
        failure_reason=failure_reason,
    )


def _report(*, figures: bool = True, chart: bool = True) -> FinalReport:
    return FinalReport(
        executive_summary=SUMMARY,
        key_figures=[KeyFigure(label="Q3 revenue", value="145", source="artifact:fetch.txt")]
        if figures
        else [],
        chart="artifact:trend.html" if chart else None,
    )


def _text(state: TaskState, *, quiet: bool = False) -> str:
    return format_result(state, output=OutputFormat.TEXT, quiet=quiet)


def _document(state: TaskState) -> dict[str, Any]:
    document = json.loads(format_result(state, output=OutputFormat.JSON))
    assert isinstance(document, dict)  # one JSON object, not a fragment
    return document


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def test_format_result_text_renders_summary_figures_chart_and_steps() -> None:
    rendered = _text(_state(report=_report()))

    assert rendered.startswith(SUMMARY)
    assert "Key figures:\n  Q3 revenue  145  artifact:fetch.txt" in rendered
    assert "Chart: artifact:trend.html" in rendered
    # The line format `test_cli.py` and every downstream grep depend on.
    assert "done     fetch  artifact:fetch.txt" in rendered
    assert "failed   analyze  -" in rendered
    assert not rendered.endswith("\n")  # the console adds the newline (§5)


def test_format_result_text_without_figures_or_chart_omits_those_blocks() -> None:
    """An empty block is dropped whole: a heading with nothing under it reads as a bug."""
    rendered = _text(_state(report=_report(figures=False, chart=False)))

    assert SUMMARY in rendered
    assert "Key figures:" not in rendered
    assert "Chart:" not in rendered
    assert "Steps:" in rendered


def test_format_result_text_quiet_drops_the_steps_but_keeps_the_report() -> None:
    """§5: `--quiet` suppresses progress, never the result. The trace is the progress."""
    rendered = _text(_state(report=_report()), quiet=True)

    assert rendered.startswith(SUMMARY)
    assert "Key figures:" in rendered
    assert "Chart: artifact:trend.html" in rendered
    assert "Steps:" not in rendered
    assert "artifact:fetch.txt" in rendered  # as a figure's source, not as a step line
    assert "done" not in rendered


def test_format_result_text_without_a_report_says_so_rather_than_printing_nothing() -> None:
    rendered = _text(_state(report=None))

    assert rendered.startswith(NO_REPORT)
    assert "Steps:" in rendered  # what did run is still worth printing


def test_format_result_text_without_a_plan_renders_the_report_alone() -> None:
    """The format layer must not be the thing that raises on a half-finished run."""
    state = TaskState(user_request=REQUEST, final_result=_report(chart=False))

    rendered = _text(state)

    assert rendered == f"{SUMMARY}\n\nKey figures:\n  Q3 revenue  145  artifact:fetch.txt"


def test_format_result_text_with_neither_plan_nor_report_renders_one_line() -> None:
    rendered = _text(TaskState(user_request=REQUEST))

    assert rendered == NO_REPORT


# --------------------------------------------------------------------------
# json — the published contract
# --------------------------------------------------------------------------


def test_format_result_json_has_the_documented_keys() -> None:
    document = _document(_state(report=_report()))

    assert set(document) == {"request", "status", "report", "subtasks", "failure_reason"}
    assert document["request"] == REQUEST
    assert document["status"] == "failed"  # one subtask failed
    assert document["failure_reason"] is None
    assert document["report"] == {
        "executive_summary": SUMMARY,
        "key_figures": [{"label": "Q3 revenue", "value": "145", "source": "artifact:fetch.txt"}],
        "chart": "artifact:trend.html",
    }
    # Roles and statuses go out as their `StrEnum` values, not as `AgentRole.ANALYTICS`.
    assert document["subtasks"][0] == {
        "id": "fetch",
        "role": "data_retrieval",
        "status": "done",
        "artifact": "artifact:fetch.txt",
    }
    assert document["subtasks"][1]["artifact"] is None


def test_format_result_json_carries_the_run_ending_reason() -> None:
    document = _document(_state(report=_report(), failure_reason="Step cap of 1 exceeded"))

    assert document["status"] == "failed"
    assert document["failure_reason"] == "Step cap of 1 exceeded"


def test_format_result_json_status_is_completed_when_nothing_failed() -> None:
    state = _state(report=_report())
    assert state.plan is not None
    for subtask in state.plan.subtasks:
        subtask.status = SubtaskStatus.DONE

    assert _document(state)["status"] == "completed"


def test_format_result_json_omits_the_ledger_bookkeeping() -> None:
    """Regression guard on the contract: the event log and `current_step` are ours, not
    the caller's, and dumping `TaskState` would publish both."""
    state = _state(report=_report())
    state.current_step = 3

    document = _document(state)

    assert "events" not in document
    assert "current_step" not in document
    assert "artifacts" not in document


@pytest.mark.parametrize(
    "state",
    [
        TaskState(user_request=REQUEST),
        _state(report=None),
        _state(report=_report(figures=False, chart=False)),
    ],
    ids=["no_plan_no_report", "no_report", "bare_report"],
)
def test_format_result_json_stays_parseable_on_a_half_finished_run(state: TaskState) -> None:
    document = _document(state)

    # The contract's shape does not vary with how far the run got: a script reading
    # `.report` finds the key holding null, never a key that is absent.
    assert set(document) == {"request", "status", "report", "subtasks", "failure_reason"}
    assert document["request"] == REQUEST


def test_format_result_json_ignores_quiet() -> None:
    """The document is a contract; a display flag must not reshape it."""
    state = _state(report=_report())

    assert format_result(state, output=OutputFormat.JSON, quiet=True) == format_result(
        state, output=OutputFormat.JSON
    )
