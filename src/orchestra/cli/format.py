"""Turning a finished run into the string stdout gets, separate from rendering (§3.3).

Returns a string and knows nothing about Rich, so both shapes are testable without a
console. What each `--output` mode *says* is decided here; `cli/app.py` picks the stream.

**The JSON document is a published contract, so it is not the ledger.** Dumping
`TaskState` would make its event log and bookkeeping a promise to whoever pipes us into
`jq`. The view models below restate only the contract's fields — §2.3 duplication across
a layer boundary, where independence outranks DRY.

**Nothing here raises**: the run that most needs printing is the one that went wrong, so
an absent plan or report is rendered as the fact it is.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from orchestra.core.state import (
    ARTIFACT_PREFIX,
    AgentRole,
    ArtifactPointer,
    SubtaskStatus,
    TaskState,
)

# Never print nothing: a silent command is indistinguishable from one that crashed.
NO_REPORT = "No report was produced for this run."


class OutputFormat(StrEnum):
    """The `--output` modes. A closed set, so an enum (§7); Typer reads the choices off it."""

    TEXT = "text"
    JSON = "json"


class RunStatus(StrEnum):
    """The run's verdict, as the JSON document states it: `TaskState.failed`, named."""

    COMPLETED = "completed"
    FAILED = "failed"


# Frozen: a document is built once, dumped, and discarded.
_VIEW_CONFIG = ConfigDict(extra="forbid", frozen=True)


class FigureView(BaseModel):
    """One sourced number in the JSON document."""

    model_config = _VIEW_CONFIG

    label: str
    value: str
    source: ArtifactPointer


class ReportView(BaseModel):
    """The final report in the JSON document."""

    model_config = _VIEW_CONFIG

    executive_summary: str
    key_figures: list[FigureView]
    chart: ArtifactPointer | None
    chart_ascii: str | None


class SubtaskView(BaseModel):
    """One step of the plan in the JSON document, with what it produced."""

    model_config = _VIEW_CONFIG

    id: str
    role: AgentRole
    status: SubtaskStatus
    artifact: ArtifactPointer | None


class ResultDocument(BaseModel):
    """The whole of `--output json`. Field order here is key order in the output."""

    model_config = _VIEW_CONFIG

    request: str
    status: RunStatus
    report: ReportView | None
    subtasks: list[SubtaskView]
    failure_reason: str | None
    # A string, so the document stays plain JSON. `report.chart` keeps its pointer: this
    # is what makes it resolvable, not a replacement for it.
    artifact_dir: str | None


def format_result(state: TaskState, *, output: OutputFormat, quiet: bool = False) -> str:
    """Render a finished run for stdout, without a trailing newline.

    `quiet` drops the step trace from the text shape — those lines are progress, which §5
    lets `--quiet` suppress, unlike the report. JSON does not vary with a flag.
    """
    if output is OutputFormat.JSON:
        return _document(state).model_dump_json(indent=2)
    return _text(state, quiet=quiet)


def _text(state: TaskState, *, quiet: bool) -> str:
    """The human shape: report, then trace, blank line between blocks.

    An empty block is omitted whole — "Key figures:" followed by silence reads as a bug.
    """
    report = state.final_result
    if report is None:
        blocks = [NO_REPORT]
    else:
        blocks = [report.executive_summary]
        if report.key_figures:
            blocks.append(
                "\n".join(
                    [
                        "Key figures:",
                        *(
                            f"  {figure.label}  {figure.value}  {figure.source}"
                            for figure in report.key_figures
                        ),
                    ]
                )
            )
        # The drawing first, then the pointer that opens the real one.
        if report.chart_ascii:
            blocks.append(report.chart_ascii)
        if report.chart is not None:
            # The path, not the pointer: the user has to be able to open it.
            blocks.append(f"Chart: {_artifact_path(report.chart, state.artifact_dir)}")

    steps = _steps(state)
    if steps and not quiet:
        blocks.append(steps)
    return "\n\n".join(blocks)


def _artifact_path(pointer: str, artifact_dir: Path | None) -> str:
    """Where `pointer`'s payload is, or the pointer itself when the run named no directory.

    String work on the directory rather than `ArtifactStore._resolve`: that one is I/O and
    raises, and this module promises neither (§2.3 — across a layer boundary).
    """
    if artifact_dir is None:
        return pointer
    return str(artifact_dir / pointer.removeprefix(ARTIFACT_PREFIX))


def _steps(state: TaskState) -> str:
    """Where the run's artifacts are, then one line per subtask: status, id, artifact.

    Fixed column widths, not fitted to the ids, so the block diffs cleanly between runs.
    The directory leads so a run that produced no chart is still findable.
    """
    lines = [] if state.artifact_dir is None else [f"Artifacts: {state.artifact_dir}"]
    if state.plan is not None:
        lines.append("Steps:")
        lines.extend(
            f"{subtask.status.value:<8} {subtask.id}  {subtask.output_pointer or '-'}"
            for subtask in state.plan.subtasks
        )
    return "\n".join(lines)


def _document(state: TaskState) -> ResultDocument:
    """Project the ledger onto the published contract."""
    report = state.final_result
    subtasks = state.plan.subtasks if state.plan is not None else []
    return ResultDocument(
        request=state.user_request,
        status=RunStatus.FAILED if state.failed else RunStatus.COMPLETED,
        report=None
        if report is None
        else ReportView(
            executive_summary=report.executive_summary,
            key_figures=[
                FigureView(label=figure.label, value=figure.value, source=figure.source)
                for figure in report.key_figures
            ],
            chart=report.chart,
            chart_ascii=report.chart_ascii,
        ),
        subtasks=[
            SubtaskView(
                id=subtask.id,
                role=subtask.role,
                status=subtask.status,
                artifact=subtask.output_pointer,
            )
            for subtask in subtasks
        ],
        failure_reason=state.failure_reason,
        artifact_dir=None if state.artifact_dir is None else str(state.artifact_dir),
    )
