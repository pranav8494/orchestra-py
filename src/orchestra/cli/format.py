"""Turning a finished run into the string stdout gets, separate from rendering (§3.3).

Formatting and rendering are split deliberately: this module returns a string and knows
nothing about Rich or terminals, so both shapes are testable without a console. `cli/app.py`
picks the stream (§5); everything about *what* each `--output` mode says is here.

**The JSON document is a published contract, so it is not the ledger.** `TaskState` carries
the event log, `current_step`, and whatever the engine needs next quarter; dumping it would
turn every internal field into a promise to whoever pipes us into `jq`, and every ledger
change into a breaking one. The view models below restate only the fields the contract
covers. They mirror `core.state` field for field on purpose — §2.3's case for duplication
across a layer boundary, where independence outranks DRY. When the two must differ, that
divergence is the point.

**Nothing here raises.** `app.py` always writes a report before the CLI formats, but the run
that most needs printing is the one that went wrong, so an absent plan or an absent report
is rendered as the fact it is rather than treated as impossible.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from orchestra.core.state import AgentRole, ArtifactPointer, SubtaskStatus, TaskState

# The line stdout gets when there is nothing to report — never an empty document, because
# a command that printed nothing at all is indistinguishable from one that crashed.
NO_REPORT = "No report was produced for this run."


class OutputFormat(StrEnum):
    """The `--output` modes. An enum over a closed set of strings (§7), which is also what
    Typer reads the choices and the default out of."""

    TEXT = "text"
    JSON = "json"


class RunStatus(StrEnum):
    """The run's verdict, as the JSON document states it: `TaskState.failed`, named."""

    COMPLETED = "completed"
    FAILED = "failed"


# Frozen, like the report they mirror: a document is built once, dumped, and discarded.
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


def format_result(state: TaskState, *, output: OutputFormat, quiet: bool = False) -> str:
    """Render a finished run for stdout.

    Args:
        state: the run's ledger, after the aggregator has written its report.
        output: which of the two documented shapes to produce.
        quiet: drop the per-subtask trace from the text shape. The report itself is never
            dropped: §5 lets `--quiet` suppress progress, never the result or the exit
            code, and the step lines *are* progress — the trace of how the run got here,
            kept in the transcript once it has finished. The report is the result.
            Ignored for JSON, whose shape is a contract and does not vary with a flag.

    Returns:
        The document, without a trailing newline.
    """
    if output is OutputFormat.JSON:
        return _document(state).model_dump_json(indent=2)
    return _text(state, quiet=quiet)


def _text(state: TaskState, *, quiet: bool) -> str:
    """The human shape: the report, then the trace, blank line between blocks.

    A block with nothing in it is omitted whole rather than printed as a heading with
    nothing under it — "Key figures:" followed by silence reads as a bug.
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
        if report.chart is not None:
            blocks.append(f"Chart: {report.chart}")

    steps = _steps(state)
    if steps and not quiet:
        blocks.append(steps)
    return "\n\n".join(blocks)


def _steps(state: TaskState) -> str:
    """One line per subtask: status, id, and the artifact it produced.

    The run's progress record. Column widths are fixed rather than fitted to the ids so
    the block stays greppable and diffable between runs.
    """
    if state.plan is None:
        return ""
    return "\n".join(
        [
            "Steps:",
            *(
                f"{subtask.status.value:<8} {subtask.id}  {subtask.output_pointer or '-'}"
                for subtask in state.plan.subtasks
            ),
        ]
    )


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
    )
