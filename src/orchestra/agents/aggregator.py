"""The synthesis pass: every completed artifact in, one `FinalReport` out.

**Previews, never payloads.** The one agent that looks behind a pointer, and only through
`ArtifactStore.preview` — a few hundred characters of prompt, not the file.

**Why a draft model.** A draft figure's `source` is a plain `str` because a model can emit
anything; the conversion to `FinalReport` is where it is checked against this run's
artifacts (§7). The chart is not in the draft — it is the ledger's, read back from the
visualization step's own artifact rather than asked for again.

**A synthesis failure never loses the report.** The artifacts are already paid for, so
every failure degrades to a ledger-only report rather than an empty stdout (§8, §10) — the
same path that serves a run with nothing completed. A refusal, or figures citing nothing
this run made, is a degraded report rather than a broken run (#8); an `OrchestraError` —
an outage, a lost artifact — also records its reason on `state.failure_reason` and fails
the run, remapping a provider error (exit 4) onto a task failure (exit 5). The trade buys
a report, and stderr is what tells the two apart.
"""

import asyncio

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.agents.structured import Rejected, parse_validated
from orchestra.agents.workers.visualization import VisualizationResult
from orchestra.artifacts import DEFAULT_PREVIEW_LIMIT, ArtifactStore
from orchestra.core.errors import OrchestraError
from orchestra.core.state import (
    AgentRole,
    ArtifactPointer,
    FinalReport,
    KeyFigure,
    Subtask,
    SubtaskStatus,
    TaskState,
)
from orchestra.prompts import AGGREGATOR_SYSTEM_PROMPT, STRUCTURED_REFORMAT_INSTRUCTION
from orchestra.providers.base import MessageRole, Provider, ProviderMessage

# How many artifacts may be previewed at once (§10). The reads share the process-wide
# thread pool, so the bound is stated here rather than left to the plan's size.
MAX_PREVIEW_READS = 4

_Completed = tuple[Subtask, ArtifactPointer]


class FigureDraft(BaseModel):
    """One key figure as the model writes it — `source` unvalidated by design."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, description="What the figure measures, in a few words.")
    value: str = Field(min_length=1, description="The figure itself, written as it should be read.")
    source: str = Field(
        min_length=1,
        description="The `artifact:` pointer of the artifact this figure was read from, "
        "copied exactly from the preview it came from.",
    )


class ReportDraft(BaseModel):
    """The report as the model writes it. Passed to the provider as `output_format`."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(
        min_length=1,
        description="Three to five sentences answering the user's request in their terms.",
    )
    key_figures: list[FigureDraft] = Field(
        default_factory=list,
        description="The numbers that answer the request, at most six, each sourced.",
    )


class Aggregator:
    """Writes the run's final report. One per run, built in `app.py` with its services."""

    def __init__(
        self,
        provider: Provider,
        store: ArtifactStore,
        *,
        preview_limit: int = DEFAULT_PREVIEW_LIMIT,
    ) -> None:
        """Store the injected services. Nothing is constructed here (§3.3).

        Args:
            preview_limit: characters of each artifact the model is shown.
        """
        self._provider = provider
        self._store = store
        self._preview_limit = preview_limit

    async def write_report(self, state: TaskState) -> FinalReport:
        """Synthesise the run's artifacts, record the report in `state`, and return it.

        `state.final_result` is always set on return, including when the model refused,
        when synthesis failed, and when nothing completed. An `OrchestraError` on either
        step also records its reason on `state.failure_reason`, marking the run failed.

        Raises:
            asyncio.CancelledError: propagated, never swallowed (§10). Nothing else.
        """
        completed = _completed(state)
        chart: ArtifactPointer | None = None
        chart_ascii: str | None = None
        report: FinalReport | None = None
        try:
            # From the visualization step's own receipt, never from the report draft: the
            # chart is a fact the run recorded, not one the model may claim.
            chart, chart_ascii = await self._chart_outputs(completed)
        except OrchestraError as exc:
            # Its own `try`: a chart nobody can read costs the chart, not the summary the
            # readable artifacts still support.
            _record_failure(state, f"The report's chart could not be read: {exc}")

        if completed:
            try:
                report = await self._synthesise(state, completed, chart, chart_ascii)
            except OrchestraError as exc:
                _record_failure(state, f"The report could not be synthesised: {exc}")

        if report is None:
            report = _ledger_report(state.user_request, completed, chart, chart_ascii)

        state.final_result = report
        return report

    async def _chart_outputs(
        self, completed: list[_Completed]
    ) -> tuple[ArtifactPointer | None, str | None]:
        """The chart to open and the chart to print, from the visualization step's receipt.

        Last rather than first: a plan that draws twice draws the summarising chart last.

        Returns `(None, None)` when no visualization ran, and when its artifact is not a
        `VisualizationResult` — `EchoWorker` still backs the role and writes plain text,
        which costs the report its chart, not the run (§8). A `TaskFailure` from the store
        is raised: `write_report` records it and writes the rest of the report anyway.
        """
        receipts = [
            pointer for subtask, pointer in completed if subtask.role is AgentRole.VISUALIZATION
        ]
        if not receipts:
            return None, None
        try:
            # `to_thread` because the store is blocking I/O; whole, not previewed, since
            # the fields wanted are the ones a preview may elide.
            payload = await asyncio.to_thread(self._store.get_text, receipts[-1])
            result = VisualizationResult.model_validate_json(payload)
        except (ValidationError, UnicodeDecodeError):
            return None, None
        return result.chart, result.ascii_chart

    async def _synthesise(
        self,
        state: TaskState,
        completed: list[_Completed],
        chart: ArtifactPointer | None,
        chart_ascii: str | None,
    ) -> FinalReport | None:
        """Ask the model for one report, retrying an unusable reply with the reason fed back.

        Returns:
            The report, or `None` when every attempt gave nothing usable — a refusal, a
            truncated reply, or figures citing no artifact of this run.
        """
        briefing = await self._briefing(state.user_request, completed)

        def validate(draft: ReportDraft) -> FinalReport:
            """Trust boundary (§7): the ledger decides which figures are real."""
            figures = state.backed_figures(_key_figures(draft))
            if draft.key_figures and not figures:
                # Every number it cited traces to nothing, so its summary is no better read.
                raise Rejected(
                    "None of those figures cite an artifact this run produced. Cite only the "
                    "`artifact:` pointers shown in the briefing, copied exactly, and drop any "
                    "figure you cannot source to one."
                )
            return FinalReport(
                executive_summary=draft.executive_summary,
                key_figures=figures,
                chart=chart,
                chart_ascii=chart_ascii,
            )

        report, _ = await parse_validated(
            provider=self._provider,
            system=AGGREGATOR_SYSTEM_PROMPT,
            messages=[ProviderMessage(role=MessageRole.USER, content=briefing)],
            output_format=ReportDraft,
            validate=validate,
            instruction=STRUCTURED_REFORMAT_INSTRUCTION,
        )
        return report

    async def _briefing(self, user_request: str, completed: list[_Completed]) -> str:
        """Build the user turn: the request, then one section per completed subtask.

        Formatting lives here, not in `prompts/` (§11), and both the request and the
        previews stay out of the system prompt — untrusted text.
        """
        # `to_thread` because the store is blocking I/O; bounded because §10 says so.
        reads = asyncio.Semaphore(MAX_PREVIEW_READS)

        async def read(pointer: ArtifactPointer) -> str:
            async with reads:
                return await asyncio.to_thread(
                    self._store.preview, pointer, limit=self._preview_limit
                )

        previews = await asyncio.gather(*(read(pointer) for _, pointer in completed))
        sections = [f"The user asked:\n{user_request}"]
        sections += [
            "\n".join(
                [
                    f"Subtask {subtask.id} ({subtask.role.value}) produced {pointer}",
                    f"Instruction: {subtask.instruction}",
                    "Preview:",
                    preview,
                ]
            )
            for (subtask, pointer), preview in zip(completed, previews, strict=True)
        ]
        return "\n\n".join(sections)


def _key_figures(draft: ReportDraft) -> list[KeyFigure]:
    """The draft's figures as ledger figures, dropping any whose `source` is not a pointer.

    A drop, not a rejection: a malformed pointer is as unbacked as an unknown one, and one
    bad citation must not cost the report the figures beside it.
    """
    figures = []
    for figure in draft.key_figures:
        try:
            figures.append(KeyFigure(label=figure.label, value=figure.value, source=figure.source))
        except ValidationError:
            continue
    return figures


def _record_failure(state: TaskState, reason: str) -> None:
    """Add a reason to the ledger, keeping any the engine already recorded."""
    state.failure_reason = f"{state.failure_reason}\n{reason}" if state.failure_reason else reason


def _completed(state: TaskState) -> list[_Completed]:
    """The subtasks that finished with something to show, in plan order."""
    if state.plan is None:
        return []
    return [
        (subtask, subtask.output_pointer)
        for subtask in state.plan.subtasks
        if subtask.status is SubtaskStatus.DONE and subtask.output_pointer is not None
    ]


def _ledger_report(
    user_request: str,
    completed: list[_Completed],
    chart: ArtifactPointer | None,
    chart_ascii: str | None,
) -> FinalReport:
    """Build a report from the ledger alone, no model involved.

    The degraded path: states what the run produced and stops. No key figures — reading a
    number out of an artifact is the model's job, and guessing one is invention.
    """
    if not completed:
        summary = f"No subtask produced a result, so this run has no answer to: {user_request}"
    else:
        summary = "\n".join(
            [
                "No synthesis was available for this run, so this report lists what it "
                "produced. Open the artifacts for the detail.",
                *(
                    f"- {subtask.id} ({subtask.role.value}) produced {pointer}: "
                    f"{subtask.instruction}"
                    for subtask, pointer in completed
                ),
            ]
        )
    return FinalReport(
        executive_summary=summary, key_figures=[], chart=chart, chart_ascii=chart_ascii
    )
