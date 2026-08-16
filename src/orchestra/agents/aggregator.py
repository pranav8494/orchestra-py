"""The synthesis pass: every completed artifact in, one `FinalReport` out.

**Previews, never payloads.** The one agent that looks behind a pointer, and it looks
through `ArtifactStore.preview` — a chart's HTML costs a few hundred characters of
prompt, not the file.

**Why a draft model.** `ReportDraft` is what the model fills in, `FinalReport` what the
ledger keeps. A draft figure's `source` is a plain `str` because a model can emit
anything; converting between the two is where it is checked against this run's artifacts
(§7). The chart is not in the draft — the ledger already knows it.

**Why one call and no retry.** The artifacts are already paid for, so a refusal or a set
of figures citing nothing real degrades to a ledger-only report rather than raising
(§8, §10). The same path serves a run with nothing completed.
"""

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from orchestra.artifacts import DEFAULT_PREVIEW_LIMIT, ArtifactStore
from orchestra.core.state import (
    AgentRole,
    ArtifactPointer,
    FinalReport,
    KeyFigure,
    Subtask,
    SubtaskStatus,
    TaskState,
)
from orchestra.prompts import AGGREGATOR_SYSTEM_PROMPT
from orchestra.providers.base import MessageRole, Provider, ProviderMessage

# How many artifacts may be previewed at once (§10). The reads share the process-wide
# thread pool, so the bound is stated here rather than left to the plan's size.
MAX_PREVIEW_READS = 4

# One completed subtask and the artifact it produced, paired at the filter.
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
            provider: the model provider to synthesise with.
            store: the run's artifact store, used only to preview payloads.
            preview_limit: characters of each artifact the model is shown.
        """
        self._provider = provider
        self._store = store
        self._preview_limit = preview_limit

    async def write_report(self, state: TaskState) -> FinalReport:
        """Synthesise the run's artifacts, record the report in `state`, and return it.

        Args:
            state: the run's ledger, after execution. `state.final_result` is always set
                on return, including when the model refused and when nothing completed.

        Returns:
            The report, the same object as `state.final_result`.

        Raises:
            TaskFailure: an artifact the ledger claims to hold is gone from the store, so
                the run has lost data (§8). Raised by the store, passed through here.
            ProviderError: the provider failed; a report is not worth a retried outage.
            asyncio.CancelledError: propagated, never swallowed (§10).
        """
        completed = _completed(state)
        # From the ledger, not the model: the chart is a fact the run already recorded.
        chart = _chart_pointer(completed)

        report = await self._synthesise(state, completed, chart) if completed else None
        if report is None:
            report = _ledger_report(state.user_request, completed, chart)

        state.final_result = report
        return report

    async def _synthesise(
        self, state: TaskState, completed: list[_Completed], chart: ArtifactPointer | None
    ) -> FinalReport | None:
        """Ask the model for one report and validate what comes back.

        Returns:
            The report, or `None` when the model gave nothing usable — a refusal, a
            truncated reply, or figures citing no artifact of this run. All three mean
            the same to the caller: fall back to the ledger.
        """
        briefing = await self._briefing(state.user_request, completed)
        draft = await self._provider.parse_structured(
            system=AGGREGATOR_SYSTEM_PROMPT,
            messages=[ProviderMessage(role=MessageRole.USER, content=briefing)],
            output_format=ReportDraft,
        )
        if draft is None:
            return None

        # Validation at the trust boundary (§7), not the guardrail framework #9 will add:
        # a figure survives only if its source is an artifact this run produced. The set
        # comes from the ledger, so survivors are valid pointers by construction.
        produced = set(state.artifacts.values())
        figures = [
            KeyFigure(label=figure.label, value=figure.value, source=figure.source)
            for figure in draft.key_figures
            if figure.source in produced
        ]
        if draft.key_figures and not figures:
            # Every number it cited traces to nothing, so its summary is no better read.
            return None

        return FinalReport(
            executive_summary=draft.executive_summary,
            key_figures=figures,
            chart=chart,
        )

    async def _briefing(self, user_request: str, completed: list[_Completed]) -> str:
        """Build the user turn: the request, then one section per completed subtask.

        Formatting lives here, not in `prompts/` (§11), and both the request and the
        previews stay out of the system prompt — untrusted text.
        """
        # `to_thread` because the store is blocking I/O, gathered because the reads are
        # independent, bounded because §10 wants the bound stated.
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


def _completed(state: TaskState) -> list[_Completed]:
    """The subtasks that finished with something to show, in plan order.

    A `DONE` subtask with no pointer produced nothing to preview.
    """
    if state.plan is None:
        return []
    return [
        (subtask, subtask.output_pointer)
        for subtask in state.plan.subtasks
        if subtask.status is SubtaskStatus.DONE and subtask.output_pointer is not None
    ]


def _chart_pointer(completed: list[_Completed]) -> ArtifactPointer | None:
    """The chart the report points at: the last completed visualization, or nothing.

    Last rather than first: a plan that draws twice draws the summarising chart last.
    """
    charts = [pointer for subtask, pointer in completed if subtask.role is AgentRole.VISUALIZATION]
    return charts[-1] if charts else None


def _ledger_report(
    user_request: str, completed: list[_Completed], chart: ArtifactPointer | None
) -> FinalReport:
    """Build a report from the ledger alone, no model involved.

    The degraded path: it states what the run produced and stops. No key figures —
    reading a number out of an artifact is the model's job, and guessing one is the
    invention this design forbids.
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
    return FinalReport(executive_summary=summary, key_figures=[], chart=chart)
