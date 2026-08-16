"""The synthesis pass: every completed artifact in, one `FinalReport` out.

**Previews, never payloads.** The ledger holds pointers (§6), and this is the one agent
that has to look behind them. It looks through `ArtifactStore.preview`, so a chart's HTML
and a hundred-thousand-row CSV each cost a few hundred characters of prompt. A report
built from raw payloads would be the largest request the run makes, and the one most
likely to be truncated.

**Why a draft model.** `ReportDraft`/`FigureDraft` are the schema the model fills in;
`core.state.FinalReport` is what the ledger keeps. They differ deliberately: a draft
figure's `source` is a plain `str`, because a model can emit anything there, and the
conversion between the two is where that string is checked against the artifacts this
run actually produced (§7). The chart is not in the draft at all — it is a fact the
ledger already holds, so asking the model for it would only create a way to get it wrong.

**Why one call and no retry.** The artifacts are already on disk and paid for. A refusal,
a truncated reply, or a set of figures that cite nothing real must not cost the user the
run, so this falls back to a report built from the ledger alone — what each step
produced, and no invented numbers — rather than retrying or raising. Partial results beat
no results (§8, §10). The same path serves a run with no completed subtasks at all.
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

# How many artifacts may be previewed at once. Bounded here rather than left to the size
# of the plan (§10): the reads run on the default thread pool, which every other
# `to_thread` in the process shares, and the plan's size is the engine's business.
MAX_PREVIEW_READS = 4

# One completed subtask and the artifact it produced — paired at the filter so the rest
# of the module never has to re-ask whether the pointer is there.
_Completed = tuple[Subtask, ArtifactPointer]


class FigureDraft(BaseModel):
    """One key figure as the model writes it — `source` unvalidated by design."""

    # extra="forbid" so a field the model invents is a visible failure, not a silent drop.
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
            state: the run's ledger, after execution. On return `state.final_result` is
                set — always, including when the model refused and when nothing completed.

        Returns:
            The report, the same object as `state.final_result`.

        Raises:
            TaskFailure: an artifact the ledger claims to hold is gone from the store, so
                the run has lost data (§8). Raised by the store, passed through here.
            ProviderError: the provider itself failed; raised at the adapter and passed
                through here untouched. The report is not worth a retried outage.
            asyncio.CancelledError: the caller cancelled the run; propagated, never
                swallowed (§10).
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
            The report, or `None` when the model gave nothing usable — a refusal, a reply
            truncated before the JSON closed, or figures that cite no artifact of this
            run. All three mean the same thing to the caller: fall back to the ledger.
        """
        briefing = await self._briefing(state.user_request, completed)
        draft = await self._provider.parse_structured(
            system=AGGREGATOR_SYSTEM_PROMPT,
            messages=[ProviderMessage(role=MessageRole.USER, content=briefing)],
            output_format=ReportDraft,
        )
        if draft is None:
            return None

        # Plain input validation at the trust boundary (§7), not the guardrail framework
        # #9 will add: a figure is kept only if its source is an artifact this run
        # produced. The set comes from the ledger, so the survivors are valid pointers by
        # construction and `KeyFigure` cannot reject them.
        produced = set(state.artifacts.values())
        figures = [
            KeyFigure(label=figure.label, value=figure.value, source=figure.source)
            for figure in draft.key_figures
            if figure.source in produced
        ]
        if draft.key_figures and not figures:
            # Every number it cited traces to nothing. The summary is built from the same
            # reading of the same artifacts, so it is not more trustworthy than they were.
            return None

        return FinalReport(
            executive_summary=draft.executive_summary,
            key_figures=figures,
            chart=chart,
        )

    async def _briefing(self, user_request: str, completed: list[_Completed]) -> str:
        """Build the user turn: the request, then one section per completed subtask.

        Formatting lives here rather than in `prompts/` (§11), and the request and the
        previews stay out of the system prompt — both are untrusted text.
        """
        # `to_thread` because the store is synchronous filesystem I/O (§10), gathered
        # because the reads are independent, semaphore-bounded because §10 wants the
        # bound stated rather than inherited from whatever the plan happened to contain.
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

    A subtask can be `DONE` with no pointer only if a worker returned nothing; there is
    no artifact to preview, so it is not part of the synthesis.
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

    Last rather than first because a plan that draws twice draws the summarising chart
    last — and one reference is the report's contract, not a gallery.
    """
    charts = [pointer for subtask, pointer in completed if subtask.role is AgentRole.VISUALIZATION]
    return charts[-1] if charts else None


def _ledger_report(
    user_request: str, completed: list[_Completed], chart: ArtifactPointer | None
) -> FinalReport:
    """Build a report from the ledger alone, with no model involved.

    The degraded path: it states what the run produced and stops there. No key figures —
    reading a number out of an artifact is the model's job, and a deterministic guess at
    one would be the invention the whole design forbids.
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
