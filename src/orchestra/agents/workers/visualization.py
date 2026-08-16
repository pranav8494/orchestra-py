"""The third real worker: figures an earlier step computed, drawn (#7).

**No tool loop.** Retrieval and analysis need tools because they act on the world; this
step is one shaped answer — pick the chart, name the points — so it is one
`parse_structured` call, the planner's shape rather than the analytics agent's.

**Two renderings, one artifact.** A chart file to open and a text chart to print are the
same deliverable, and the ledger records one pointer per subtask. So the pointer names a
receipt holding both: the chart's own pointer, and the text inline. `agents/aggregator.py`
reads it back — the report's chart stays a fact of the ledger, not of the model.

**Thin data degrades, never fails.** One point is not a chart, but it is also not a broken
run: the receipt says so in place of the drawing, and the step still completes (§8).
"""

import asyncio
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from orchestra.artifacts import DEFAULT_PREVIEW_LIMIT, ArtifactStore
from orchestra.charts import ChartSpec, insufficient_data, render_ascii, render_html
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import (
    ArtifactPointer,
    EventKind,
    SubtaskContext,
    TaskEvent,
)
from orchestra.prompts import VISUALIZATION_SYSTEM_PROMPT
from orchestra.providers.base import MessageRole, Provider, ProviderMessage


class ChartDraft(BaseModel):
    """The chart as the model writes it. Passed to the provider as `output_format`."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        min_length=1, description="One or two sentences saying what the chart shows."
    )
    spec: ChartSpec


class VisualizationResult(BaseModel):
    """The artifact this worker writes, and the aggregator reads back.

    A contract between two agents, so it is frozen and forbids extras: a field that
    drifted on one side would otherwise be dropped in silence on the other.

    Field order is the preview budget, as in `AnalysisResult`: the aggregator is shown
    `ArtifactStore.preview`, and what it needs to *write about* is the summary. It reads
    `chart` and `ascii_chart` from the whole payload, so they may be elided from the
    preview without loss.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str
    # `None` when the data was too thin to draw. The report then has no chart to open,
    # and `ascii_chart` says why instead of showing bars.
    chart: ArtifactPointer | None
    ascii_chart: str
    instruction: str


class VisualizationWorker:
    """Draws what an earlier step computed. Built in `app.py` (§3.3)."""

    def __init__(
        self,
        *,
        provider: Provider,
        store: ArtifactStore,
        broker: Broker[TaskEvent],
        preview_limit: int = DEFAULT_PREVIEW_LIMIT,
    ) -> None:
        """Take the wired services.

        Args:
            store: the run's artifact store — read for the inputs' previews, written for
                the chart and the receipt.
            broker: the run's event stream, for reporting thin data as a degraded step.
            preview_limit: characters of each input artifact the model is shown.
        """
        self._provider = provider
        self._store = store
        self._broker = broker
        self._preview_limit = preview_limit

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Draw the subtask's chart and store it. See `Worker.run`.

        Raises:
            TaskFailure: the model returned no chart, or the store rejected a write.
            ProviderError: the provider failed. One call and no retry, as in the
                aggregator: a chart is not worth a retried outage.
            asyncio.CancelledError: propagated from the provider or the store (§10).
        """
        briefing = await self._briefing(context)
        draft = await self._provider.parse_structured(
            system=VISUALIZATION_SYSTEM_PROMPT,
            messages=[ProviderMessage(role=MessageRole.USER, content=briefing)],
            output_format=ChartDraft,
        )
        if draft is None:
            # A refusal or a truncated reply. Nothing to draw and nothing to invent, so
            # the step fails and the report names it (§8).
            raise TaskFailure(
                f"Visualization for {context.subtask.id!r} got no chart back from the model."
            )

        thin = insufficient_data(draft.spec)
        if thin is not None:
            await self._warn(context, thin)

        result = VisualizationResult(
            summary=thin or draft.summary,
            chart=None if thin else await self._write_chart(context, draft.spec),
            ascii_chart=thin or render_ascii(draft.spec),
            instruction=context.subtask.instruction,
        )
        return await asyncio.to_thread(
            self._store.put_text, f"{context.subtask.id}.json", result.model_dump_json(indent=2)
        )

    async def _write_chart(self, context: SubtaskContext, spec: ChartSpec) -> ArtifactPointer:
        """Render the chart file and store it, off the event loop.

        Both halves in the one thread: Plotly's HTML is built in Python and the write is
        blocking I/O, and either on the loop would serialise the engine's concurrent
        dispatch (§10).
        """
        name = f"{context.subtask.id}.html"
        return await asyncio.to_thread(lambda: self._store.put_text(name, render_html(spec)))

    async def _warn(self, context: SubtaskContext, reason: str) -> None:
        """Report a step that completed without a drawing.

        Must-deliver, like the tool loop's: the operator's belief about what the run
        produced changes, which is the same class of fact as a state transition (§6).
        """
        await self._broker.publish_lifecycle(
            TaskEvent(
                kind=EventKind.SUBTASK_WARNING,
                subtask_id=context.subtask.id,
                message=reason,
            )
        )

    async def _briefing(self, context: SubtaskContext) -> str:
        """Build the user turn: the step, the request behind it, and the numbers to draw.

        Formatting lives here, not in `prompts/` (§11), and the untrusted text stays out
        of the system prompt.

        Previews, not pointers, unlike the tool loop's briefing: this agent has no tool
        to open an artifact with, so what it is shown is all it can chart.
        """
        lines = [
            f"Subtask: {context.subtask.instruction}",
            f"The request this serves: {context.user_request}",
        ]
        if context.inputs:
            lines.append("Earlier steps produced:")
            lines += await asyncio.to_thread(self._previews, context.inputs)
        lines += [
            f"Clarification asked: {item.question}\nThe user answered: {item.answer}"
            for item in context.clarifications
        ]
        return "\n".join(lines)

    def _previews(self, inputs: Mapping[str, ArtifactPointer]) -> list[str]:
        """Read every input's preview. Blocking — call it in a thread.

        Sequential in one thread rather than a bounded gather: a subtask declares a
        handful of inputs, so the fan-out §10 asks to be bounded is not worth having.
        """
        return [
            f"{name} ({pointer})\n{self._store.preview(pointer, limit=self._preview_limit)}"
            for name, pointer in sorted(inputs.items())
        ]
