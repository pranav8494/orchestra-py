"""Tests for the Visualization worker.

Asserts the contract the aggregator depends on: one receipt pointer out, the chart's own
pointer behind it, thin data degrading to a warning rather than a failure, a refusal
failing the step, and cancellation propagated.

Rendering itself belongs to `orchestra.charts` — see `test_charts.py`. What is checked here
is that the worker wrote what it rendered.
"""

import asyncio
import json

import pytest

from conftest import FakeProvider, wait_until
from orchestra.agents.workers.analytics import AnalysisResult
from orchestra.agents.workers.visualization import (
    ChartDraft,
    VisualizationResult,
    VisualizationWorker,
)
from orchestra.artifacts import ArtifactStore
from orchestra.charts import ChartKind, ChartSeries, ChartSpec
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import (
    AgentRole,
    EventKind,
    Subtask,
    SubtaskContext,
    TaskEvent,
    TaskState,
)
from orchestra.prompts import VISUALIZATION_SYSTEM_PROMPT

REQUEST = "Summarize the last 3 quarters' financial trends and create a chart"
SUBTASK = "chart_trends"
INSTRUCTION = "Plot the quarterly revenue trend"
SUMMARY = "Revenue rose in each of the last three quarters."

UPSTREAM = "analyse_trends"
# A figure that appears only in the payload, so finding it in the briefing proves the
# worker sent the preview and not just the pointer.
UPSTREAM_FIGURE = "2025Q4 revenue growth: 10.65% QoQ"

QUARTERS = ["2025Q2", "2025Q3", "2025Q4"]
REVENUE = [5_820_000.0, 6_340_000.0, 7_015_000.0]

# What the receipt says in place of a drawing when the model returns a single point.
# Spelled out rather than re-derived from `charts.insufficient_data`: this is the sentence
# the user reads as the subtask's summary, so the body is pinned here too.
ONE_POINT = "Insufficient data to chart: 1 point is not a trend, 2 are needed."


def _spec(points: int = len(QUARTERS)) -> ChartSpec:
    """A drawable spec, truncated to `points` categories."""
    return ChartSpec(
        title="Quarterly revenue",
        kind=ChartKind.BAR,
        x_label="Quarter",
        y_label="Revenue",
        categories=QUARTERS[:points],
        series=[ChartSeries(name="Revenue", values=REVENUE[:points])],
    )


def _draft(points: int = len(QUARTERS)) -> ChartDraft:
    """What the model returns from its one structured call."""
    return ChartDraft(summary=SUMMARY, spec=_spec(points))


def _context(inputs: dict[str, str] | None = None) -> SubtaskContext:
    """A worker's slice, built through the ledger like the engine does."""
    subtask = Subtask(
        id=SUBTASK,
        role=AgentRole.VISUALIZATION,
        instruction=INSTRUCTION,
        inputs=sorted(inputs or {}),
    )
    return TaskState(user_request=REQUEST, artifacts=dict(inputs or {})).state_slice(subtask)


def _worker(
    provider: FakeProvider, store: ArtifactStore, broker: Broker[TaskEvent] | None = None
) -> VisualizationWorker:
    """The worker under test; an unsubscribed `Broker` stands in when a test ignores
    warnings, as an unobserved run would."""
    return VisualizationWorker(
        provider=provider,
        store=store,
        broker=broker if broker is not None else Broker(),
    )


def _stored(store: ArtifactStore, pointer: str) -> VisualizationResult:
    """Read the receipt back through its own schema — the aggregator's view of it."""
    return VisualizationResult.model_validate(json.loads(store.get_text(pointer)))


def _seed_upstream(store: ArtifactStore) -> str:
    """Write the artifact #6 would have written, so the preview the model sees is real."""
    analysis = AnalysisResult(
        summary="Growth accelerated into the final quarter.",
        figures=[UPSTREAM_FIGURE],
        instruction="Compute quarter-over-quarter revenue growth",
    )
    return store.put_text(f"{UPSTREAM}.json", analysis.model_dump_json(indent=2))


@pytest.mark.asyncio
async def test_worker_stores_the_chart_and_returns_the_receipt_pointer(
    store: ArtifactStore,
) -> None:
    """One receipt naming the chart, both renderings reachable from it (#7)."""
    provider = FakeProvider(responses=[_draft()])

    pointer = await _worker(provider, store).run(_context())

    assert pointer == f"artifact:{SUBTASK}.json"
    result = _stored(store, pointer)
    assert result.chart is not None
    assert result.chart == f"artifact:{SUBTASK}.html"
    assert "<html" in store.get_text(result.chart)  # real Plotly output, not a stub
    assert result.ascii_chart
    for quarter in QUARTERS:
        assert quarter in result.ascii_chart
    assert result.summary == SUMMARY
    assert result.instruction == INSTRUCTION


@pytest.mark.asyncio
async def test_worker_with_one_point_warns_and_completes_without_a_chart(
    store: ArtifactStore,
) -> None:
    """§8: one point is not a chart, but it is not a broken run either. The receipt says so
    in place of the drawing, and the operator hears about it on the event stream (§6)."""
    provider = FakeProvider(responses=[_draft(points=1)])
    broker: Broker[TaskEvent] = Broker()

    async with broker.subscribe() as queue:
        pointer = await _worker(provider, store, broker).run(_context())
        published = [queue.get_nowait() for _ in range(queue.qsize())]

    result = _stored(store, pointer)
    assert result.chart is None
    assert not (store.root / f"{SUBTASK}.html").exists()  # the chart write never started
    assert result.ascii_chart == ONE_POINT
    assert result.summary == ONE_POINT  # the model's summary described a chart that is absent
    warnings = [event for event in published if event.kind is EventKind.SUBTASK_WARNING]
    assert [(event.subtask_id, event.message) for event in warnings] == [(SUBTASK, ONE_POINT)]


@pytest.mark.asyncio
async def test_worker_retries_an_unusable_reply_and_charts_the_second(store: ArtifactStore) -> None:
    """A refusal is unusable output, so it is retried; the step still completes."""
    provider = FakeProvider(responses=[None, _draft()])

    pointer = await _worker(provider, store).run(_context())

    assert len(provider.calls) == 2
    assert _stored(store, pointer).summary == SUMMARY


@pytest.mark.asyncio
async def test_worker_that_got_no_chart_back_fails_the_subtask(store: ArtifactStore) -> None:
    """Every attempt refused or truncated: nothing to draw and nothing to invent."""
    provider = FakeProvider(responses=[None, None, None])

    with pytest.raises(TaskFailure, match=SUBTASK):
        await _worker(provider, store).run(_context())

    assert len(provider.calls) == 3
    assert list(store.root.iterdir()) == []  # the step failed before either write started


@pytest.mark.asyncio
async def test_worker_propagates_cancellation(store: ArtifactStore) -> None:
    """§10: a cancelled run unwinds through the worker, never swallowed."""
    provider = FakeProvider(responses=[_draft()], blocker=asyncio.Event())  # never set
    task = asyncio.create_task(_worker(provider, store).run(_context()))
    await wait_until(lambda: bool(provider.calls), what="the worker to reach the provider")

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_worker_briefs_the_model_with_the_request_and_each_input_preview(
    store: ArtifactStore,
) -> None:
    """One structured call, the untrusted text kept out of the system prompt (§11), and the
    inputs' *contents* in it: this agent has no tool to open an artifact with, so a pointer
    alone would leave it charting numbers it never saw."""
    upstream = _seed_upstream(store)
    provider = FakeProvider(responses=[_draft()])

    await _worker(provider, store).run(_context({UPSTREAM: upstream}))

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call.system == VISUALIZATION_SYSTEM_PROMPT
    assert call.output_format is ChartDraft
    assert REQUEST not in call.system
    briefing = call.messages[0].content
    assert INSTRUCTION in briefing
    assert REQUEST in briefing
    assert upstream in briefing
    assert UPSTREAM_FIGURE in briefing  # the preview, not merely the pointer
