"""Tests for the composition root (§3.1).

The end-to-end check the walking skeleton exists for: request in, plan, DAG walk, stub
workers write artifacts, aggregator writes the report, ledger comes back with pointers to
all of it — over `FakeProvider`, so no network (§12).

Two structured calls per run, in order: the plan, then the report; `_responses()` queues
both. Tests that build their own `Orchestra` use `EchoWorker` throughout, so a wiring
failure stays distinguishable from an agent failure. The ones going through
`build_orchestra` get the real #5-#7 agents and so also queue `_turns()`, plus `chart=True`
when the run gets as far as the visualization step — #7 makes a structured call of its own,
between those two.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, SecretStr

from conftest import FakeProvider, wait_until
from orchestra import app as app_module
from orchestra.agents.aggregator import Aggregator, FigureDraft, ReportDraft
from orchestra.agents.engine import DEFAULT_STEP_CAP, ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.toolsets import QUERY_CSV_TOOL
from orchestra.agents.workers.analytics import AnalyticsWorker
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.data_retrieval import DataRetrievalWorker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.agents.workers.visualization import ChartDraft
from orchestra.app import RUN_DIR_FORMAT, Orchestra, build_orchestra, run_once
from orchestra.artifacts import ArtifactStore
from orchestra.charts import ChartKind, ChartSeries, ChartSpec
from orchestra.cli.format import OutputFormat, format_result
from orchestra.config import Config
from orchestra.core.errors import ProviderError
from orchestra.core.events import Broker
from orchestra.core.state import (
    ARTIFACT_PREFIX,
    AgentRole,
    EventKind,
    SubtaskStatus,
    TaskEvent,
    artifact_path,
)
from orchestra.providers.anthropic import AnthropicProvider
from orchestra.providers.base import AssistantTurn, Provider
from orchestra.tools.base import ToolCall
from orchestra.tools.python_exec import TOOL_NAME as RUN_PYTHON_TOOL
from scenarios import LINEAR

SUMMARY = "Revenue grew in each of the last three quarters."
# From the scenario rather than spelled out, so a renamed step fails on the name and not
# on a stale literal.
FIRST_STEP = LINEAR.draft().subtasks[0].id
FIRST_POINTER = f"{ARTIFACT_PREFIX}{FIRST_STEP}.txt"
ANALYSIS_STEP = next(
    subtask.id for subtask in LINEAR.draft().subtasks if subtask.role is AgentRole.ANALYTICS
)
ANALYSIS_POINTER = f"{ARTIFACT_PREFIX}{ANALYSIS_STEP}.json"
CHART_STEP = next(
    subtask.id for subtask in LINEAR.draft().subtasks if subtask.role is AgentRole.VISUALIZATION
)
# One category of `_chart_draft`'s spec, so the assertion tracks the fixture.
CHART_CATEGORY = "2025Q3"
# What `_turns` asks for and `_ANALYSIS_CODE` counts, so the fixture figure is one the run
# computes rather than one invented here.
QUARTERS_ANALYSED = 3


def _responses(
    *, figure_source: str | None = None, chart: bool = False
) -> list[BaseModel | BaseException | None]:
    """The answers a run consumes, in order: the plan, the chart draft when the real
    Visualization worker is wired in, then the report draft."""
    figures = (
        [FigureDraft(label="Quarters analysed", value=str(QUARTERS_ANALYSED), source=figure_source)]
        if figure_source is not None
        else []
    )
    drafts = [_chart_draft()] if chart else []
    return [
        LINEAR.draft(),
        *drafts,
        ReportDraft(executive_summary=SUMMARY, key_figures=figures),
    ]


def _chart_draft() -> ChartDraft:
    """A drawable chart for the plan's visualization step — two points, so the step
    completes rather than degrading."""
    return ChartDraft(
        summary="Revenue rose in each quarter.",
        spec=ChartSpec(
            title="Quarterly revenue",
            kind=ChartKind.LINE,
            x_label="Quarter",
            y_label="Revenue",
            categories=[CHART_CATEGORY, "2025Q4"],
            series=[ChartSeries(name="Revenue", values=[6_340_000.0, 7_015_000.0])],
        ),
    )


# Run for real in a subprocess. Stdlib only: this proves the retrieval step's pointer
# resolves inside the executor, and a pandas import would cost a second to prove no more.
_ANALYSIS_CODE = """\
import json
data = json.load(open("fetch_quarterly_financials.json"))
rows = [row for table in data["datasets"] for row in table["csv"].splitlines()[1:] if row]
print("quarters analysed:", len(rows))
"""


def _turns() -> list[AssistantTurn | BaseException]:
    """Both real agents' conversations, in the order the plan runs them.

    Both tool calls are real, so this also checks that the shipped dataset is readable and
    that one worker's pointer reaches the next one's subprocess.
    """
    return [
        AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(id="c1", name=QUERY_CSV_TOOL, arguments={"last_n": QUARTERS_ANALYSED}),
            ),
            usage_tokens=100,
        ),
        AssistantTurn(text="Retrieved the last three quarters.", usage_tokens=50),
        AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(
                    id="c2",
                    name=RUN_PYTHON_TOOL,
                    arguments={
                        "code": _ANALYSIS_CODE,
                        "inputs": ["artifact:fetch_quarterly_financials.json"],
                    },
                ),
            ),
            usage_tokens=100,
        ),
        AssistantTurn(text="Revenue rose in each quarter.", usage_tokens=50),
    ]


def _orchestra(
    store: ArtifactStore, provider: FakeProvider, *, step_cap: int = DEFAULT_STEP_CAP
) -> Orchestra:
    """The real wiring, with the provider substituted — nothing else is faked.

    One store for workers and aggregator, as `build_orchestra` builds it: the aggregator
    resolves the pointers the workers minted.
    """
    broker: Broker[TaskEvent] = Broker()
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, EchoWorker(store))
    return Orchestra(
        planner=Planner(provider),
        engine=ExecutionEngine(workers=workers, broker=broker, step_cap=step_cap),
        aggregator=Aggregator(provider, store),
        provider=provider,
        broker=broker,
        artifact_dir=store.root,
    )


def _config(tmp_path: Path, **overrides: Any) -> Config:
    """A valid config that writes under `tmp_path` and nowhere else, plus `overrides`.

    Here rather than in `conftest.py`: this is the only module that builds a `Config` by
    hand, and a fixture nobody else can use is one shared too early.
    """
    return Config(
        anthropic_api_key=SecretStr("test-key"),
        artifact_dir=tmp_path / "artifacts",
        **overrides,
    )


@pytest.mark.asyncio
async def test_run_task_plans_executes_and_returns_a_ledger_of_pointers(
    store: ArtifactStore,
) -> None:
    provider = FakeProvider(responses=_responses(figure_source=FIRST_POINTER))

    state = await _orchestra(store, provider).run_task(LINEAR.prompt)

    assert state.plan is not None
    assert state.user_request == LINEAR.prompt
    assert state.artifact_dir == store.root  # the ledger records the store it was wired to
    assert [subtask.status for subtask in state.plan.subtasks] == [SubtaskStatus.DONE] * 3
    assert state.failed_subtasks == []
    for subtask in state.plan.subtasks:
        # Every step's output is readable through the pointer state carries.
        assert subtask.instruction in store.get_text(state.artifacts[subtask.id])
    assert state.events[0].kind is EventKind.PLAN_CREATED
    assert state.events[-1].kind is EventKind.RUN_FINISHED
    # The run hands back an answer, not just a ledger.
    assert state.final_result is not None
    assert state.final_result.executive_summary == SUMMARY
    assert [figure.source for figure in state.final_result.key_figures] == [FIRST_POINTER]


@pytest.mark.asyncio
async def test_run_task_subscribers_see_the_run_from_plan_to_finish(store: ArtifactStore) -> None:
    """The dashboard (#11) attaches to `broker` before the run and needs both ends."""
    orchestra = _orchestra(store, FakeProvider(responses=_responses()))

    async with orchestra.broker.subscribe() as queue:
        await orchestra.run_task(LINEAR.prompt)
        kinds = [queue.get_nowait().kind for _ in range(queue.qsize())]

    assert kinds[0] is EventKind.PLAN_CREATED
    assert kinds[-1] is EventKind.RUN_FINISHED
    assert kinds.count(EventKind.SUBTASK_COMPLETED) == 3


@pytest.mark.asyncio
async def test_run_task_reports_what_finished_when_the_engine_ends_the_run(
    store: ArtifactStore,
) -> None:
    """Exit 5 with empty stdout would leave the artifacts on disk unmentioned, which to the
    user is the same as never producing them (#8)."""
    provider = FakeProvider(responses=_responses(figure_source=FIRST_POINTER))

    state = await _orchestra(store, provider, step_cap=1).run_task(LINEAR.prompt)

    assert state.failure_reason is not None
    assert "Step cap of 1 exceeded" in state.failure_reason
    assert state.failed
    assert state.failed_subtasks == []  # nothing failed; the run was stopped
    assert state.artifacts == {FIRST_STEP: FIRST_POINTER}  # only the first step ran
    assert state.final_result is not None
    # Written over what did finish, not over an empty run.
    assert [figure.source for figure in state.final_result.key_figures] == [FIRST_POINTER]


@pytest.mark.asyncio
async def test_run_task_closes_the_provider(store: ArtifactStore) -> None:
    provider = FakeProvider(responses=_responses())
    orchestra = _orchestra(store, provider)

    await orchestra.run_task(LINEAR.prompt)
    await orchestra.aclose()

    assert provider.closed


@pytest.mark.asyncio
async def test_build_orchestra_wires_the_app_from_config_without_touching_the_network(
    tmp_path: Path,
) -> None:
    """Construction is offline and eager: the artifact directory exists before step one (§9)."""
    config = _config(tmp_path)

    orchestra = build_orchestra(config)
    try:
        assert config.artifact_dir.is_dir()
        assert orchestra.broker.subscriber_count == 0
    finally:
        await orchestra.aclose()


@pytest.mark.asyncio
async def test_build_orchestra_roots_the_run_in_a_timestamped_subdirectory(
    tmp_path: Path,
) -> None:
    """A run's files land under a stamp of their own, never in `ARTIFACT_DIR` itself."""
    config = _config(tmp_path)

    orchestra = build_orchestra(config)
    try:
        run_dirs = list(config.artifact_dir.iterdir())
        assert len(run_dirs) == 1
        assert run_dirs[0].is_dir()
        datetime.strptime(run_dirs[0].name, RUN_DIR_FORMAT)  # the stamp it claims to be
    finally:
        await orchestra.aclose()


@pytest.mark.asyncio
async def test_build_orchestra_gives_each_run_its_own_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_run_directory` is scripted rather than waited on: two runs in the same second are
    meant to share a directory, so only two stamps show the name tracking them."""
    config = _config(tmp_path)
    stamps = iter(["2026-08-16T09-30-00Z", "2026-08-16T09-30-01Z"])
    monkeypatch.setattr(app_module, "_run_directory", lambda: next(stamps))

    first = build_orchestra(config)
    second = build_orchestra(config)
    try:
        assert sorted(path.name for path in config.artifact_dir.iterdir()) == [
            "2026-08-16T09-30-00Z",
            "2026-08-16T09-30-01Z",
        ]
        assert first._artifact_dir != second._artifact_dir
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_run_once_keeps_one_run_s_artifacts_together_and_says_where(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ledger is self-describing: every pointer it holds resolves under the directory
    it names, and that directory is a child of `ARTIFACT_DIR`, never `ARTIFACT_DIR` itself."""
    provider = FakeProvider(responses=_responses(chart=True), turns=_turns())
    artifacts = _offline_run(monkeypatch, tmp_path, provider)

    state = await run_once(LINEAR.prompt)

    assert state.artifact_dir is not None
    assert state.artifact_dir.parent == artifacts
    assert state.artifacts  # the run wrote something to be wrong about
    written = {path.name for path in state.artifact_dir.iterdir()}
    for pointer in state.artifacts.values():
        assert pointer.removeprefix(ARTIFACT_PREFIX) in written


@pytest.mark.asyncio
async def test_build_orchestra_carries_every_configured_bound_to_its_consumer(
    tmp_path: Path,
) -> None:
    """Each bound reaches the object that enforces it.

    Read through private attributes deliberately: a settable field nothing passes on is
    worse than no field, because it reports as configurable and is not, and only a live
    request would otherwise show `ANTHROPIC_MAX_TOKENS` never arriving.
    """
    config = _config(
        tmp_path,
        anthropic_max_tokens=1234,
        max_concurrency=7,
        worker_token_budget=4321,
        worker_max_turns=3,
    )

    orchestra = build_orchestra(config)
    try:
        engine = orchestra._engine
        assert cast("AnthropicProvider", orchestra._provider)._max_tokens == 1234
        assert engine._max_concurrency == 7
        # Both real workers, not just the first: each builds its own loop.
        retrieval = cast("DataRetrievalWorker", engine._workers[AgentRole.DATA_RETRIEVAL])
        analytics = cast("AnalyticsWorker", engine._workers[AgentRole.ANALYTICS])
        for loop in (retrieval._loop, analytics._loop):
            assert (loop._max_turns, loop._token_budget) == (3, 4321)
    finally:
        await orchestra.aclose()


@pytest.mark.asyncio
async def test_run_once_enforces_the_configured_turn_cap_on_a_real_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bound is not just held, it bites: one turn, a model still calling tools, and the
    retrieval step fails naming the cap it was given rather than the default."""
    monkeypatch.setenv("WORKER_MAX_TURNS", "1")
    provider = FakeProvider(responses=_responses(), turns=_turns()[:1])
    _offline_run(monkeypatch, tmp_path, provider)

    state = await run_once(LINEAR.prompt)

    assert [subtask.id for subtask in state.failed_subtasks] == [FIRST_STEP]
    failures = [event for event in state.events if event.kind is EventKind.SUBTASK_FAILED]
    assert "still calling tools after 1 turns" in failures[0].message


# --------------------------------------------------------------------------
# `run_once` and its observer: the seam the dashboard attaches to (#11).
# --------------------------------------------------------------------------


@dataclass
class RecordingObserver:
    """A `RunObserver` standing in for `cli/render.py`: subscribes on enter, keeps what came
    through, counts both edges. Nothing draws — §12 asserts on the renderer's input."""

    entered: int = 0
    exited: int = 0
    events: list[TaskEvent] = field(default_factory=list)

    @asynccontextmanager
    async def __call__(self, broker: Broker[TaskEvent]) -> AsyncIterator[None]:
        async with broker.subscribe() as queue:
            self.entered += 1
            try:
                yield
            finally:
                # Drained on the way out rather than by a reader task: the assertion is
                # which events reached a subscriber attached this early. Runs on
                # cancellation too, which is the point (§10).
                self.events.extend(queue.get_nowait() for _ in range(queue.qsize()))
                self.exited += 1


def _offline_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: FakeProvider) -> Path:
    """Let `run_once` load its own config and wire its own services; the vendor adapter is
    swapped at the provider port, the one seam that keeps this offline (§12).

    Returns the `ARTIFACT_DIR` it exported, so a caller asserting on where the run wrote
    reads it from here rather than rebuilding the path.
    """
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))

    def _create_provider(*, api_key: SecretStr, model: str, max_tokens: int) -> Provider:
        return provider

    monkeypatch.setattr(app_module, "create_provider", _create_provider)
    return artifacts


@pytest.mark.asyncio
async def test_run_once_keeps_the_observer_attached_from_the_first_event_to_the_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The plan rides on the first event, so an observer entered late never learns the
    pending rows."""
    provider = FakeProvider(responses=_responses(chart=True), turns=_turns())
    _offline_run(monkeypatch, tmp_path, provider)
    observer = RecordingObserver()

    state = await run_once(LINEAR.prompt, observer=observer)

    assert (observer.entered, observer.exited) == (1, 1)
    kinds = [event.kind for event in observer.events]
    assert kinds[0] is EventKind.PLAN_CREATED  # nothing was published before it attached
    assert observer.events[0].plan is not None
    assert kinds[-1] is EventKind.RUN_FINISHED  # and it was still attached at the end
    assert state.final_result is not None
    assert provider.closed


@pytest.mark.asyncio
async def test_run_once_without_an_observer_runs_headless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default path `cli/app.py` takes today: no subscriber, same ledger."""
    provider = FakeProvider(responses=_responses(chart=True), turns=_turns())
    _offline_run(monkeypatch, tmp_path, provider)

    state = await run_once(LINEAR.prompt)

    assert state.final_result is not None
    assert not state.failed
    assert provider.closed


@pytest.mark.asyncio
async def test_run_once_carries_the_visualization_chart_into_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#7's seam end to end: the worker draws and writes a receipt, the aggregator reads
    both halves back out of it. Each half is covered alone elsewhere; only this proves
    they meet."""
    provider = FakeProvider(responses=_responses(chart=True), turns=_turns())
    _offline_run(monkeypatch, tmp_path, provider)

    state = await run_once(LINEAR.prompt)

    report = state.final_result
    assert report is not None
    assert report.chart == f"{ARTIFACT_PREFIX}{CHART_STEP}.html"
    # Resolved through the run's own directory, which is where the store writes.
    assert state.artifact_dir is not None
    store = ArtifactStore(state.artifact_dir)
    assert "<html" in store.get_text(report.chart)
    # The two composers of the pointer-to-path rule, pinned to each other: the path the
    # CLI prints is the one the store resolves.
    assert f"Chart: {store.path_for(report.chart)}" in format_result(
        state, output=OutputFormat.TEXT
    )
    # The drawing, not just a pointer to one: what the terminal prints is the criterion.
    assert report.chart_ascii is not None
    assert CHART_CATEGORY in report.chart_ascii


@pytest.mark.asyncio
async def test_run_once_reports_a_computed_figure_and_its_chart_in_both_output_shapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#8's criterion end to end: the report cites a number the analytics subprocess really
    printed, sourced to the pointer that step minted, and both shapes carry it.

    Checked against the analysis artifact rather than a literal, so this fails the day the
    step stops computing it instead of passing on the fixture.
    """
    provider = FakeProvider(
        responses=_responses(figure_source=ANALYSIS_POINTER, chart=True), turns=_turns()
    )
    _offline_run(monkeypatch, tmp_path, provider)

    state = await run_once(LINEAR.prompt)

    report = state.final_result
    assert report is not None
    assert state.artifact_dir is not None
    store = ArtifactStore(state.artifact_dir)
    # Survived the aggregator's backing filter, citing a pointer this run actually minted.
    (figure,) = report.key_figures
    assert figure.source == state.artifacts[ANALYSIS_STEP]
    assert f"quarters analysed: {figure.value}" in store.get_text(figure.source)
    assert report.chart is not None
    # `artifact_path`, not `store.path_for`: the store raises on a missing file, so that
    # assertion could only ever raise, never fail.
    assert artifact_path(state.artifact_dir, report.chart).is_file()

    # That the figure reached the text, not how it is laid out there — the column widths
    # are `test_format.py`'s to pin.
    text = format_result(state, output=OutputFormat.TEXT)
    assert figure.value in text
    assert figure.source in text
    assert CHART_CATEGORY in text
    # One document, not a fragment; the stream it goes out on is `test_cli.py`'s.
    document = json.loads(format_result(state, output=OutputFormat.JSON))
    assert document["report"]["key_figures"] == [
        {"label": figure.label, "value": figure.value, "source": figure.source}
    ]
    assert CHART_CATEGORY in document["report"]["chart_ascii"]


@pytest.mark.asyncio
async def test_run_once_exits_the_observer_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A provider failure while planning: `Live` must not be left owning the terminal (§5)."""
    provider = FakeProvider(responses=[ProviderError("The provider is unavailable.")])
    _offline_run(monkeypatch, tmp_path, provider)
    observer = RecordingObserver()

    with pytest.raises(ProviderError):
        await run_once(LINEAR.prompt, observer=observer)

    assert (observer.entered, observer.exited) == (1, 1)
    assert provider.closed


@pytest.mark.asyncio
async def test_run_once_cancellation_exits_the_observer_and_closes_the_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl-C's path (§10): observer torn down, provider released, cancellation re-raised
    rather than swallowed into a ledger nobody asked for."""
    provider = FakeProvider(responses=_responses(), blocker=asyncio.Event())  # never set
    _offline_run(monkeypatch, tmp_path, provider)
    observer = RecordingObserver()

    run = asyncio.create_task(run_once(LINEAR.prompt, observer=observer))
    await wait_until(lambda: bool(provider.calls), what="the run to reach the provider")

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert (observer.entered, observer.exited) == (1, 1)
    assert provider.closed
