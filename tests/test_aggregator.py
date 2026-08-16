"""Tests for the synthesis pass.

Everything runs against `FakeProvider` and a `tmp_path` store. The assertions are about
what the aggregator does around the model — what it puts in the prompt, what it refuses to
keep from the answer, what it writes when there is no answer — never the synthesis quality.
"""

import asyncio
import threading
import time

import pytest

from conftest import FakeProvider, wait_until
from orchestra.agents.aggregator import (
    MAX_PREVIEW_READS,
    Aggregator,
    FigureDraft,
    ReportDraft,
)
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.agents.workers.visualization import VisualizationResult
from orchestra.artifacts import DEFAULT_PREVIEW_LIMIT, ArtifactStore
from orchestra.core.errors import ProviderError, TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import (
    ARTIFACT_PREFIX,
    AgentRole,
    Plan,
    Subtask,
    SubtaskStatus,
    TaskEvent,
    TaskState,
)
from orchestra.prompts import AGGREGATOR_SYSTEM_PROMPT

REQUEST = "Summarize the last 3 quarters' financial trends and create a chart"
REVENUE_CSV = "quarter,revenue\nQ1,120\nQ2,131\nQ3,145\n"
CHART = "artifact:chart.html"
ASCII_CHART = "Q1 ######\nQ2 #######\nQ3 ########"


def _plan() -> Plan:
    return Plan(
        subtasks=[
            Subtask(
                id="fetch",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load revenue for the last three quarters.",
            ),
            Subtask(
                id="analyse",
                role=AgentRole.ANALYTICS,
                instruction="Compute quarter-over-quarter growth.",
                inputs=["fetch"],
                depends_on=["fetch"],
            ),
            Subtask(
                id="chart",
                role=AgentRole.VISUALIZATION,
                instruction="Plot the quarterly revenue trend.",
                inputs=["analyse"],
                depends_on=["analyse"],
            ),
        ]
    )


def _finish(
    state: TaskState, store: ArtifactStore, subtask_id: str, name: str, payload: str
) -> str:
    """Complete one subtask exactly as the engine does: artifact, pointer, status."""
    assert state.plan is not None
    subtask = next(item for item in state.plan.subtasks if item.id == subtask_id)
    pointer = store.put_text(name, payload)
    subtask.output_pointer = pointer
    subtask.status = SubtaskStatus.DONE
    state.artifacts[subtask_id] = pointer
    return pointer


def _receipt(*, chart: str | None = CHART, ascii_chart: str = ASCII_CHART) -> str:
    """The visualization worker's artifact: the chart file's pointer and the text chart."""
    return VisualizationResult(
        summary="Revenue rose in each quarter.",
        chart=chart,
        ascii_chart=ascii_chart,
        instruction="Plot the quarterly revenue trend.",
    ).model_dump_json(indent=2)


def _finished_run(store: ArtifactStore, *, receipt: str | None = None) -> TaskState:
    """The walking skeleton's happy ending: three subtasks done, three artifacts stored."""
    state = TaskState(user_request=REQUEST, plan=_plan())
    _finish(state, store, "fetch", "revenue.csv", REVENUE_CSV)
    _finish(state, store, "analyse", "growth.md", "Revenue grew 9.2% in Q2 and 10.7% in Q3.")
    # The visualization subtask's pointer names the receipt, not the drawing.
    _finish(state, store, "chart", "chart.json", receipt if receipt is not None else _receipt())
    return state


def _draft(*figures: FigureDraft) -> ReportDraft:
    return ReportDraft(
        executive_summary="Revenue grew in each of the last three quarters.",
        key_figures=list(figures),
    )


@pytest.mark.asyncio
async def test_write_report_keeps_backed_figures_and_takes_the_chart_from_the_ledger(
    store: ArtifactStore,
) -> None:
    state = _finished_run(store)
    draft = _draft(
        FigureDraft(label="Q3 revenue", value="145", source=state.artifacts["fetch"]),
        FigureDraft(label="Q3 growth", value="10.7%", source=state.artifacts["analyse"]),
    )
    provider = FakeProvider(responses=[draft])

    report = await Aggregator(provider, store).write_report(state)

    assert state.final_result is report
    assert report.executive_summary == draft.executive_summary
    assert [(figure.label, figure.value) for figure in report.key_figures] == [
        ("Q3 revenue", "145"),
        ("Q3 growth", "10.7%"),
    ]
    assert [figure.source for figure in report.key_figures] == [
        state.artifacts["fetch"],
        state.artifacts["analyse"],
    ]
    # From the visualization subtask's receipt: the draft schema has no chart field.
    assert report.chart == CHART
    assert report.chart_ascii == ASCII_CHART


@pytest.mark.asyncio
async def test_write_report_shows_the_model_a_preview_not_the_payload(
    store: ArtifactStore,
) -> None:
    """The point of pointers (§6): a large artifact must not reach the prompt."""
    state = TaskState(user_request=REQUEST, plan=_plan())
    _finish(state, store, "fetch", "revenue.csv", "x" * 5_000 + "TAIL_OF_THE_FILE")
    provider = FakeProvider(responses=[_draft()])

    await Aggregator(provider, store).write_report(state)

    briefing = provider.calls[0].messages[0].content
    assert "[elided," in briefing
    assert "TAIL_OF_THE_FILE" not in briefing
    assert len(briefing) < 2_000


@pytest.mark.asyncio
async def test_write_report_briefs_the_model_with_the_request_role_and_instruction(
    store: ArtifactStore,
) -> None:
    """§11: untrusted input goes in the user turn, never spliced into the instructions."""
    state = _finished_run(store)
    provider = FakeProvider(responses=[_draft()])

    await Aggregator(provider, store).write_report(state)

    call = provider.calls[0]
    assert call.system == AGGREGATOR_SYSTEM_PROMPT
    assert REQUEST not in call.system
    assert call.output_format is ReportDraft
    briefing = call.messages[0].content
    assert REQUEST in briefing
    assert "Subtask analyse (analytics) produced artifact:growth.md" in briefing
    assert "Compute quarter-over-quarter growth." in briefing
    assert "Revenue grew 9.2% in Q2" in briefing  # short artifacts arrive whole


@pytest.mark.asyncio
async def test_write_report_drops_a_figure_citing_an_artifact_the_run_never_produced(
    store: ArtifactStore,
) -> None:
    """A sourced-looking number with no artifact behind it is an invented one (§7)."""
    state = _finished_run(store)
    provider = FakeProvider(
        responses=[
            _draft(
                FigureDraft(label="Q3 revenue", value="145", source=state.artifacts["fetch"]),
                FigureDraft(
                    label="Q4 forecast", value="160", source=f"{ARTIFACT_PREFIX}forecast.csv"
                ),
            )
        ]
    )

    report = await Aggregator(provider, store).write_report(state)

    assert [figure.label for figure in report.key_figures] == ["Q3 revenue"]


@pytest.mark.asyncio
async def test_write_report_drops_a_figure_whose_source_is_not_a_pointer(
    store: ArtifactStore,
) -> None:
    """A malformed citation is as unbacked as an unknown one, and costs only its own figure."""
    state = _finished_run(store)
    provider = FakeProvider(
        responses=[
            _draft(
                FigureDraft(label="Q3 revenue", value="145", source=state.artifacts["fetch"]),
                FigureDraft(label="Q4 forecast", value="160", source="the spreadsheet"),
            )
        ]
    )

    report = await Aggregator(provider, store).write_report(state)

    assert [figure.label for figure in report.key_figures] == ["Q3 revenue"]
    assert len(provider.calls) == 1  # dropped, so no retry was needed


@pytest.mark.asyncio
async def test_write_report_falls_back_to_the_ledger_when_the_model_returns_nothing(
    store: ArtifactStore,
) -> None:
    """A refusal or a truncated reply costs the summary, never the run (§10)."""
    state = _finished_run(store)
    provider = FakeProvider(responses=[None, None, None])

    report = await Aggregator(provider, store).write_report(state)

    assert state.final_result is report
    assert report.key_figures == []  # nothing is invented on the way down
    assert report.chart == CHART
    assert report.chart_ascii == ASCII_CHART
    for subtask_id, pointer in state.artifacts.items():
        assert f"{subtask_id} (" in report.executive_summary
        assert pointer in report.executive_summary
    assert len(provider.calls) == 3  # the retries, then the ledger
    assert not state.failed  # a model with nothing to say is not a broken run


@pytest.mark.asyncio
async def test_write_report_retries_unbacked_figures_with_the_reason_fed_back(
    store: ArtifactStore,
) -> None:
    """The drop is the ledger's rule (§7); the retry is what makes it actionable."""
    state = _finished_run(store)
    backed = FigureDraft(label="Q3 revenue", value="145", source=state.artifacts["fetch"])
    provider = FakeProvider(
        responses=[
            _draft(
                FigureDraft(label="Q4 revenue", value="160", source=f"{ARTIFACT_PREFIX}ghost.csv")
            ),
            _draft(backed),
        ]
    )

    report = await Aggregator(provider, store).write_report(state)

    assert [figure.label for figure in report.key_figures] == ["Q3 revenue"]
    assert len(provider.calls) == 2
    retry = provider.calls[1].messages[1].content
    assert "cite an artifact this run produced" in retry
    assert f"{ARTIFACT_PREFIX}ghost.csv" in retry  # the rejected draft goes back too


@pytest.mark.asyncio
async def test_write_report_falls_back_when_every_figure_is_unbacked(
    store: ArtifactStore,
) -> None:
    """Figures that all trace to nothing discredit the summary drawn from them."""
    state = _finished_run(store)
    unbacked = _draft(
        FigureDraft(label="Q4 revenue", value="160", source=f"{ARTIFACT_PREFIX}ghost.csv")
    )
    provider = FakeProvider(responses=[unbacked, unbacked, unbacked])

    report = await Aggregator(provider, store).write_report(state)

    assert report.key_figures == []
    assert "No synthesis was available" in report.executive_summary


@pytest.mark.asyncio
async def test_write_report_with_no_completed_subtasks_still_produces_a_report(
    store: ArtifactStore,
) -> None:
    """The run the step cap stopped: `app.py` still wants something to print (#8)."""
    state = TaskState(user_request=REQUEST, plan=_plan(), failure_reason="Step cap exceeded")
    provider = FakeProvider(responses=[])

    report = await Aggregator(provider, store).write_report(state)

    assert state.final_result is report
    assert report.key_figures == []
    assert report.chart is None
    assert REQUEST in report.executive_summary
    assert provider.calls == []  # nothing to synthesise, so nothing is paid for
    assert state.failed


@pytest.mark.asyncio
async def test_write_report_without_a_visualization_subtask_leaves_the_chart_unset(
    store: ArtifactStore,
) -> None:
    state = _finished_run(store)
    assert state.plan is not None
    state.plan.subtasks[2].status = SubtaskStatus.FAILED  # the chart step never ran
    provider = FakeProvider(responses=[_draft()])

    report = await Aggregator(provider, store).write_report(state)

    assert report.chart is None
    assert report.chart_ascii is None


@pytest.mark.asyncio
async def test_write_report_keeps_the_text_chart_when_the_data_was_too_thin_to_draw(
    store: ArtifactStore,
) -> None:
    """Thin data costs the drawing, not the explanation: the message is what gets printed."""
    thin = "Only one point to plot, so there is no trend to draw."
    state = _finished_run(store, receipt=_receipt(chart=None, ascii_chart=thin))
    provider = FakeProvider(responses=[_draft()])

    report = await Aggregator(provider, store).write_report(state)

    assert report.chart is None
    assert report.chart_ascii == thin


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses", [[_draft()], [None, None, None]], ids=["synthesised", "ledger"]
)
async def test_write_report_degrades_when_the_visualization_artifact_is_not_a_receipt(
    store: ArtifactStore, responses: list[ReportDraft | None]
) -> None:
    """`EchoWorker` still backs the role and writes plain text. An artifact the aggregator
    cannot read costs the report its chart, never the run (§8) — on both paths."""
    state = _finished_run(store, receipt="role: visualization\ninstruction: plot it")
    provider = FakeProvider(responses=list(responses))

    report = await Aggregator(provider, store).write_report(state)

    assert report.chart is None
    assert report.chart_ascii is None
    assert report.executive_summary  # the rest of the report survives


@pytest.mark.asyncio
async def test_write_report_takes_the_last_visualization_when_a_plan_draws_twice(
    store: ArtifactStore,
) -> None:
    """A plan that draws twice draws the summarising chart last."""
    state = _finished_run(store)
    assert state.plan is not None
    state.plan.subtasks.append(
        Subtask(
            id="chart_2",
            role=AgentRole.VISUALIZATION,
            instruction="Plot the summary.",
            depends_on=["chart"],
        )
    )
    _finish(
        state,
        store,
        "chart_2",
        "chart_2.json",
        _receipt(chart="artifact:summary.html", ascii_chart="Q3 ########"),
    )
    provider = FakeProvider(responses=[_draft()])

    report = await Aggregator(provider, store).write_report(state)

    assert report.chart == "artifact:summary.html"
    assert report.chart_ascii == "Q3 ########"


@pytest.mark.asyncio
async def test_write_report_degrades_only_the_chart_when_the_receipt_cannot_be_read(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt is read whole and the briefing reads previews, so the receipt's read can
    fail alone. It costs the chart, not the answer: the artifacts the summary is written
    from are all still readable, and skipping synthesis would throw away a paid-for run."""
    state = _finished_run(store)

    def unreadable(pointer: str) -> str:
        raise TaskFailure(f"Artifact not found: {pointer!r}")

    monkeypatch.setattr(store, "get_text", unreadable)
    draft = _draft()
    provider = FakeProvider(responses=[draft])

    report = await Aggregator(provider, store).write_report(state)

    assert state.final_result is report
    assert (report.chart, report.chart_ascii) == (None, None)
    assert report.executive_summary == draft.executive_summary  # the summary survives
    assert len(provider.calls) == 1  # the synthesis was not skipped with it
    assert state.failed
    assert "Artifact not found" in (state.failure_reason or "")


@pytest.mark.asyncio
async def test_write_report_returns_the_ledger_report_when_the_provider_fails(
    store: ArtifactStore,
) -> None:
    """An outage costs the synthesis, never the run's report: exit 5 with the artifacts
    named beats exit 4 with an empty stdout."""
    state = _finished_run(store)
    provider = FakeProvider(responses=[ProviderError("401 authentication_error")])

    report = await Aggregator(provider, store).write_report(state)

    assert state.final_result is report
    assert "No synthesis was available" in report.executive_summary
    assert report.chart == CHART  # the chart read succeeded before the provider call
    assert state.failed
    assert "authentication_error" in (state.failure_reason or "")
    assert len(provider.calls) == 1  # an outage is not retried


@pytest.mark.asyncio
async def test_write_report_keeps_a_failure_reason_the_engine_already_recorded(
    store: ArtifactStore,
) -> None:
    """`app.py` records why a run stopped short; the synthesis failure joins it."""
    state = _finished_run(store)
    state.failure_reason = "Step cap exceeded"
    provider = FakeProvider(responses=[ProviderError("503 overloaded_error")])

    await Aggregator(provider, store).write_report(state)

    assert "Step cap exceeded" in (state.failure_reason or "")
    assert "overloaded_error" in (state.failure_reason or "")


@pytest.mark.asyncio
async def test_write_report_degrades_when_an_artifact_the_ledger_claims_is_gone(
    store: ArtifactStore,
) -> None:
    """A ledger pointing at a payload it can no longer read has lost data (§8), so the
    report degrades to the ledger and the run is marked failed — exit 5, with an answer."""
    state = _finished_run(store)
    store.path_for(state.artifacts["analyse"]).unlink()
    provider = FakeProvider(responses=[_draft()])

    report = await Aggregator(provider, store).write_report(state)

    assert state.final_result is report
    assert provider.calls == []  # the briefing failed before the call was paid for
    assert state.failed
    assert "Artifact not found" in (state.failure_reason or "")


@pytest.mark.asyncio
async def test_write_report_bounds_how_many_previews_it_reads_at_once(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10: never unbounded fan-out. The reads share the process-wide thread pool, so the
    bound belongs here, not to the subtask count."""
    plan = Plan(
        subtasks=[
            Subtask(id=f"step_{index}", role=AgentRole.ANALYTICS, instruction="Do the thing")
            for index in range(MAX_PREVIEW_READS * 3)
        ]
    )
    state = TaskState(user_request=REQUEST, plan=plan)
    for subtask in plan.subtasks:
        _finish(state, store, subtask.id, f"{subtask.id}.csv", REVENUE_CSV)

    live = 0
    peak = 0
    counter = threading.Lock()  # the reads are on threads, so the tally must be too
    real = store.preview

    def counted(pointer: str, *, limit: int = DEFAULT_PREVIEW_LIMIT) -> str:
        nonlocal live, peak
        with counter:
            live += 1
            peak = max(peak, live)
        try:
            time.sleep(0.01)  # hold the slot long enough for the others to pile up
            return real(pointer, limit=limit)
        finally:
            with counter:
                live -= 1

    monkeypatch.setattr(store, "preview", counted)

    await Aggregator(FakeProvider(responses=[_draft()]), store).write_report(state)

    assert peak <= MAX_PREVIEW_READS
    assert peak > 1  # bounded, not serialised into a loop that waits on each read


@pytest.mark.asyncio
async def test_write_report_is_cancellable(store: ArtifactStore) -> None:
    """§10: a run the user cannot stop is a defect, so cancellation must propagate."""
    state = _finished_run(store)
    provider = FakeProvider(responses=[_draft()], blocker=asyncio.Event())

    task = asyncio.create_task(Aggregator(provider, store).write_report(state))
    # In flight and blocked: the preview reads run in threads before the request.
    await wait_until(lambda: bool(provider.calls), what="the aggregator to reach the provider")
    task.cancel()

    # Bounded: an aggregator that swallowed the cancellation would hang on the blocker.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert state.final_result is None


@pytest.mark.asyncio
async def test_write_report_closes_the_skeleton_over_real_engine_and_stub_worker_output(
    store: ArtifactStore,
) -> None:
    """The last link of the walking skeleton (#8): plan -> engine -> stub artifacts ->
    report, with only the model faked."""
    state = TaskState(user_request=REQUEST, plan=_plan())
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, EchoWorker(store))
    broker: Broker[TaskEvent] = Broker()
    await ExecutionEngine(workers=workers, broker=broker).run(state)

    provider = FakeProvider(
        responses=[
            _draft(FigureDraft(label="Q3 growth", value="10.7%", source=state.artifacts["analyse"]))
        ]
    )
    report = await Aggregator(provider, store).write_report(state)

    assert not state.failed
    assert [figure.source for figure in report.key_figures] == ["artifact:analyse.txt"]
    # `EchoWorker` writes text, not a receipt, so there is no chart to point at.
    assert report.chart is None
    assert report.chart_ascii is None
    # Proof the previews resolve against what the engine wrote, not against a fixture.
    assert "Plot the quarterly revenue trend." in provider.calls[0].messages[0].content


def test_aggregator_prompt_names_the_artifact_pointer_prefix() -> None:
    """Stops the prompt's `artifact:` instruction drifting from `ARTIFACT_PREFIX`."""
    assert ARTIFACT_PREFIX in AGGREGATOR_SYSTEM_PROMPT
