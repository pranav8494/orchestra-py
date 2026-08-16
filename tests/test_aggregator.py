"""Tests for the synthesis pass (CONVENTIONS.md §12).

Everything runs against `FakeProvider` and a `tmp_path` store. The assertions are about
what the aggregator does around the model — what it puts in the prompt, what it refuses
to keep from the answer, and what it writes when there is no answer at all — never about
the quality of the synthesis, which is not ours to test.
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from conftest import FakeProvider
from orchestra.agents.aggregator import (
    MAX_PREVIEW_READS,
    Aggregator,
    FigureDraft,
    ReportDraft,
)
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.artifacts import DEFAULT_PREVIEW_LIMIT, ArtifactStore
from orchestra.core.errors import ExitCode, ProviderError, TaskFailure
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


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


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


def _finished_run(store: ArtifactStore) -> TaskState:
    """The walking skeleton's happy ending: three subtasks done, three artifacts stored."""
    state = TaskState(user_request=REQUEST, plan=_plan())
    _finish(state, store, "fetch", "revenue.csv", REVENUE_CSV)
    _finish(state, store, "analyse", "growth.md", "Revenue grew 9.2% in Q2 and 10.7% in Q3.")
    _finish(state, store, "chart", "trend.html", "<html>chart</html>")
    return state


def _draft(*figures: FigureDraft) -> ReportDraft:
    return ReportDraft(
        executive_summary="Revenue grew in each of the last three quarters.",
        key_figures=list(figures),
    )


async def _wait_for_call(provider: FakeProvider) -> None:
    """Let the aggregator reach the provider — the preview reads run in threads first.

    Polled rather than awaited on an event (ASYNC110): the condition is set by a thread
    the test has no handle on. Every caller bounds the wait with `asyncio.wait_for`.
    """
    while not provider.calls:  # noqa: ASYNC110
        await asyncio.sleep(0.001)


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
    # Derived from the visualization subtask, not from anything the model said — the
    # draft schema has no chart field to say it with.
    assert report.chart == state.artifacts["chart"]


@pytest.mark.asyncio
async def test_write_report_shows_the_model_a_preview_not_the_payload(
    store: ArtifactStore,
) -> None:
    """The whole point of pointers (§6): a large artifact must not reach the prompt."""
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
async def test_write_report_falls_back_to_the_ledger_when_the_model_returns_nothing(
    store: ArtifactStore,
) -> None:
    """A refusal or a truncated reply costs the summary, never the run (§10)."""
    state = _finished_run(store)
    provider = FakeProvider(responses=[None])

    report = await Aggregator(provider, store).write_report(state)

    assert state.final_result is report
    assert report.key_figures == []  # nothing is invented on the way down
    assert report.chart == state.artifacts["chart"]
    for subtask_id, pointer in state.artifacts.items():
        assert f"{subtask_id} (" in report.executive_summary
        assert pointer in report.executive_summary
    assert len(provider.calls) == 1  # one call, no retry loop
    assert not state.failed


@pytest.mark.asyncio
async def test_write_report_falls_back_when_every_figure_is_unbacked(
    store: ArtifactStore,
) -> None:
    """Figures that all trace to nothing discredit the summary drawn from them."""
    state = _finished_run(store)
    provider = FakeProvider(
        responses=[
            _draft(
                FigureDraft(label="Q4 revenue", value="160", source=f"{ARTIFACT_PREFIX}ghost.csv")
            )
        ]
    )

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


@pytest.mark.asyncio
async def test_write_report_propagates_a_provider_failure(store: ArtifactStore) -> None:
    """A transport failure is not a refusal: the ledger fallback would hide an outage."""
    state = _finished_run(store)
    provider = FakeProvider(responses=[ProviderError("401 authentication_error")])

    with pytest.raises(ProviderError, match="authentication_error"):
        await Aggregator(provider, store).write_report(state)

    assert state.final_result is None


@pytest.mark.asyncio
async def test_write_report_raises_when_an_artifact_the_ledger_claims_is_gone(
    store: ArtifactStore,
) -> None:
    """The store's one error path, reached through the previews.

    A ledger pointing at a payload the run can no longer read has lost data, so the run
    ends on the taxonomy's `TaskFailure` (§8) rather than reporting over the hole — and
    it ends before the provider call is paid for.
    """
    state = _finished_run(store)
    store.path_for(state.artifacts["analyse"]).unlink()
    provider = FakeProvider(responses=[_draft()])

    with pytest.raises(TaskFailure, match="Artifact not found") as exc_info:
        await Aggregator(provider, store).write_report(state)

    assert exc_info.value.exit_code == ExitCode.TASK_FAILURE
    assert provider.calls == []
    assert state.final_result is None


@pytest.mark.asyncio
async def test_write_report_bounds_how_many_previews_it_reads_at_once(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10: never unbounded fan-out. These reads run on the default thread pool that
    every other `to_thread` in the process shares, so the bound belongs here and not to
    however many subtasks the plan happened to contain."""
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
    await asyncio.wait_for(_wait_for_call(provider), timeout=1)  # in flight, blocked
    task.cancel()

    # Bounded: an aggregator that swallowed the cancellation would sit on the blocker
    # forever, and a suite that hangs tells you nothing.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert state.final_result is None


@pytest.mark.asyncio
async def test_write_report_closes_the_skeleton_over_real_engine_and_stub_worker_output(
    store: ArtifactStore,
) -> None:
    """The last link of the walking skeleton (#8): plan -> engine -> stub artifacts ->
    report, with only the model faked. Nothing here reaches into the store by hand."""
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
    assert report.chart == "artifact:chart.txt"
    # The stub's payload is what the model was shown — proof the previews resolve against
    # what the engine actually wrote, not against a fixture.
    assert "Plot the quarterly revenue trend." in provider.calls[0].messages[0].content


def test_aggregator_prompt_names_the_artifact_pointer_prefix() -> None:
    """The prompt tells the model to copy an `artifact:` pointer; this is what stops that
    instruction drifting from `ARTIFACT_PREFIX` if the prefix ever changes."""
    assert ARTIFACT_PREFIX in AGGREGATOR_SYSTEM_PROMPT
