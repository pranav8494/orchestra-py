"""Tests for the live dashboard (CONVENTIONS.md §5, §10, §12).

Almost nothing here asserts on Rich's drawn output. The subject is the data handed to
the renderer: `RunView` is what the events fold into, `run_table`/`event_line`/
`result_renderable` are pure functions of it, and what is left for Rich to do is a
choice of glyphs no test should pin. The single exception is the panel-folding
regression, where the defect *was* the rendering — a console setting and a panel's
fixed width combining to crop the result — and its docstring says so.

The async tests are about the two things that go wrong in a subscriber and not in a
function: the startup race (`plan_created` is the only event carrying the plan, so
missing it means an empty table for the whole run) and teardown (§10 — an abandoned
subscription costs the engine the lifecycle timeout on every later event, and an
un-exited `Live` region corrupts the user's shell, §8).
"""

import asyncio
import functools
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rich.panel import Panel
from rich.table import Table

from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.artifacts import ArtifactStore
from orchestra.cli.console import console
from orchestra.cli.format import OutputFormat
from orchestra.cli.render import (
    DASHBOARD_TASK_NAME,
    PLANNING_HEADLINE,
    RenderMode,
    RunView,
    dashboard,
    event_line,
    result_renderable,
    run_table,
)
from orchestra.core.events import Broker
from orchestra.core.state import (
    AgentRole,
    EventKind,
    FinalReport,
    KeyFigure,
    Plan,
    Subtask,
    SubtaskStatus,
    TaskEvent,
    TaskState,
)

PROMPT = "Summarize the last 3 quarters' financial trends"
SUMMARY = "Revenue grew in each of the last three quarters."

if TYPE_CHECKING:
    # The shape `app.run_once(prompt, observer=...)` requires, checked by mypy rather
    # than at runtime. Written against `AbstractAsyncContextManager[object]`, not
    # `app.RunObserver`: that alias yields `None` today, and the type is covariant, so a
    # dashboard yielding its view does not satisfy it until the alias is widened.
    _OBSERVER_CHECK: Callable[[Broker[TaskEvent]], AbstractAsyncContextManager[object]] = (
        functools.partial(dashboard, mode=RenderMode.LIVE)
    )


def _plan() -> Plan:
    return Plan(
        subtasks=[
            Subtask(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Load the ledger"),
            Subtask(id="crunch", role=AgentRole.ANALYTICS, instruction="Compute the trend"),
            Subtask(id="chart", role=AgentRole.VISUALIZATION, instruction="Plot it"),
        ]
    )


def _plan_created(plan: Plan | None = None) -> TaskEvent:
    """The engine's first publish: the message, and the plan copy the rows come from."""
    plan = plan if plan is not None else _plan()
    return TaskEvent(
        kind=EventKind.PLAN_CREATED,
        message=f"Executing {len(plan.subtasks)} subtasks",
        plan=plan,
    )


def _seeded_view() -> RunView:
    view = RunView()
    view.apply(_plan_created())
    return view


def _finished_state() -> TaskState:
    """A ledger shaped the way `run_task` hands one back: a plan, an artifact, a report."""
    plan = _plan()
    plan.subtasks[0].status = SubtaskStatus.DONE
    plan.subtasks[0].output_pointer = "artifact:fetch.txt"
    return TaskState(
        user_request=PROMPT,
        plan=plan,
        artifacts={"fetch": "artifact:fetch.txt"},
        final_result=FinalReport(
            executive_summary=SUMMARY,
            key_figures=[KeyFigure(label="Q3 revenue", value="145", source="artifact:fetch.txt")],
        ),
    )


# --------------------------------------------------------------------------
# RunView — the model every other piece is a function of.
# --------------------------------------------------------------------------


def test_run_view_before_any_event_shows_the_planning_headline() -> None:
    """An empty first frame reads as a hang; planning is the slowest part of a short run."""
    view = RunView()

    assert view.headline == PLANNING_HEADLINE
    assert view.rows == {}
    assert not view.finished


def test_run_view_plan_created_seeds_every_subtask_pending_with_its_role() -> None:
    """The pending rows exist only on this event (`TaskEvent.plan`) — a subscriber cannot
    learn a step from a transition it has not seen yet."""
    view = _seeded_view()

    assert view.headline == "Executing 3 subtasks"
    assert list(view.rows) == ["fetch", "crunch", "chart"]  # insertion order is plan order
    assert [row.role for row in view.rows.values()] == [
        AgentRole.DATA_RETRIEVAL,
        AgentRole.ANALYTICS,
        AgentRole.VISUALIZATION,
    ]
    assert {row.status for row in view.rows.values()} == {SubtaskStatus.PENDING}
    assert not view.finished


def test_run_view_transitions_move_only_the_named_row() -> None:
    """Subtasks run concurrently, so an event for one must leave its neighbours alone."""
    view = _seeded_view()

    view.apply(
        TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id="fetch", message="Load the ledger")
    )
    view.apply(
        TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id="crunch", message="Compute the trend")
    )
    view.apply(
        TaskEvent(
            kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:fetch.txt"
        )
    )
    view.apply(
        TaskEvent(kind=EventKind.SUBTASK_FAILED, subtask_id="crunch", message="worker exploded")
    )

    assert [(row.id, row.status) for row in view.rows.values()] == [
        ("fetch", SubtaskStatus.DONE),
        ("crunch", SubtaskStatus.FAILED),
        ("chart", SubtaskStatus.PENDING),
    ]
    # `message` means something different per kind; the row shows whichever last applied.
    assert view.rows["fetch"].detail == "artifact:fetch.txt"
    assert view.rows["crunch"].detail == "worker exploded"
    assert view.rows["chart"].detail == ""


def test_run_view_run_finished_sets_the_flag_and_the_headline() -> None:
    """The engine's own count is the verdict — recounting the rows would state a second one."""
    view = _seeded_view()

    view.apply(TaskEvent(kind=EventKind.RUN_FINISHED, message="2 of 3 subtasks completed"))

    assert view.finished
    assert view.headline == "2 of 3 subtasks completed"


def test_run_view_second_plan_replaces_the_rows_rather_than_merging() -> None:
    """Replanning (#3) publishes a new plan; keeping the old rows would show steps that
    are no longer going to run."""
    view = _seeded_view()
    replan = Plan(subtasks=[Subtask(id="retry", role=AgentRole.ANALYTICS, instruction="Try again")])

    view.apply(_plan_created(replan))

    assert list(view.rows) == ["retry"]


def test_run_view_event_for_an_unknown_subtask_is_ignored() -> None:
    """Error path: a subscriber that attached after `plan_created` sees only ids it has no
    row for. The renderer is not a validator — it must not raise, and must not invent a
    row whose role and position nothing in the stream states."""
    view = _seeded_view()

    view.apply(TaskEvent(kind=EventKind.SUBTASK_COMPLETED, subtask_id="ghost", message="x"))
    view.apply(TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id=None, message="x"))

    assert list(view.rows) == ["fetch", "crunch", "chart"]
    assert {row.status for row in view.rows.values()} == {SubtaskStatus.PENDING}


# --------------------------------------------------------------------------
# The drawing functions — pure, so asserted on their inputs' projection only.
# --------------------------------------------------------------------------


def test_run_table_carries_the_view_rows_as_cells() -> None:
    """§12: assert on the data handed to the renderer. The columns and the cell text are
    ours; the box drawing around them is Rich's and is not a contract."""
    view = _seeded_view()
    view.apply(
        TaskEvent(
            kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:fetch.txt"
        )
    )

    table = run_table(view)

    assert isinstance(table, Table)
    assert [column.header for column in table.columns] == ["Step", "Role", "Status", "Detail"]
    assert [str(cell) for cell in table.columns[0].cells] == ["fetch", "crunch", "chart"]
    assert [str(cell) for cell in table.columns[2].cells] == ["done", "pending", "pending"]
    assert [str(cell) for cell in table.columns[3].cells] == ["artifact:fetch.txt", "", ""]


def test_run_table_title_is_the_headline() -> None:
    view = _seeded_view()

    assert str(run_table(view).title) == "Executing 3 subtasks"


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (_plan_created(), "plan    Executing 3 subtasks"),
        (
            TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id="fetch", message="Load it"),
            "start   fetch Load it",
        ),
        (
            TaskEvent(
                kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:fetch.txt"
            ),
            "done    fetch artifact:fetch.txt",
        ),
        (
            TaskEvent(kind=EventKind.SUBTASK_FAILED, subtask_id="crunch", message="boom"),
            "failed  crunch boom",
        ),
        (
            TaskEvent(kind=EventKind.RUN_FINISHED, message="3 of 3 subtasks completed"),
            "finish  3 of 3 subtasks completed",
        ),
    ],
    ids=["plan", "started", "completed", "failed", "finished"],
)
def test_event_line_names_the_transition_and_its_subtask(event: TaskEvent, expected: str) -> None:
    """The non-TTY fallback is read as a log: the id and the message have to survive it."""
    assert event_line(event) == expected


# --------------------------------------------------------------------------
# result_renderable — the final report, and the one thing here bound for stdout.
# --------------------------------------------------------------------------


def test_result_renderable_json_is_a_bare_string_even_on_a_terminal() -> None:
    """§5: `--output json` emits one document and nothing else. A panel's box characters
    would break the first `json.loads` that met them, terminal or not."""
    rendered = result_renderable(
        _finished_state(), output=OutputFormat.JSON, quiet=False, terminal=True
    )

    assert isinstance(rendered, str)
    assert rendered.startswith("{")


def test_result_renderable_text_without_a_terminal_is_a_bare_string() -> None:
    """Piped, in a file, or in CI: the frame is corruption rather than an affordance."""
    rendered = result_renderable(
        _finished_state(), output=OutputFormat.TEXT, quiet=False, terminal=False
    )

    assert isinstance(rendered, str)
    assert SUMMARY in rendered
    assert "done     fetch" in rendered  # the trace `cli/format.py` owns, unchanged


def test_result_renderable_text_on_a_terminal_is_a_panel_around_the_report() -> None:
    rendered = result_renderable(
        _finished_state(), output=OutputFormat.TEXT, quiet=False, terminal=True
    )

    assert isinstance(rendered, Panel)
    assert SUMMARY in str(rendered.renderable)


def test_result_renderable_panel_folds_a_long_line_instead_of_cropping_it() -> None:
    """Regression: the panel silently deleted the end of a long step line.

    The one place this file asserts on drawn output, and deliberately: the bug was an
    interaction between the stdout console's `soft_wrap=True` — which means `no_wrap`
    and `overflow="ignore"` — and a panel's fixed width, so nothing short of rendering
    through the real console reproduces it. The contract under test is §5's, that the
    result reaches the reader whole, not any choice of glyph.
    """
    long_id = "fetch_quarterly_financials_by_region"
    pointer = f"artifact:{long_id}.txt"
    plan = Plan(subtasks=[Subtask(id=long_id, role=AgentRole.DATA_RETRIEVAL, instruction="Pull")])
    plan.subtasks[0].status = SubtaskStatus.DONE
    plan.subtasks[0].output_pointer = pointer
    state = TaskState(
        user_request=PROMPT,
        plan=plan,
        artifacts={long_id: pointer},
        final_result=FinalReport(executive_summary=SUMMARY),
    )

    rendered = result_renderable(state, output=OutputFormat.TEXT, quiet=False, terminal=True)
    with console.capture() as captured:
        console.print(rendered)

    # Wrapped onto a second line, not truncated at the border.
    assert pointer in captured.get()


def test_result_renderable_quiet_drops_the_trace_and_keeps_the_report() -> None:
    """§5: `--quiet` suppresses progress, never the result. The trace is the progress."""
    rendered = result_renderable(
        _finished_state(), output=OutputFormat.TEXT, quiet=True, terminal=True
    )

    assert isinstance(rendered, Panel)
    body = str(rendered.renderable)
    assert SUMMARY in body
    assert "Steps:" not in body


# --------------------------------------------------------------------------
# dashboard — subscription lifetime, the startup race, and teardown (§10).
# --------------------------------------------------------------------------


def _pending_tasks() -> list[asyncio.Task[None]]:
    """The dashboard's own live tasks — a leaked consumer shows up here.

    Filtered by name rather than "every task but mine". The broad version made this
    file's assertions hostage to the whole suite: a provider's `httpx` client collected
    at the wrong moment schedules its `aclose()` on whatever loop happens to be running,
    so an unrelated test's garbage could fail a leak check here. `dashboard` names its
    consumer, which is the thing these tests actually mean.
    """
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task.get_name() == DASHBOARD_TASK_NAME
    ]


@pytest.mark.asyncio
async def test_dashboard_none_mode_never_subscribes_and_writes_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--quiet` and `--output json` must cost the run nothing: an attached queue is a
    buffer and a delivery attempt per event, even if the subscriber discards them."""
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.NONE) as view:
        assert broker.subscriber_count == 0
        assert _pending_tasks() == []
        await broker.publish_lifecycle(_plan_created())

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert view.rows == {}  # nothing was subscribed, so nothing was folded in
    assert broker.dropped_lifecycle == 0  # and nobody was there to time out


@pytest.mark.asyncio
async def test_dashboard_subscribes_before_yielding_so_no_event_is_missed() -> None:
    """The startup race: `plan_created` is the only event carrying the plan, so a
    dashboard that yields before its queue is attached draws an empty table all run."""
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.PLAIN) as view:
        assert broker.subscriber_count == 1
        await broker.publish_lifecycle(_plan_created())
        await broker.publish_lifecycle(
            TaskEvent(
                kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:fetch.txt"
            )
        )
        await broker.publish_lifecycle(
            TaskEvent(kind=EventKind.RUN_FINISHED, message="1 of 3 subtasks completed")
        )

    # Drained on the way out: teardown lands a scheduling pass after the last publish,
    # and a final frame reading "running" would describe a run that is over.
    assert view.finished
    assert view.headline == "1 of 3 subtasks completed"
    assert view.rows["fetch"].status is SubtaskStatus.DONE
    assert broker.subscriber_count == 0
    assert broker.dropped_lifecycle == 0


@pytest.mark.asyncio
async def test_dashboard_plain_mode_writes_the_events_to_stderr_and_nothing_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§5/§12: progress is a diagnostic. Both streams are asserted, separately —
    checking only stderr is how a progress line ends up in a piped result."""
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.PLAIN):
        await broker.publish_lifecycle(_plan_created())
        await broker.publish_lifecycle(
            TaskEvent(kind=EventKind.SUBTASK_FAILED, subtask_id="crunch", message="[boom]")
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "plan    Executing 3 subtasks" in captured.err
    # Markup off: a bracketed token in a worker's error would be eaten as a style tag.
    assert "failed  crunch [boom]" in captured.err


@pytest.mark.asyncio
async def test_dashboard_live_mode_folds_events_in_and_leaves_stdout_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `Live` region is on `err_console` (§5). What Rich paints is not asserted — only
    that the model kept up and stdout stayed clean for the result."""
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.LIVE) as view:
        await broker.publish_lifecycle(_plan_created())
        await broker.publish_lifecycle(
            TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id="fetch", message="Load it")
        )

    assert view.rows["fetch"].status is SubtaskStatus.RUNNING
    assert capsys.readouterr().out == ""
    assert broker.subscriber_count == 0


@pytest.mark.asyncio
async def test_dashboard_cancelled_body_unsubscribes_and_leaks_no_task() -> None:
    """§10/§12: cancellation is the path that leaks. An abandoned subscription costs the
    engine `lifecycle_timeout` on every later event — the decision recorded on #11 is
    that the subscription's lifetime *is* the consumer's, which is what this pins."""
    broker: Broker[TaskEvent] = Broker()
    attached = asyncio.Event()

    async def watch() -> None:
        async with dashboard(broker, mode=RenderMode.LIVE):
            attached.set()
            await asyncio.Event().wait()  # held open until cancelled

    task = asyncio.create_task(watch())
    await attached.wait()
    assert broker.subscriber_count == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):  # propagated, never swallowed (§10)
        await task

    assert broker.subscriber_count == 0
    assert _pending_tasks() == []  # the consumer went with it


@pytest.mark.asyncio
async def test_dashboard_body_failure_still_tears_the_consumer_down() -> None:
    """The run raising is the common case (planning failed), and it must not leave a
    subscriber attached or a `Live` region owning the terminal (§8)."""
    broker: Broker[TaskEvent] = Broker()

    with pytest.raises(RuntimeError, match="planner failed"):
        async with dashboard(broker, mode=RenderMode.PLAIN):
            raise RuntimeError("planner failed")

    assert broker.subscriber_count == 0
    assert _pending_tasks() == []


# --------------------------------------------------------------------------
# The whole point, end to end: the renderer against the real engine's stream.
# Every test above feeds hand-built events, so all of them would still pass if
# the engine stopped attaching the plan to `plan_created` — and the table would
# be empty for the whole run. This is the one that notices (#11's second AC).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_follows_a_real_engine_run_to_every_row_done(tmp_path: Path) -> None:
    """No fake events: `ExecutionEngine` publishes, the dashboard subscribes, and the
    view ends up describing the run the engine actually performed."""
    plan = Plan(
        subtasks=[
            Subtask(
                id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Pull the quarterly figures"
            ),
            Subtask(
                id="analyse",
                role=AgentRole.ANALYTICS,
                instruction="Compare the quarters",
                depends_on=["fetch"],
            ),
        ]
    )
    state = TaskState(user_request=PROMPT, plan=plan)
    broker: Broker[TaskEvent] = Broker()
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, EchoWorker(ArtifactStore(tmp_path)))

    async with dashboard(broker, mode=RenderMode.PLAIN) as view:
        await ExecutionEngine(workers=workers, broker=broker).run(state)

    # Plan order, seeded from the event's copy — including the step that had not started
    # when the table was first drawn.
    assert list(view.rows) == ["fetch", "analyse"]
    assert [row.status for row in view.rows.values()] == [SubtaskStatus.DONE] * 2
    assert view.rows["fetch"].detail == state.artifacts["fetch"]  # the pointer it minted
    assert view.finished
    assert view.headline == "2 of 2 subtasks completed"
