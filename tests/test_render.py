"""Tests for the live dashboard (§5, §10).

Almost nothing here asserts on Rich's drawn output. The subject is the data handed to the
renderer: `RunView` is what the events fold into, and `run_table`/`active_panel`/
`event_log`/`event_line`/`result_renderable` are pure functions of it. The exceptions are
the two layout regressions — the folded report panel and the wrapped active row — where
the defect *was* the rendering.

The async tests cover what goes wrong in a subscriber rather than a function: the startup
race and teardown.
"""

import asyncio
import functools
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table

from conftest import ScriptedWorker, planned, wait_until
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.app import RunObserver
from orchestra.artifacts import ArtifactStore
from orchestra.cli import render
from orchestra.cli.console import console
from orchestra.cli.format import OutputFormat
from orchestra.cli.render import (
    DASHBOARD_TASK_NAME,
    EVENT_LOG_LINES,
    PLANNING_HEADLINE,
    SPINNER_TASK_NAME,
    LiveRegion,
    RenderMode,
    RunView,
    active_panel,
    dashboard,
    dashboard_frame,
    event_line,
    event_log,
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
from scenarios import FAN_OUT

PROMPT = "Summarize the last 3 quarters' financial trends"
SUMMARY = "Revenue grew in each of the last three quarters."

if TYPE_CHECKING:
    # `dashboard` has to satisfy the seam `run_once(prompt, observer=...)` takes. Annotated
    # with the alias, so narrowing `RunObserver` fails here rather than at the call site.
    _OBSERVER_CHECK: RunObserver = functools.partial(dashboard, mode=RenderMode.LIVE)


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
    """An empty first frame reads as a hang, and planning is the slowest part."""
    view = RunView()

    assert view.headline == PLANNING_HEADLINE
    assert view.rows == {}
    assert not view.finished


def test_run_view_plan_created_seeds_every_subtask_pending_with_its_role() -> None:
    """The pending rows exist only on this event: a subscriber cannot learn a step from a
    transition it has not seen yet."""
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
    """The engine's count is the verdict; recounting the rows would state a second one."""
    view = _seeded_view()

    view.apply(TaskEvent(kind=EventKind.RUN_FINISHED, message="2 of 3 subtasks completed"))

    assert view.finished
    assert view.headline == "2 of 3 subtasks completed"


def test_run_view_second_plan_replaces_the_rows_rather_than_merging() -> None:
    """Replanning (#3) publishes a new plan; old rows would show steps that will not run."""
    view = _seeded_view()
    replan = Plan(subtasks=[Subtask(id="retry", role=AgentRole.ANALYTICS, instruction="Try again")])

    view.apply(_plan_created(replan))

    assert list(view.rows) == ["retry"]


def test_run_view_event_for_an_unknown_subtask_is_ignored() -> None:
    """A subscriber attached after `plan_created` sees only ids it has no row for. The
    renderer is not a validator: it must not raise, and must not invent a row."""
    view = _seeded_view()

    view.apply(TaskEvent(kind=EventKind.SUBTASK_COMPLETED, subtask_id="ghost", message="x"))
    view.apply(TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id=None, message="x"))

    assert list(view.rows) == ["fetch", "crunch", "chart"]
    assert {row.status for row in view.rows.values()} == {SubtaskStatus.PENDING}


# --------------------------------------------------------------------------
# The drawing functions — pure, so asserted on their inputs' projection only.
# --------------------------------------------------------------------------


def test_run_table_carries_the_view_rows_as_cells() -> None:
    """The columns and cell text are ours; the box drawing around them is Rich's (§12)."""
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


# --------------------------------------------------------------------------
# The polish half of #11: spinners, the active-agent panel, and the event log.
# --------------------------------------------------------------------------


def _started(subtask_id: str) -> TaskEvent:
    return TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id=subtask_id, message="working")


def test_run_view_seeds_each_row_with_its_instruction() -> None:
    """The active panel names the work, and only the plan carries it."""
    assert _seeded_view().rows["crunch"].instruction == "Compute the trend"


def test_run_view_active_lists_every_running_row_in_plan_order() -> None:
    """The fan-out signal (#17): two subtasks dispatched at once are two spinners, not one
    summary line."""
    view = _seeded_view()
    view.apply(_started("fetch"))
    view.apply(_started("crunch"))
    view.apply(
        TaskEvent(kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:f.txt")
    )
    view.apply(_started("chart"))

    assert [row.id for row in view.active] == ["crunch", "chart"]


def test_run_view_log_keeps_the_most_recent_events_and_drops_the_oldest() -> None:
    """Bounded, so it scrolls: a long run must not grow the region past the terminal."""
    view = RunView()
    for index in range(EVENT_LOG_LINES + 3):
        view.apply(
            TaskEvent(kind=EventKind.SUBTASK_WARNING, subtask_id="fetch", message=str(index))
        )

    assert len(view.log) == EVENT_LOG_LINES
    assert view.log[-1].message == str(EVENT_LOG_LINES + 2)
    assert view.log[0].message == "3"  # the first three scrolled off


def test_run_view_log_records_an_event_for_a_subtask_it_has_no_row_for() -> None:
    """`apply` drops such an event from the table on purpose; the log is the only place a
    plan/stream mismatch stays visible."""
    view = _seeded_view()

    view.apply(TaskEvent(kind=EventKind.SUBTASK_FAILED, subtask_id="ghost", message="boom"))

    assert "ghost" not in view.rows
    assert event_line(view.log[-1]) == "failed  ghost boom"


def _active_grid(view: RunView) -> Table:
    grid = active_panel(view).renderable
    assert isinstance(grid, Table)
    return grid


def test_active_panel_draws_one_spinner_per_running_subtask() -> None:
    """The renderable, not Rich's painting of it (§12): one `Spinner` per active row."""
    view = _seeded_view()
    view.apply(_started("fetch"))
    view.apply(_started("crunch"))

    grid = _active_grid(view)

    assert all(isinstance(cell, Spinner) for cell in grid.columns[0].cells)
    assert [str(cell) for cell in grid.columns[1].cells] == [
        "data_retrieval  Load the ledger",
        "analytics  Compute the trend",
    ]


def test_active_panel_elides_a_long_instruction_instead_of_wrapping_it() -> None:
    """Regression: the label was a `Spinner(text=...)`, and `Spinner.render` rebuilds it
    through `Text.assemble`, which drops `no_wrap`. An unbounded planner instruction
    wrapped on an 80-column terminal, resizing the region every frame."""
    plan = Plan(
        subtasks=[Subtask(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Pull " * 40)]
    )
    view = RunView()
    view.apply(_plan_created(plan))
    view.apply(_started("fetch"))

    label = _active_grid(view).columns[1]
    assert label.no_wrap
    assert label.overflow == "ellipsis"

    with console.capture() as captured:
        console.print(active_panel(view), width=60)
    assert len(captured.get().splitlines()) == 3  # the two borders and one row


def test_active_panel_holds_an_idle_row_between_steps() -> None:
    """A sequential plan has a frame with nothing running at every handoff. Dropping the
    panel there would resize the region once per step."""
    view = _seeded_view()

    grid = _active_grid(view)

    assert not view.active
    assert [str(cell) for cell in grid.columns[1].cells] == ["waiting for the next step"]


def test_event_log_shows_the_stream_oldest_first() -> None:
    view = _seeded_view()
    view.apply(_started("fetch"))

    body = str(event_log(view).renderable)

    assert body.splitlines() == ["plan    Executing 3 subtasks", "start   fetch working"]


def test_dashboard_frame_grows_to_three_panels_and_keeps_them() -> None:
    """Before the first event there is only the table; once there is a plan the panels stay
    put, so a handoff costs no region resize. `run_finished` drops the active panel for
    good."""
    empty = dashboard_frame(RunView())
    assert isinstance(empty, Group)
    assert [type(part) for part in empty.renderables] == [Table]

    view = _seeded_view()
    running = dashboard_frame(view)
    assert isinstance(running, Group)
    assert [type(part) for part in running.renderables] == [Table, Panel, Panel]

    view.apply(_started("fetch"))
    still_running = dashboard_frame(view)
    assert isinstance(still_running, Group)
    assert [type(part) for part in still_running.renderables] == [Table, Panel, Panel]

    view.apply(TaskEvent(kind=EventKind.RUN_FINISHED, message="3 of 3 subtasks completed"))
    finished = dashboard_frame(view)
    assert isinstance(finished, Group)
    assert [type(part) for part in finished.renderables] == [Table, Panel]


def _log_lines(frame: RenderableType) -> list[str]:
    """The Events panel's lines, or `[]` if the frame has no Events panel."""
    assert isinstance(frame, Group)
    panels = [part for part in frame.renderables if isinstance(part, Panel)]
    events = [panel for panel in panels if panel.title == "Events"]
    return str(events[0].renderable).splitlines() if events else []


def _long_run() -> RunView:
    """Eight subtasks and a full log — the shape that overran a 24-line terminal."""
    plan = Plan(
        subtasks=[
            Subtask(id=f"step{index}", role=AgentRole.ANALYTICS, instruction="Work")
            for index in range(8)
        ]
    )
    view = RunView()
    view.apply(_plan_created(plan))
    for index in range(EVENT_LOG_LINES):
        view.apply(
            TaskEvent(kind=EventKind.SUBTASK_WARNING, subtask_id="step0", message=str(index))
        )
    return view


def test_dashboard_frame_without_a_height_keeps_the_whole_log() -> None:
    """`None` is unbounded: nothing but a real terminal has a height to fit."""
    assert len(_log_lines(dashboard_frame(_long_run()))) == EVENT_LOG_LINES


def test_dashboard_frame_trims_the_log_to_fit_a_short_terminal() -> None:
    """Each part is bounded but the sum was not, so an 8-step plan overran a 24-line
    terminal and Rich cropped the frame from the bottom (#39). The log gives way."""
    view = _long_run()

    frame = dashboard_frame(view, height=24)

    # 8 rows + 5 of table chrome + the idle active panel's 3 = 16, leaving 8 for the
    # Events panel: 6 lines inside its 2 borders.
    assert len(_log_lines(frame)) == 6
    assert _log_lines(frame)[-1].endswith(str(EVENT_LOG_LINES - 1))  # the newest survive


def test_dashboard_frame_drops_the_log_when_the_plan_alone_fills_the_terminal() -> None:
    """A two-line box with nothing in it is worse than no box: the table has to fit."""
    frame = dashboard_frame(_long_run(), height=16)

    assert _log_lines(frame) == []
    assert isinstance(frame, Group)
    assert [type(part) for part in frame.renderables] == [Table, Panel]  # table + active


def test_dashboard_frame_drops_the_active_panel_once_the_stream_detaches() -> None:
    """Ctrl-C cancels the steps rather than failing them, so the rows still read `running`;
    animating them in the frame it freezes would claim work that stopped (#39)."""
    view = _seeded_view()
    view.apply(_started("fetch"))
    running = dashboard_frame(view)
    assert isinstance(running, Group)
    assert [type(part) for part in running.renderables] == [Table, Panel, Panel]

    view.stopped = True
    frame = dashboard_frame(view)

    assert view.active  # the ledger still says so — it is the drawing that must not
    assert isinstance(frame, Group)
    assert [type(part) for part in frame.renderables] == [Table, Panel]  # table + events


def test_event_line_collapses_a_multi_line_message_onto_one_line() -> None:
    """Regression: `engine.py` publishes `subtask_failed` with `str(exc)`, and
    `str(ValidationError)` is multi-line — a newline adds a row to the region rather than
    wrapping inside a cell."""
    event = TaskEvent(
        kind=EventKind.SUBTASK_FAILED,
        subtask_id="crunch",
        message="1 validation error for FinalReport\nexecutive_summary\n  Field required",
    )

    assert event_line(event) == (
        "failed  crunch 1 validation error for FinalReport executive_summary Field required"
    )


def test_run_view_collapses_a_multi_line_message_in_the_cells_too() -> None:
    """The table reads `row.detail`, not `event_line`, so the rule has to apply to both."""
    view = _seeded_view()

    view.apply(TaskEvent(kind=EventKind.SUBTASK_FAILED, subtask_id="fetch", message="boom\n  at x"))
    view.apply(TaskEvent(kind=EventKind.SUBTASK_WARNING, subtask_id="crunch", message="a\nb"))

    assert view.rows["fetch"].detail == "boom at x"
    assert view.rows["crunch"].warning == "a b"


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
    """§5: a panel's box characters would break the first `json.loads` that met them."""
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

    The one place this file asserts on drawn output: the defect was `soft_wrap=True`
    meeting a panel's fixed width, which nothing short of the real console reproduces.
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


def test_result_renderable_panel_embeds_the_ascii_chart_and_its_openable_path(
    tmp_path: Path,
) -> None:
    """#11's last polish item, pinned at the panel rather than the string: `test_format.py`
    already covers both blocks, and this says the final *panel* carries them."""
    state = _finished_state()
    state.artifact_dir = tmp_path
    state.final_result = FinalReport(
        executive_summary=SUMMARY,
        chart="artifact:trend.html",
        chart_ascii="Q1 ###\nQ2 #####",
    )

    rendered = result_renderable(state, output=OutputFormat.TEXT, quiet=False, terminal=True)

    assert isinstance(rendered, Panel)
    body = str(rendered.renderable)
    assert "Q2 #####" in body
    assert f"Chart: {tmp_path / 'trend.html'}" in body


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

    By name, not "every task but mine": an unrelated `httpx` client collected at the wrong
    moment schedules its `aclose()` on whatever loop is running, and the broad version
    counted that as a leak.
    """
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_name() in {DASHBOARD_TASK_NAME, SPINNER_TASK_NAME}
    ]


@pytest.mark.asyncio
async def test_dashboard_none_mode_never_subscribes_and_writes_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--quiet` and `--output json` must cost the run nothing: an attached queue is a
    delivery attempt per event even if the subscriber discards them."""
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
    """`plan_created` is the only event carrying the plan, so a dashboard that yields
    before its queue is attached draws an empty table all run."""
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
    """§5: progress is a diagnostic. Both streams are asserted separately — checking only
    stderr is how a progress line ends up in a piped result."""
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
    """The `Live` region is on `err_console` (§5). What Rich paints is not asserted, only
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
async def test_dashboard_live_mode_opens_no_region_until_the_first_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#10: the planner asks its questions before the first event, and a `Live` already
    owning the terminal would swallow the prompt. So the region waits, and a plain line
    says the run is alive."""
    started = 0
    start = Live.start

    def counting_start(live: Live, refresh: bool = False) -> None:
        nonlocal started
        started += 1
        start(live, refresh=refresh)

    monkeypatch.setattr(Live, "start", counting_start)
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.LIVE) as view:
        assert started == 0  # nothing has been published, so nothing owns the terminal
        await broker.publish_lifecycle(_plan_created())
        await wait_until(lambda: started == 1, what="the region to open on the first event")

    assert view.rows  # the event that opened it was folded in, not dropped
    captured = capsys.readouterr()
    assert PLANNING_HEADLINE in captured.err  # a diagnostic, so stderr (§5)
    assert captured.out == ""


@pytest.mark.asyncio
async def test_dashboard_live_mode_redraws_between_events_so_the_spinners_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rich advances a spinner only when the region is redrawn, and lifecycle events arrive
    seconds apart. Counting refreshes, not frames: which glyph Rich picked is Rich's (§12)."""
    monkeypatch.setattr(render, "SPINNER_TICK_SECONDS", 0.001)
    refreshes = 0

    def count(_live: Live) -> None:
        nonlocal refreshes
        refreshes += 1

    monkeypatch.setattr(Live, "refresh", count)
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.LIVE) as view:
        idle = refreshes
        await asyncio.sleep(0.05)
        assert refreshes == idle  # nothing is running, so there is nothing to animate

        await broker.publish_lifecycle(_plan_created())
        await broker.publish_lifecycle(_started("fetch"))
        await asyncio.sleep(0.01)  # let the consumer drain both
        assert view.active
        after_events = refreshes
        await asyncio.sleep(0.05)  # no further events: only the tick can redraw
        ticked = refreshes - after_events

    assert ticked > 0
    assert _pending_tasks() == []  # and the ticker went with the consumer


@pytest.mark.asyncio
async def test_spin_keeps_ticking_when_stderr_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`orchestra run ... | head -1` closes stderr. §5 lets that cost the animation, never
    the run, so the ticker drops the `OSError` rather than dying on the first one."""
    monkeypatch.setattr(render, "SPINNER_TICK_SECONDS", 0.001)
    refreshes = 0

    class BrokenLive:
        def refresh(self) -> None:
            nonlocal refreshes
            refreshes += 1
            raise BrokenPipeError("stderr closed")

    view = _seeded_view()
    view.apply(_started("fetch"))
    ticker = asyncio.create_task(render._spin(cast(Live, BrokenLive()), view, LiveRegion()))
    await asyncio.sleep(0.02)

    assert refreshes > 1  # it kept going rather than stopping on the first failure
    assert not ticker.done()

    ticker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ticker


@pytest.mark.asyncio
async def test_dashboard_reports_a_ticker_that_died_instead_of_losing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§8: `Task.cancel()` clears the unretrieved-exception flag even on a task that has
    already raised, so cancelling the ticker without reading it first erases the failure."""

    async def boom(_live: Live, _view: RunView, _region: LiveRegion) -> None:
        raise RuntimeError("ticker bug")

    monkeypatch.setattr(render, "_spin", boom)
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.LIVE):
        # The ticker belongs to the region, and the region opens on the first event.
        await broker.publish_lifecycle(_plan_created())
        await asyncio.sleep(0)  # let the ticker start and die
        outcome = "the run still produced its result"

    assert outcome == "the run still produced its result"
    captured = capsys.readouterr()
    assert captured.out == ""  # a diagnostic, so stderr only (§5)
    assert "The dashboard stopped: ticker bug" in captured.err
    assert _pending_tasks() == []


@pytest.mark.asyncio
async def test_dashboard_marks_the_view_stopped_so_the_last_frame_does_not_animate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teardown paints one more frame before the region is exited (#39). The ticker stops
    with it, or it would keep refreshing a run that is over."""
    monkeypatch.setattr(render, "SPINNER_TICK_SECONDS", 0.001)
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.LIVE) as view:
        await broker.publish_lifecycle(_plan_created())
        await broker.publish_lifecycle(_started("fetch"))
        await wait_until(lambda: bool(view.active), what="the step to show as running")

    assert view.stopped
    assert view.active  # the rows are untouched; only the drawing gives up on them
    frame = dashboard_frame(view)
    assert isinstance(frame, Group)
    assert [type(part) for part in frame.renderables] == [Table, Panel]


@pytest.mark.asyncio
async def test_dashboard_cancelled_while_a_step_runs_leaks_no_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§15: the other teardown tests cancel a ticker parked in `sleep` having never
    refreshed. This one cancels it mid-work."""
    monkeypatch.setattr(render, "SPINNER_TICK_SECONDS", 0.001)
    broker: Broker[TaskEvent] = Broker()
    working = asyncio.Event()

    async def watch() -> None:
        async with dashboard(broker, mode=RenderMode.LIVE) as view:
            await broker.publish_lifecycle(_plan_created())
            await broker.publish_lifecycle(_started("fetch"))
            await wait_until(lambda: bool(view.active), what="the step to show as running")
            await asyncio.sleep(0.01)  # and the ticker to actually refresh the region
            working.set()
            await asyncio.Event().wait()  # held open until cancelled

    task = asyncio.create_task(watch())
    await working.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert broker.subscriber_count == 0
    assert _pending_tasks() == []  # both the consumer and its ticker went with it


@pytest.mark.asyncio
async def test_dashboard_cancelled_body_unsubscribes_and_leaks_no_task() -> None:
    """§10: cancellation is the path that leaks, and an abandoned subscription costs the
    engine `lifecycle_timeout` on every later event. #11 decided the subscription's
    lifetime *is* the consumer's; this pins that."""
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
    """A raising run must not leave a subscriber attached or a `Live` region owning the
    terminal (§8)."""
    broker: Broker[TaskEvent] = Broker()

    with pytest.raises(RuntimeError, match="planner failed"):
        async with dashboard(broker, mode=RenderMode.PLAIN):
            raise RuntimeError("planner failed")

    assert broker.subscriber_count == 0
    assert _pending_tasks() == []


@pytest.mark.parametrize("mode", [RenderMode.PLAIN, RenderMode.LIVE])
@pytest.mark.asyncio
async def test_dashboard_cancelled_during_teardown_still_propagates(mode: RenderMode) -> None:
    """Regression: Ctrl-C arriving while teardown awaited the consumer was swallowed, and
    the run exited 0 with a `TaskState` instead of 130. Cancelling the caller mid-`await`
    is delivered by cancelling the awaited future, so the consumer reads `cancelled()` on
    both paths. Swept across scheduling passes: the window is a couple wide and moves with
    the mode.
    """

    async def run_to_completion() -> str:
        async with dashboard(Broker[TaskEvent](), mode=mode):
            pass  # the body returns, so teardown is where the cancellation lands
        return "returned normally"

    cancelled_while_live = 0
    for passes in range(40):
        task = asyncio.ensure_future(run_to_completion())
        for _ in range(passes):
            await asyncio.sleep(0)
        if task.done() or not task.cancel():
            continue  # never got a genuine in-flight cancellation on this pass
        cancelled_while_live += 1
        with pytest.raises(asyncio.CancelledError):
            await task

    # The sweep has to have actually hit the window, or the test asserts nothing.
    assert cancelled_while_live > 0


@pytest.mark.asyncio
async def test_dashboard_sink_failure_costs_the_diagnostic_not_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: `orchestra run ... | head -1` closes stderr, the sink raises
    `BrokenPipeError`, teardown re-raised it, and a successful run came back exit 1 with
    empty stdout. Progress is a diagnostic; §5 does not let it cost the result."""

    def broken_sink(_event: TaskEvent) -> None:
        raise BrokenPipeError("stderr closed")

    monkeypatch.setattr(render, "_write_line", broken_sink)
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.PLAIN):
        await broker.publish_lifecycle(_plan_created())
        await asyncio.sleep(0)  # let the consumer pick it up and die on the write
        outcome = "the run still produced its result"

    assert outcome == "the run still produced its result"
    assert broker.subscriber_count == 0  # and it still detached
    assert _pending_tasks() == []


@pytest.mark.asyncio
async def test_dashboard_sink_failure_does_not_mask_the_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the run failed *and* the renderer died, the CLI reports why the run failed."""

    def broken_sink(_event: TaskEvent) -> None:
        raise BrokenPipeError("stderr closed")

    monkeypatch.setattr(render, "_write_line", broken_sink)
    broker: Broker[TaskEvent] = Broker()

    with pytest.raises(RuntimeError, match="planner failed"):
        async with dashboard(broker, mode=RenderMode.PLAIN):
            await broker.publish_lifecycle(_plan_created())
            await asyncio.sleep(0)
            raise RuntimeError("planner failed")


# --------------------------------------------------------------------------
# End to end against the real engine's stream. Every test above feeds hand-built
# events, so all of them would pass if the engine stopped attaching the plan to
# `plan_created` — and the table would be empty all run. This is the one that
# notices (#11's second AC).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_follows_a_real_engine_run_to_every_row_done(store: ArtifactStore) -> None:
    """No fake events: the engine publishes and the view ends up describing the run."""
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
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, EchoWorker(store))

    async with dashboard(broker, mode=RenderMode.PLAIN) as view:
        await ExecutionEngine(workers=workers, broker=broker).run(state)

    # Plan order, seeded from the event's copy, including the step that had not started
    # when the table was first drawn.
    assert list(view.rows) == ["fetch", "analyse"]
    assert [row.status for row in view.rows.values()] == [SubtaskStatus.DONE] * 2
    assert view.rows["fetch"].detail == state.artifacts["fetch"]  # the pointer it minted
    assert view.finished
    assert view.headline == "2 of 2 subtasks completed"


@pytest.mark.asyncio
async def test_dashboard_shows_both_fan_out_retrievals_spinning_at_once() -> None:
    """#17's dashboard AC, and the gap the two `active` tests above leave: they fold
    hand-written `started` events, so both would pass with the retrievals run one after the
    other. `test_engine` proves the engine dispatches them together; this is the one that
    proves the frame says so. The gate holds both open, so the frame read here is the run's
    rather than whichever moment the assertion arrived in.
    """
    retrievals = ["fetch_our_growth", "fetch_industry_benchmarks"]
    state = await planned(FAN_OUT)
    assert state.plan is not None
    instructions = {subtask.id: subtask.instruction for subtask in state.plan.subtasks}
    gate = asyncio.Event()
    worker = ScriptedWorker(gate=gate, gate_ids=frozenset(retrievals))
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, worker)
    broker: Broker[TaskEvent] = Broker()

    async with dashboard(broker, mode=RenderMode.LIVE) as view:
        run = asyncio.create_task(ExecutionEngine(workers=workers, broker=broker).run(state))
        await wait_until(lambda: len(view.active) == 2, what="both retrievals to report started")

        # The renderable (§12): one spinner each, naming the work rather than the count.
        # Read off the plan, not spelled out, so the wording stays `scenarios.py`'s.
        grid = _active_grid(view)
        assert [row.id for row in view.active] == retrievals
        assert [type(cell) for cell in grid.columns[0].cells] == [Spinner, Spinner]
        assert [str(cell) for cell in grid.columns[1].cells] == [
            f"{AgentRole.DATA_RETRIEVAL.value}  {instructions[subtask_id]}"
            for subtask_id in retrievals
        ]

        # And painted, for the "visibly" in the AC: two rows in the one panel, each still
        # its own agent's work. Width pinned, and only the head of the instruction asserted,
        # because the panel elides the tail — the behaviour the test above it pins.
        with console.capture() as painted:
            console.print(active_panel(view), width=80)
        drawn = painted.get().splitlines()[1:-1]  # between the panel's two borders
        assert len(drawn) == 2
        for subtask_id, line in zip(retrievals, drawn, strict=True):
            assert f"{AgentRole.DATA_RETRIEVAL.value}  {instructions[subtask_id][:20]}" in line

        gate.set()
        await run

    # Released, the gated run still finished — the frame above was mid-run, not a stall.
    assert {row.status for row in view.rows.values()} == {SubtaskStatus.DONE}


# --------------------------------------------------------------------------
# Warnings: a step that succeeded, but not on the path the operator assumes.
# --------------------------------------------------------------------------


def test_run_view_warning_leaves_the_row_status_alone() -> None:
    """A warning is a caveat on a step, not a transition: colouring the status or inventing
    a fourth one would say the run went worse than it did."""
    view = _seeded_view()
    view.apply(TaskEvent(kind=EventKind.SUBTASK_STARTED, subtask_id="fetch", message="Load it"))

    view.apply(
        TaskEvent(
            kind=EventKind.SUBTASK_WARNING,
            subtask_id="fetch",
            message="Live search was unavailable: HTTP 401.",
        )
    )

    row = view.rows["fetch"]
    assert row.status is SubtaskStatus.RUNNING
    assert row.warning == "Live search was unavailable: HTTP 401."
    assert row.detail == "Load it"  # untouched
    # And only the named row.
    assert view.rows["crunch"].warning == ""


def test_run_view_warning_survives_the_completion_that_follows_it() -> None:
    """Regression: a warning is raised mid-step and the completion lands after it, so one
    folded into `detail` vanished the instant the step succeeded — exactly when the
    operator reads the row to decide whether to trust the answer."""
    view = _seeded_view()
    view.apply(TaskEvent(kind=EventKind.SUBTASK_WARNING, subtask_id="fetch", message="fell back"))

    view.apply(
        TaskEvent(
            kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:fetch.json"
        )
    )

    assert view.rows["fetch"].status is SubtaskStatus.DONE
    assert view.rows["fetch"].warning == "fell back"
    assert view.rows["fetch"].detail == "artifact:fetch.json"


def test_run_table_shows_the_warning_in_place_of_the_detail() -> None:
    """The pointer is in the final report; the degradation is only shown here."""
    view = _seeded_view()
    view.apply(TaskEvent(kind=EventKind.SUBTASK_WARNING, subtask_id="fetch", message="fell back"))
    view.apply(
        TaskEvent(
            kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:fetch.json"
        )
    )

    table = run_table(view)

    assert [str(cell) for cell in table.columns[3].cells] == ["fell back", "", ""]
    # Status still reads as the success it was.
    assert [str(cell) for cell in table.columns[2].cells] == ["done", "pending", "pending"]


def test_run_view_failure_replaces_the_retry_warning_it_follows() -> None:
    """Regression: `run_table` shows the warning in place of the detail, so a subtask that
    failed on its last attempt went red with "Attempt 1 of 2 failed, retrying" rather than
    with why it failed."""
    view = _seeded_view()
    view.apply(
        TaskEvent(
            kind=EventKind.SUBTASK_WARNING,
            subtask_id="fetch",
            message="Attempt 1 of 2 failed, retrying: rate limited",
        )
    )

    view.apply(TaskEvent(kind=EventKind.SUBTASK_FAILED, subtask_id="fetch", message="rate limited"))

    assert view.rows["fetch"].status is SubtaskStatus.FAILED
    assert view.rows["fetch"].warning == ""
    assert [str(cell) for cell in run_table(view).columns[3].cells] == ["rate limited", "", ""]


def test_run_view_a_new_attempt_clears_the_previous_ones_retry_notice() -> None:
    """The notice belongs to the attempt that raised it: once the next one starts, a
    success has to show its pointer rather than a stale "retrying"."""
    view = _seeded_view()
    view.apply(TaskEvent(kind=EventKind.SUBTASK_WARNING, subtask_id="fetch", message="retrying: x"))

    view.apply(_started("fetch"))
    view.apply(
        TaskEvent(
            kind=EventKind.SUBTASK_COMPLETED, subtask_id="fetch", message="artifact:fetch.json"
        )
    )

    assert view.rows["fetch"].warning == ""
    assert view.rows["fetch"].detail == "artifact:fetch.json"


def test_event_line_labels_a_warning_for_the_plain_sink() -> None:
    """Non-TTY runs get the same notice; `--quiet` and `--output json` are the gap."""
    line = event_line(
        TaskEvent(kind=EventKind.SUBTASK_WARNING, subtask_id="fetch", message="fell back")
    )

    assert line.startswith("warn")
    assert "fetch" in line and "fell back" in line


@pytest.mark.asyncio
async def test_spin_holds_still_while_the_region_is_suspended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt's prompt owns the terminal during a pause (#12), so the ticker must not
    write into it. Rich happens to drop a stopped region's refresh too — this pins the
    intent, so nobody removes the guard on the grounds that it looks redundant."""
    monkeypatch.setattr(render, "SPINNER_TICK_SECONDS", 0.001)
    refreshes = 0

    class CountingLive:
        def refresh(self) -> None:
            nonlocal refreshes
            refreshes += 1

    view = _seeded_view()
    view.apply(_started("fetch"))
    region = LiveRegion()
    ticker = asyncio.create_task(render._spin(cast(Live, CountingLive()), view, region))

    with region.suspended():
        await asyncio.sleep(0.02)
        assert refreshes == 0

    await asyncio.sleep(0.02)
    assert refreshes > 0  # and it picks straight back up

    ticker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ticker
