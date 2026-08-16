"""The live dashboard: the one module that may import `rich.live` (§3.1, §5).

**A model, then a drawing of it.** `RunView` is Rich-free and `run_table`/`active_panel`/
`event_log`/`event_line` are pure functions of it, which is what makes §12's "assert on
the data handed to the renderer" practical. `dashboard_frame` composes them into the
region `Live` owns.

**The renderer never reads state.** Everything it draws arrives as a `TaskEvent`,
including the pending rows (`TaskEvent.plan`). Polling `TaskState` would race the
engine's in-place status writes.

**Progress is a diagnostic, so all of it goes to stderr** (§5). Only the caller's result
goes to stdout, framed in a `Panel` on a terminal only — box characters down a pipe are
the corruption §5 forbids. The mode is `cli/app.py`'s decision; no flag policy here.
"""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum

from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from orchestra.cli.console import err_console
from orchestra.cli.format import OutputFormat, format_result
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, SubtaskStatus, TaskEvent, TaskState

# Printed before `plan_created` arrives, above the region rather than in it (see
# `_consume`): planning is the slowest part of a short run, and an empty opening frame
# reads as a hang. Also `RunView`'s opening headline, which only the plain sink now draws.
PLANNING_HEADLINE = "Planning the request"

# Named so a leaked task is identifiable in a task dump, and a constant so a test
# asserting cleanup cannot drift from the string that creates it.
DASHBOARD_TASK_NAME = "dashboard"
SPINNER_TASK_NAME = "dashboard-spinner"

# How many events the log keeps. Bounded because a run publishes unboundedly many and the
# region has to fit a terminal; dropping the oldest is what makes it scroll.
EVENT_LOG_LINES = 8

# ASCII frames (`-\|/`), not the default braille: this is the one animated glyph on
# screen, and #31 has a unicode one raising `UnicodeEncodeError` on a non-UTF-8 stream.
SPINNER_NAME = "line"

# Rich advances a spinner only when the region is redrawn, and lifecycle events arrive
# seconds apart. Half `SPINNER_NAME`'s own 130 ms frame interval: ticking on the frame
# boundary aliases against it and the animation stutters.
SPINNER_TICK_SECONDS = 0.065

# Valued with the ledger's own `SubtaskStatus`: a second status enum for the table would
# parallel an existing closed set (§1.5) and drift when the engine gains a state.
_STATUS_BY_KIND: Mapping[EventKind, SubtaskStatus] = {
    EventKind.SUBTASK_STARTED: SubtaskStatus.RUNNING,
    EventKind.SUBTASK_COMPLETED: SubtaskStatus.DONE,
    EventKind.SUBTASK_FAILED: SubtaskStatus.FAILED,
}

# Colour only, no glyphs: Rich drops colour for `NO_COLOR`, `TERM=dumb` and pipes,
# whereas a unicode marker survives all three and lands in a CI log as mojibake.
_STATUS_STYLES: Mapping[SubtaskStatus, str] = {
    SubtaskStatus.PENDING: "dim",
    SubtaskStatus.RUNNING: "cyan",
    SubtaskStatus.DONE: "green",
    SubtaskStatus.FAILED: "red",
}

# The plain sink's first column. Shorter than the enum names: read as a scrolling log.
_EVENT_LABELS: Mapping[EventKind, str] = {
    EventKind.PLAN_CREATED: "plan",
    EventKind.SUBTASK_STARTED: "start",
    EventKind.SUBTASK_WARNING: "warn",
    EventKind.SUBTASK_COMPLETED: "done",
    EventKind.SUBTASK_FAILED: "failed",
    EventKind.RUN_FINISHED: "finish",
}

# Not a `SubtaskStatus` style: a warning is a caveat on a step that has not transitioned
# yet, and colouring the status would pre-empt the outcome.
_WARNING_STYLE = "yellow"

# What the active panel says while the run is between steps — see `active_panel`.
_IDLE_LINE = "waiting for the next step"

# What each part costs in rows beyond its content, for `dashboard_frame`'s height budget.
# A panel is two borders; `run_table` adds its title, the header and the rule under it.
# Counted, not measured: `Console.measure` gives only a width, and rendering a frame to
# learn how tall it is before deciding what to put in it is a loop.
_PANEL_CHROME = 2
_TABLE_CHROME = 5


def _one_line(message: str) -> str:
    """`message` with its whitespace collapsed onto one line.

    Messages here are model-written or `str(exc)`, and `str(ValidationError)` is multi-line
    by construction. An embedded newline defeats every `no_wrap` in this module: it adds a
    *row* to the region rather than a wrap inside a cell.
    """
    return " ".join(message.split())


class RenderMode(StrEnum):
    """How a run is drawn. A closed set, so an enum (§7).

    `NONE` is not "silent LIVE": it attaches nothing, so `--quiet` and `--output json`
    cost the publisher neither a queue nor a per-event copy.
    """

    LIVE = "live"
    PLAIN = "plain"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RunRow:
    """One subtask as the stream last described it — one line of the table.

    Frozen (§7): `RunView` swaps a whole row rather than editing one, so a row can never
    change underneath the frame being built from it.
    """

    id: str
    role: AgentRole
    status: SubtaskStatus
    # From the plan, so the active panel can name the work before the step starts. It
    # duplicates `detail` on running rows today — deliberately (§2.3): reading the
    # instruction off an event message would break the day the engine words it differently.
    instruction: str = ""
    detail: str = ""
    # Kept out of `detail` because the two arrive in the wrong order: a warning is raised
    # mid-step and the completion that overwrites `detail` lands after it.
    warning: str = ""


@dataclass(slots=True)
class RunView:
    """The run as a subscriber can know it. Rich-free: this is what tests assert on (§12)."""

    headline: str = PLANNING_HEADLINE
    # The engine's verdict arrived.
    finished: bool = False
    # The stream detached: cancelled, or torn down around a run that raised. Distinct from
    # `finished` — there is no verdict, only an end.
    stopped: bool = False
    # Insertion order is plan order, seeded from `Plan.subtasks` in one pass. Keyed by id
    # because every later event names a subtask, not a position.
    rows: dict[str, RunRow] = field(default_factory=dict)
    # The events themselves, in arrival order, not lines formatted from them: the model
    # stays presentation-free and `event_line` is applied when it is drawn.
    log: deque[TaskEvent] = field(default_factory=lambda: deque(maxlen=EVENT_LOG_LINES))

    @property
    def active(self) -> list[RunRow]:
        """The rows the stream last reported as running — one spinner each.

        A list, not a count: concurrent subtasks have to be individually visible (#17).
        """
        return [row for row in self.rows.values() if row.status is SubtaskStatus.RUNNING]

    @property
    def resting(self) -> bool:
        """Is the run over, as far as this view can know?

        Nothing animates past this: on Ctrl-C the rows still read `running` — cancelled,
        not failed — so a spinner in the last painted frame would claim work that stopped
        happening (#39).
        """
        return self.finished or self.stopped

    def apply(self, event: TaskEvent) -> None:
        """Fold one event into the view.

        Tolerant by design: an event naming an unknown subtask is ignored, never raised
        on and never turned into a row. It means a subscriber attached after
        `plan_created`; inventing a row would state a role the stream never gave, and
        crashing would lose the dashboard over one frame.
        """
        # Before the row lookup: an event naming an unknown subtask still happened, and
        # the log is the one place it can be seen.
        self.log.append(event)

        if event.kind is EventKind.PLAN_CREATED:
            self.headline = _one_line(event.message)
            if event.plan is not None:
                # Rebuilt, not merged: a second plan replaces the first (#3), and merging
                # would keep rows from a plan no longer being executed. Status comes off
                # the event — a resumed run's plan need not be all-pending.
                self.rows = {
                    subtask.id: RunRow(
                        id=subtask.id,
                        role=subtask.role,
                        status=subtask.status,
                        instruction=subtask.instruction,
                    )
                    for subtask in event.plan.subtasks
                }
            return

        if event.kind is EventKind.RUN_FINISHED:
            # The engine's own verdict. Recounting the rows here would risk a second,
            # divergent one.
            self.headline = _one_line(event.message)
            self.finished = True
            return

        row = self.rows.get(event.subtask_id or "")
        if row is None:
            return  # an unknown subtask — see the docstring

        if event.kind is EventKind.SUBTASK_WARNING:
            # Status and detail untouched: the step has not transitioned, it has acquired
            # a caveat that outlives whatever it finishes as.
            self.rows[row.id] = replace(row, warning=_one_line(event.message))
            return

        status = _STATUS_BY_KIND.get(event.kind)
        if status is None:
            return  # an unknown kind — see the docstring
        # `run_table` shows a warning *instead of* the detail, so a retry notice kept past
        # its attempt hides the failure message on a red row. Only completion carries one
        # forward: a degradation arrives mid-step, after the start that would clear it.
        self.rows[row.id] = replace(
            row,
            status=status,
            detail=_one_line(event.message),
            warning=row.warning if event.kind is EventKind.SUBTASK_COMPLETED else "",
        )


def run_table(view: RunView) -> Table:
    """Draw `view` as the dashboard's table. Pure — no console, no I/O.

    Every cell is a `Text`, never a `str`: Rich parses markup in string cells, so an
    error naming a bracketed token ("[not_found_error]") would lose it as a style tag.
    A fresh table per event is cheap enough to keep a transition inside the ~1 s target.
    """
    table = Table(title=Text(view.headline), title_justify="left", expand=True)
    table.add_column("Step", no_wrap=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    # Elided rather than wrapped: a reflowed line changes the region's height mid-run, and
    # a `Live` that resizes per event flickers and scrolls the frames above it off screen.
    table.add_column("Detail", ratio=1, no_wrap=True, overflow="ellipsis")
    for row in view.rows.values():
        table.add_row(
            Text(row.id),
            Text(row.role.value),
            Text(row.status.value, style=_STATUS_STYLES.get(row.status, "")),
            # The warning wins the cell: `detail` is the artifact pointer, which the final
            # report prints anyway, whereas the caveat is only visible here.
            Text(row.warning, style=_WARNING_STYLE) if row.warning else Text(row.detail),
        )
    return table


def active_panel(view: RunView) -> Panel:
    """The working agents, one spinner each. Pure — no console, no I/O.

    Instruction rather than status: the table already says "running", and a fan-out has to
    show *which* agent is doing *what*.

    A grid, not `Spinner(text=...)`: `Spinner.render` rebuilds its text through
    `Text.assemble`, which drops the `no_wrap` a long instruction needs, so a wrapped row
    resizes the region and leaves the glyph on the first line only.
    """
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True)  # the spinner
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    for row in view.active:
        grid.add_row(
            Spinner(SPINNER_NAME, style=_STATUS_STYLES[SubtaskStatus.RUNNING]),
            # `Text`, as in `run_table`: an instruction quoting a bracketed token must
            # survive Rich's markup parser.
            Text(f"{row.role.value}  {row.instruction}"),
        )
    if not view.active:
        # A row, not an absent panel: every handoff on a sequential plan has a frame with
        # nothing running, and a panel that comes and goes resizes the region each time.
        grid.add_row(Text(""), Text(_IDLE_LINE, style=_STATUS_STYLES[SubtaskStatus.PENDING]))
    return Panel(grid, title="Active", title_align="left")


def event_log(view: RunView, *, lines: int = EVENT_LOG_LINES) -> Panel:
    """The last `lines` events, oldest first. Pure — no console, no I/O.

    Elided, not wrapped, for the same reason as the table's detail column — and that only
    holds because `event_line` collapses the message to one line first.
    """
    shown = list(view.log)[-lines:] if lines else []
    return Panel(
        Text("\n".join(event_line(event) for event in shown), no_wrap=True, overflow="ellipsis"),
        title="Events",
        title_align="left",
    )


def dashboard_frame(view: RunView, *, height: int | None = None) -> RenderableType:
    """The whole `Live` region: the plan, who is working, and what just happened.

    The panels appear once there is a run to describe and then stay between steps; the
    active one goes for good once the run is `resting`, since the last frame must not
    claim work that stopped.

    `height` is the terminal's, when there is one: each part is bounded on its own but the
    sum is not, so a tall plan overran a short terminal and Rich cropped the frame from the
    bottom (#39). The log is what gives way — the table is the deliverable. `None` leaves
    it unbounded.
    """
    parts: list[RenderableType] = [run_table(view)]
    spent = len(view.rows) + _TABLE_CHROME
    if view.rows and not view.resting:
        parts.append(active_panel(view))
        spent += max(1, len(view.active)) + _PANEL_CHROME
    if view.log:
        budget = EVENT_LOG_LINES if height is None else height - spent - _PANEL_CHROME
        if (lines := max(0, min(EVENT_LOG_LINES, budget))) > 0:
            parts.append(event_log(view, lines=lines))
    return Group(*parts)


def event_line(event: TaskEvent) -> str:
    """`"<label> <subtask id> <message>"` for the plain sink and the event log.

    The id is omitted for run-level events. An unlabelled kind falls back to its own
    value rather than raising — a renderer must not be what fails when the taxonomy grows.
    """
    label = _EVENT_LABELS.get(event.kind, event.kind.value)
    subject = f"{event.subtask_id} " if event.subtask_id else ""
    return f"{label:<8}{subject}{_one_line(event.message)}"


def result_renderable(
    state: TaskState,
    *,
    output: OutputFormat,
    quiet: bool,
    terminal: bool,
) -> RenderableType:
    """The run's final report for `console.print` on stdout: a `Panel` for text on a
    terminal, the bare string otherwise.

    The text is `cli/format.py`'s, unchanged — re-deriving it here would give the project
    two answers to one question (§2). JSON is never framed: a `Panel`'s box characters
    would break the first `json.loads` that met them (§5).

    `terminal` is a parameter rather than a `console.is_terminal` read inside, so both
    arms are testable without a fake console.
    """
    text = format_result(state, output=output, quiet=quiet)
    if output is OutputFormat.TEXT and terminal:
        # `Text`, as in `run_table`: a report quoting a bracketed token must keep it.
        # `no_wrap`/`overflow` are stated, not inherited — the stdout console's
        # `soft_wrap=True` implies `overflow="ignore"`, which inside a panel's fixed width
        # cropped long artifact pointers. Overriding here leaves the piped shape alone.
        return Panel(
            Text(text, no_wrap=False, overflow="fold"),
            title="Report",
            title_align="left",
            padding=(1, 2),
        )
    return text


@asynccontextmanager
async def dashboard(broker: Broker[TaskEvent], *, mode: RenderMode) -> AsyncIterator[RunView]:
    """Draw `broker`'s events for the duration of the block, yielding the view they fold
    into (empty under `NONE`).

    Shaped to satisfy `app.RunObserver`, so `cli/app.py` hands `run_once` a
    `functools.partial(dashboard, mode=...)`.

    **Recorded decision (issue #11): the subscription's lifetime is the consumer's
    lifetime.** `broker.subscribe()` is entered *inside* the consuming task, so any way
    the consumer stops detaches the queue on the same unwind. Evicting timed-out
    subscribers in `Broker` was declined — an attached renderer that stopped draining
    costs the engine 5 s per later lifecycle event, and this makes that unreachable
    without changing `Broker` for every other subscriber.

    On cancellation the consumer is torn down first, so the `Live` region is always
    exited (§8), then `CancelledError` is re-raised (§10).
    """
    view = RunView()

    if mode is RenderMode.NONE:
        # Not a silent subscriber: an attached queue costs the publisher a buffer and a
        # delivery attempt per event, and `--quiet` should cost the run nothing.
        yield view
        return

    # Without this, `plan_created` — the only event carrying the plan, and so the only way
    # to draw pending rows — races the consumer's first scheduling and is often lost.
    subscribed = asyncio.Event()
    consumer = asyncio.create_task(
        _consume(broker, view, mode, subscribed), name=DASHBOARD_TASK_NAME
    )
    try:
        # Raced against the consumer, not awaited outright: a consumer that died before
        # subscribing would hang the run here forever. Unreachable today; guard is a line.
        waiter = asyncio.ensure_future(subscribed.wait())
        try:
            await asyncio.wait({waiter, consumer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        yield view
    finally:
        consumer.cancel()
        # `asyncio.wait`, never `await consumer`: awaiting re-raises the consumer's
        # outcome, which would let a dead renderer replace the run's result (§5) and would
        # make a real Ctrl-C indistinguishable from our own cancel — exit 0, not 130
        # (§8, §10). `asyncio.wait` propagates only our cancellation; the consumer's
        # outcome is read deliberately below.
        await asyncio.wait({consumer})
        _report_failure(consumer)


def _report_failure(task: asyncio.Task[None]) -> None:
    """Say that a renderer task died, if it did. Never raises.

    Retrieval matters as much as the message: an unread exception is one asyncio logs at
    collection, and `Task.cancel()` clears even that, so a task cancelled after it had
    already raised would vanish entirely (§8).

    Best-effort output: the usual reason a renderer dies is that this stream is gone, and
    a second write would raise again.
    """
    # `done()` first: `exception()` on a task still running raises `InvalidStateError`.
    if not task.done() or task.cancelled() or (failure := task.exception()) is None:
        return
    with suppress(OSError):
        err_console.print(f"The dashboard stopped: {failure}", markup=False, highlight=False)


async def _consume(
    broker: Broker[TaskEvent],
    view: RunView,
    mode: RenderMode,
    subscribed: asyncio.Event,
) -> None:
    """Subscribe, then draw every event until cancelled. Runs as its own task.

    The subscription is entered here, not in `dashboard` — see its recorded decision —
    so this coroutine's unwind is what detaches the queue.
    """
    async with broker.subscribe() as queue:
        subscribed.set()

        if mode is RenderMode.PLAIN:
            await _pump(queue, view, _write_line)
            return

        # The region must not own the terminal before the first event: the planner's
        # clarification round (#10) prompts on this stream while it is still planning, and
        # a `Live` up at that moment leaves the cursor inside it. `plan_created` is that
        # first event, published once planning has settled either way.
        err_console.print(PLANNING_HEADLINE)
        try:
            view.apply(await queue.get())
        except asyncio.CancelledError:
            # `_pump`'s contract, one phase earlier: a run torn down before it drew
            # anything still folds what it was sent, or the view describes the moment the
            # region would have opened rather than the run.
            _drain(queue, view, _undrawn)
            raise

        # Read per frame, not once: a terminal resized mid-run has to be picked up, and
        # `Console.size` is an `ioctl` against a run whose steps take seconds.
        def frame() -> RenderableType:
            return dashboard_frame(view, height=err_console.size.height)

        # One `Live`, on stderr (§5). `auto_refresh=False` keeps Rich's refresh thread out
        # of the event loop; refreshing per event is cheaper and prompter than its 4/s.
        with Live(frame(), console=err_console, auto_refresh=False) as live:

            def redraw(_event: TaskEvent) -> None:
                live.update(frame(), refresh=True)

            spinner = asyncio.create_task(_spin(live, view), name=SPINNER_TASK_NAME)
            try:
                await _pump(queue, view, redraw)
            finally:
                # One last frame, first: the stream is detached, so a spinner left in the
                # region Ctrl-C freezes would claim work that stopped (#39). Best-effort,
                # like every other write here.
                view.stopped = True
                with suppress(OSError):
                    live.update(frame(), refresh=True)
                # Read before the cancel, for the reason `_report_failure` gives. Inside
                # the `with` because Rich prints above a live region, and bounded by it so
                # no task outlives the consumer that owns it (§10).
                _report_failure(spinner)
                spinner.cancel()
                await asyncio.wait({spinner})


async def _spin(live: Live, view: RunView) -> None:
    """Redraw while an agent is working, so the spinners advance between events.

    A task on our own loop rather than `Live(auto_refresh=True)`, which runs Rich's refresh
    thread beside the event loop.

    A closed stderr is dropped rather than ending the ticker: `orchestra run ... | head -1`
    must cost the animation, never the result (§5). Anything else propagates to
    `_report_failure`.
    """
    while True:
        await asyncio.sleep(SPINNER_TICK_SECONDS)
        if view.active and not view.resting:
            with suppress(OSError):
                live.refresh()


async def _pump(
    queue: asyncio.Queue[TaskEvent],
    view: RunView,
    draw: Callable[[TaskEvent], None],
) -> None:
    """Fold every event into `view` and hand it to `draw`, until cancelled.

    Runs past `run_finished`: detaching there would leave a late event — a replan's, or a
    second run's — with nobody subscribed. The cancellation arm drains what is queued
    first, or a last frame reading "running" would describe a finished run; `get_nowait`
    and `draw` are sync, so the drain has no await to be cut at.
    """
    try:
        while True:
            _fold(await queue.get(), view, draw)
    except asyncio.CancelledError:
        _drain(queue, view, draw)
        raise  # never swallowed (§10)


def _drain(
    queue: asyncio.Queue[TaskEvent], view: RunView, draw: Callable[[TaskEvent], None]
) -> None:
    """Fold everything already queued. Sync throughout, so cancellation cannot cut it."""
    while not queue.empty():
        _fold(queue.get_nowait(), view, draw)


def _fold(event: TaskEvent, view: RunView, draw: Callable[[TaskEvent], None]) -> None:
    """Update the model, then the screen — `draw` reads the view it was just given."""
    view.apply(event)
    draw(event)


def _undrawn(_event: TaskEvent) -> None:
    """Draw nothing: no region has opened yet, so folding is all a late event can get."""


def _write_line(event: TaskEvent) -> None:
    """The plain sink: one line per event on stderr, no region to own or restore.

    markup/highlight off as in `cli/app.py` — a worker's error text can name a bracketed
    token Rich would eat as a style tag.
    """
    err_console.print(event_line(event), markup=False, highlight=False)
