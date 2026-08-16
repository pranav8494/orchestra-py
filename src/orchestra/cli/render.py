"""The live dashboard: the one module that may import `rich.live` (§3.1, §5).

**A model, then a drawing of it.** `RunView` is Rich-free and `run_table`/`event_line`
are pure functions of it, which is what makes §12's "assert on the data handed to the
renderer" practical.

**The renderer never reads state.** Everything it draws arrives as a `TaskEvent`,
including the pending rows (`TaskEvent.plan`). Polling `TaskState` would race the
engine's in-place status writes.

**Progress is a diagnostic, so all of it goes to stderr** (§5). Only the caller's result
goes to stdout, framed in a `Panel` on a terminal only — box characters down a pipe are
the corruption §5 forbids. The mode is `cli/app.py`'s decision; no flag policy here.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum

from rich.console import RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from orchestra.cli.console import err_console
from orchestra.cli.format import OutputFormat, format_result
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, SubtaskStatus, TaskEvent, TaskState

# Shown before `plan_created` arrives: an empty opening frame reads as a hang, and
# planning is the slowest part of a short run.
PLANNING_HEADLINE = "Planning the request"

# Named so a leaked task is identifiable in a task dump, and a constant so a test
# asserting cleanup cannot drift from the string that creates it.
DASHBOARD_TASK_NAME = "dashboard"

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

# Not a `SubtaskStatus` style: the step still finishes `done`, and colouring the status
# would say the run went worse than it did.
_WARNING_STYLE = "yellow"


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
    detail: str = ""
    # Kept out of `detail` because the two arrive in the wrong order: a warning is raised
    # mid-step and the completion that overwrites `detail` lands after it.
    warning: str = ""


@dataclass(slots=True)
class RunView:
    """The run as a subscriber can know it. Rich-free: this is what tests assert on (§12)."""

    headline: str = PLANNING_HEADLINE
    finished: bool = False
    # Insertion order is plan order, seeded from `Plan.subtasks` in one pass. Keyed by id
    # because every later event names a subtask, not a position.
    rows: dict[str, RunRow] = field(default_factory=dict)

    def apply(self, event: TaskEvent) -> None:
        """Fold one event into the view.

        Tolerant by design: an event naming an unknown subtask is ignored, never raised
        on and never turned into a row. It means a subscriber attached after
        `plan_created`; inventing a row would state a role the stream never gave, and
        crashing would lose the dashboard over one frame.
        """
        if event.kind is EventKind.PLAN_CREATED:
            self.headline = event.message
            if event.plan is not None:
                # Rebuilt, not merged: a second plan replaces the first (#3), and merging
                # would keep rows from a plan no longer being executed. Status comes off
                # the event — a resumed run's plan need not be all-pending.
                self.rows = {
                    subtask.id: RunRow(id=subtask.id, role=subtask.role, status=subtask.status)
                    for subtask in event.plan.subtasks
                }
            return

        if event.kind is EventKind.RUN_FINISHED:
            # The engine's own verdict. Recounting the rows here would risk a second,
            # divergent one.
            self.headline = event.message
            self.finished = True
            return

        row = self.rows.get(event.subtask_id or "")
        if row is None:
            return  # an unknown subtask — see the docstring

        if event.kind is EventKind.SUBTASK_WARNING:
            # Status and detail untouched: the step has not transitioned, it has acquired
            # a caveat that outlives whatever it finishes as.
            self.rows[row.id] = replace(row, warning=event.message)
            return

        status = _STATUS_BY_KIND.get(event.kind)
        if status is None:
            return  # an unknown kind — see the docstring
        self.rows[row.id] = replace(row, status=status, detail=event.message)


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


def event_line(event: TaskEvent) -> str:
    """`"<label> <subtask id> <message>"` for the plain sink; the caller prints it.

    The id is omitted for run-level events. An unlabelled kind falls back to its own
    value rather than raising — a renderer must not be what fails when the taxonomy grows.
    """
    label = _EVENT_LABELS.get(event.kind, event.kind.value)
    subject = f"{event.subtask_id} " if event.subtask_id else ""
    return f"{label:<8}{subject}{event.message}"


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
        if not consumer.cancelled() and (failure := consumer.exception()) is not None:
            # Reported, not swallowed (§8), but best-effort: the usual reason the renderer
            # died is that this stream is gone, and a second write would raise again.
            with suppress(OSError):
                err_console.print(
                    f"The dashboard stopped: {failure}", markup=False, highlight=False
                )


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

        # One `Live`, on stderr (§5). `auto_refresh=False` keeps Rich's refresh thread out
        # of the event loop; refreshing per event is cheaper and prompter than its 4/s.
        with Live(run_table(view), console=err_console, auto_refresh=False) as live:

            def redraw(_event: TaskEvent) -> None:
                live.update(run_table(view), refresh=True)

            await _pump(queue, view, redraw)


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
        while not queue.empty():
            _fold(queue.get_nowait(), view, draw)
        raise  # never swallowed (§10)


def _fold(event: TaskEvent, view: RunView, draw: Callable[[TaskEvent], None]) -> None:
    """Update the model, then the screen — `draw` reads the view it was just given."""
    view.apply(event)
    draw(event)


def _write_line(event: TaskEvent) -> None:
    """The plain sink: one line per event on stderr, no region to own or restore.

    markup/highlight off as in `cli/app.py` — a worker's error text can name a bracketed
    token Rich would eat as a style tag.
    """
    err_console.print(event_line(event), markup=False, highlight=False)
