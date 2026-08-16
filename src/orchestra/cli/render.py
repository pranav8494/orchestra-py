"""The live dashboard: the one module that may import `rich.live` (§3.1, §5).

**A model, then a drawing of it.** `RunView` is the run as the event stream describes
it — no Rich, no I/O, no console — and `run_table`/`event_line` are pure functions of
it. That split is what lets §12's "assert on the data handed to the renderer" be a
practical instruction rather than an aspiration, and it is why the sinks below contain
no logic worth testing through Rich's output.

**The renderer never reads state.** Everything it draws arrives as a `TaskEvent`,
including the pending rows: the engine attaches the plan to `plan_created` precisely so
a subscriber can draw the steps it has not seen run yet (see `TaskEvent.plan`). Polling
`TaskState` would race the engine's in-place status writes and re-introduce the shared
mutable state that copy exists to avoid.

**Progress is a diagnostic, so all of it goes to stderr** (§5). stdout carries one
document a script parses; a `Live` region or a progress line on it would corrupt that.
The only thing this module puts on stdout is the caller's result — and `result_renderable`
wraps it in a `Panel` *only* on a terminal, because box characters down a pipe are
exactly the corruption §5 forbids.

**Three modes, chosen by the caller.** `LIVE` for a terminal, `PLAIN` for a pipe, CI or a
recording, `NONE` for `--quiet` and `--output json`. Which one applies is `cli/app.py`'s
decision — this module holds no policy about flags.
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

# What the headline says before `plan_created` arrives. A dashboard that opens on an
# empty frame reads as a hang; planning is the slowest part of a short run.
PLANNING_HEADLINE = "Planning the request"

# The consumer task's name. Named at all so a leak is identifiable in a traceback or a
# task dump — and a constant so a test asserting the task was cleaned up cannot drift
# from the string that creates it.
DASHBOARD_TASK_NAME = "dashboard"

# The three transitions a subtask row can take. Keyed off `EventKind` and valued with the
# ledger's own `SubtaskStatus` — a second status enum for the table would be a parallel
# abstraction over the same closed set (§1.5), and would drift the first time the engine
# gains a state.
_STATUS_BY_KIND: Mapping[EventKind, SubtaskStatus] = {
    EventKind.SUBTASK_STARTED: SubtaskStatus.RUNNING,
    EventKind.SUBTASK_COMPLETED: SubtaskStatus.DONE,
    EventKind.SUBTASK_FAILED: SubtaskStatus.FAILED,
}

# Colour only, no glyphs: `cli/console.py` already lets Rich drop colour for `NO_COLOR`,
# `TERM=dumb` and pipes, whereas a unicode marker would survive all three and land in a
# CI log as mojibake.
_STATUS_STYLES: Mapping[SubtaskStatus, str] = {
    SubtaskStatus.PENDING: "dim",
    SubtaskStatus.RUNNING: "cyan",
    SubtaskStatus.DONE: "green",
    SubtaskStatus.FAILED: "red",
}

# The plain sink's first column. Shorter than the enum names on purpose — this stream is
# read as a scrolling log, where the verb matters more than the ceremony.
_EVENT_LABELS: Mapping[EventKind, str] = {
    EventKind.PLAN_CREATED: "plan",
    EventKind.SUBTASK_STARTED: "start",
    EventKind.SUBTASK_COMPLETED: "done",
    EventKind.SUBTASK_FAILED: "failed",
    EventKind.RUN_FINISHED: "finish",
}


class RenderMode(StrEnum):
    """How a run is drawn. A closed set, so an enum (§7).

    `NONE` is not "silent LIVE": it attaches nothing at all, so `--quiet` and
    `--output json` cost the publisher neither a queue nor a per-event copy.
    """

    LIVE = "live"
    PLAIN = "plain"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RunRow:
    """One subtask as the stream last described it — one line of the table.

    Frozen (§7): `RunView` swaps a whole row rather than editing one, so a row handed to
    `run_table` can never change underneath the frame being built from it.
    """

    id: str
    role: AgentRole
    status: SubtaskStatus
    detail: str = ""


@dataclass(slots=True)
class RunView:
    """The run, as a subscriber can know it. Rich-free on purpose — this is the thing
    tests assert on (§12), and the drawing functions below are pure functions of it.
    """

    headline: str = PLANNING_HEADLINE
    finished: bool = False
    # Insertion order is plan order, because it is seeded from `Plan.subtasks` in one
    # pass. Keyed by id because every later event names a subtask, not a position.
    rows: dict[str, RunRow] = field(default_factory=dict)

    def apply(self, event: TaskEvent) -> None:
        """Fold one event into the view.

        Tolerant by design: an event naming a subtask no row exists for is **ignored**,
        not raised on and not turned into a new row. The renderer is not a validator —
        the engine is the authority on what the plan contains, and the case that reaches
        here is a subscriber that attached after `plan_created` (or, once replanning
        lands in #3, one holding the previous plan). Inventing a row would put a step in
        the table whose role and position nothing in the stream states; crashing would
        take the dashboard down over a frame. The headline still moves, so a late
        subscriber shows the run's progress rather than a frozen screen.

        Args:
            event: the next lifecycle event off the broker.
        """
        if event.kind is EventKind.PLAN_CREATED:
            self.headline = event.message
            if event.plan is not None:
                # Rebuilt, not merged: a second plan replaces the first (#3), and a
                # merge would leave rows from a plan that is no longer being executed.
                # Status comes off the event rather than being hardcoded PENDING — the
                # event is the authority, and a resumed run's copy need not be all-pending.
                self.rows = {
                    subtask.id: RunRow(id=subtask.id, role=subtask.role, status=subtask.status)
                    for subtask in event.plan.subtasks
                }
            return

        if event.kind is EventKind.RUN_FINISHED:
            # The engine's own count ("N of M subtasks completed", or why it stopped
            # short). Recounting the rows here would state a second, divergent verdict.
            self.headline = event.message
            self.finished = True
            return

        status = _STATUS_BY_KIND.get(event.kind)
        row = self.rows.get(event.subtask_id or "")
        if status is None or row is None:
            return  # an unknown kind, or an unknown subtask — see the docstring
        # `message` is the instruction, the artifact pointer, or the error text,
        # depending on the kind: in every case it is what the row's line should say next.
        self.rows[row.id] = replace(row, status=status, detail=event.message)


def run_table(view: RunView) -> Table:
    """Draw `view` as the dashboard's table. Pure — no console, no I/O.

    Every cell is a `Text`, never a `str`: Rich parses markup in string cells, so an
    error message naming a bracketed token ("[not_found_error]") would have it deleted as
    a style tag — the regression `cli/app.py` already carries a comment about.

    Args:
        view: the run so far.

    Returns:
        A fresh table. Cheap enough to rebuild per event, which is what keeps a
        transition visible well inside the ticket's ~1 s.
    """
    table = Table(title=Text(view.headline), title_justify="left", expand=True)
    table.add_column("Step", no_wrap=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    # Elided rather than wrapped: an instruction or a traceback line that reflowed would
    # change the region's height mid-run, and a Live region that resizes on every event
    # flickers and scrolls the frames above it off the screen.
    table.add_column("Detail", ratio=1, no_wrap=True, overflow="ellipsis")
    for row in view.rows.values():
        table.add_row(
            Text(row.id),
            Text(row.role.value),
            Text(row.status.value, style=_STATUS_STYLES.get(row.status, "")),
            Text(row.detail),
        )
    return table


def event_line(event: TaskEvent) -> str:
    """One line describing `event`, for the plain sink. Pure — the caller prints it.

    Args:
        event: the event just received.

    Returns:
        `"<label> <subtask id> <message>"`, the id omitted for run-level events. An
        unlabelled kind falls back to its own value rather than raising: a renderer must
        not be what fails when the event taxonomy grows.
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
    """The run's final report, ready for `console.print` on stdout.

    The text itself is `cli/format.py`'s, unchanged: what each `--output` mode *says* is
    that module's contract, and re-deriving it here would give the project two answers to
    the same question (§2).

    `terminal` is a parameter rather than a `console.is_terminal` read inside, so both
    arms are testable without a fake console — and so the caller can force the piped
    shape.

    Args:
        state: the finished ledger.
        output: the `--output` mode. JSON is never framed: a `Panel`'s box characters
            around a document would break the first `json.loads` that met them (§5).
        quiet: passed through — it drops the step trace, never the report.
        terminal: whether stdout is a terminal. False means a pipe, a file, or CI, where
            the frame is corruption rather than an affordance.

    Returns:
        A `Panel` for text on a terminal; the bare string otherwise.
    """
    text = format_result(state, output=output, quiet=quiet)
    if output is OutputFormat.TEXT and terminal:
        # `Text`, for the same reason `run_table` uses it: a report quoting a bracketed
        # token must reach the reader with the token in it.
        #
        # `no_wrap`/`overflow` are set explicitly rather than inherited, and that is
        # load-bearing: the stdout console is built `soft_wrap=True` so a long result
        # line survives a pipe unbroken, but soft wrapping means `no_wrap` and
        # `overflow="ignore"`, and inside a panel — which *is* a fixed width — that
        # stops being "don't wrap" and becomes "crop". A step line naming a long
        # artifact then reached the reader with the end of the pointer silently
        # deleted. Stating both here overrides the console's defaults for this
        # renderable only, so the pipe keeps its unbroken lines and the panel folds.
        return Panel(
            Text(text, no_wrap=False, overflow="fold"),
            title="Report",
            title_align="left",
            padding=(1, 2),
        )
    return text


@asynccontextmanager
async def dashboard(broker: Broker[TaskEvent], *, mode: RenderMode) -> AsyncIterator[RunView]:
    """Draw `broker`'s events for the duration of the block.

    Shaped to satisfy `app.RunObserver`, so `functools.partial(dashboard, mode=...)` is
    what `cli/app.py` hands to `run_once`. The yielded `RunView` is there for tests and
    for a caller that wants the model; the observer contract ignores it.

    **Recorded decision (issue #11, first comment): the subscription's lifetime is the
    consumer's lifetime.** `broker.subscribe()` is entered *inside* the consuming task,
    never out here, so any way the consumer stops — cancellation, a bug, teardown —
    detaches the queue on the same unwind. The alternative was evicting timed-out
    subscribers inside `Broker`, and we deliberately did not take it: a renderer that
    stopped draining but stayed attached costs the engine `lifecycle_timeout` (5 s) on
    *every* later lifecycle event, ~2 per subtask, which turns a ten-step run into two
    minutes of stalling. Keeping the two lifetimes identical makes that state
    unreachable without touching the broker's contract for every other subscriber.

    Args:
        broker: the run's event stream.
        mode: how to draw. `NONE` subscribes to nothing at all.

    Yields:
        The view the events are folded into. Empty and never updated under `NONE`.

    Raises:
        asyncio.CancelledError: the caller was cancelled. The consumer is cancelled and
            awaited first, so the `Live` region is always exited — an un-exited one
            corrupts the user's shell (§8) — and the error is then re-raised (§10).
    """
    view = RunView()

    if mode is RenderMode.NONE:
        # Not a silent subscriber: an attached queue still costs the publisher a bounded
        # buffer and a delivery attempt per event, and `--quiet` should cost the run
        # nothing at all.
        yield view
        return

    # Set by the consumer once its queue is attached. Without it, `plan_created` — the
    # only event carrying the plan, and so the only way to draw the pending rows — races
    # the task's first scheduling and is lost about as often as not.
    subscribed = asyncio.Event()
    consumer = asyncio.create_task(
        _consume(broker, view, mode, subscribed), name=DASHBOARD_TASK_NAME
    )
    try:
        # Raced against the consumer rather than awaited outright: a consumer that died
        # before subscribing would otherwise leave this waiting for an event nobody will
        # ever set, hanging the run before it starts. Nothing on that path can raise
        # today — subscribing appends to a list — but the cost of being wrong is a hang,
        # and the guard is one call.
        waiter = asyncio.ensure_future(subscribed.wait())
        try:
            await asyncio.wait({waiter, consumer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        yield view
    finally:
        consumer.cancel()
        # `asyncio.wait`, never `await consumer`. Awaiting the task directly conflates
        # two different cancellations and re-raises a third thing:
        #
        #   * Ours. `await consumer` raises the `CancelledError` we just asked for, and
        #     telling it apart from a real one is not possible from the consumer's state:
        #     when the caller is cancelled *while suspended here*, asyncio delivers that
        #     by cancelling the future being awaited — the consumer — so the consumer ends
        #     up `cancelled()` either way. A guard on `consumer.cancelled()` therefore
        #     reads a genuine Ctrl-C as its own teardown and swallows it, and the run
        #     exits 0 instead of 130 (§8, §10).
        #   * The consumer's own failure. `await` re-raises it, so a broken *diagnostic*
        #     stream — `BrokenPipeError` from `orchestra run ... | head -1` — would
        #     replace the run's result on stdout and its exit code, or mask the error the
        #     body was already raising. Progress must never cost the result (§5).
        #
        # `asyncio.wait` raises only if *we* are cancelled, which is exactly the one case
        # that must propagate. The consumer's outcome is then read deliberately below.
        await asyncio.wait({consumer})
        if not consumer.cancelled() and (failure := consumer.exception()) is not None:
            # Reported, not swallowed (§8) — but on a best-effort basis: the usual reason
            # the renderer died is that this very stream is gone, and a second write
            # would raise from the teardown all over again.
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

    The subscription is entered here rather than in `dashboard` — see that function's
    recorded decision — so this coroutine's unwind is what detaches the queue.
    """
    async with broker.subscribe() as queue:
        subscribed.set()

        if mode is RenderMode.PLAIN:
            await _pump(queue, view, _write_line)
            return

        # One `Live`, on stderr (§5), refreshed only when something changed:
        # `auto_refresh=False` keeps Rich's background refresh thread out of the event
        # loop, and an explicit refresh per event is both cheaper and more prompt than
        # its four-per-second default.
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

    Runs past `run_finished` rather than returning on it: the observer's job is to stay
    attached for the run, and detaching early would mean a late event — a replan's, or a
    second run's — arriving with nobody subscribed.

    The cancellation arm drains what is already queued before re-raising. Teardown
    normally lands one scheduling pass after the engine's `run_finished`, and a last
    frame reading "running" describes a run that is over. Both `get_nowait` and `draw`
    are synchronous, so this cleanup has no await to be interrupted at.
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

    markup and highlight off, as in `cli/app.py`: the message can be a worker's error
    text, and a bracketed token in it would otherwise be parsed as a style tag and
    silently deleted.
    """
    err_console.print(event_line(event), markup=False, highlight=False)
