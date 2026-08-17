"""Tests for the interrupt chat and the live region it borrows the terminal from (#12).

Input is scripted through `stream=`, Rich's own seam, and no test hands anything a real
terminal (§12). Without a tty `_listen` is a no-op, so `requested` stays false and
`session`/`_line_mode` degrade to nothing — that degraded path is the one under test, and
the two writing tests assert stdout and stderr separately as §5 requires.
"""

import asyncio
import io
import os
import sys
import termios
from collections.abc import Callable
from typing import TextIO, cast

import pytest
from rich.live import Live

from conftest import fake_terminal, wait_until
from orchestra.cli.chat import (
    BANNER,
    INTERRUPT_KEY,
    RESUME_COMMANDS,
    RESUMING,
    ConsoleChat,
)
from orchestra.cli.prompt import DECLINED
from orchestra.cli.render import LiveRegion


class DeadStdin:
    """stdin that fails on read: Ctrl-D at the prompt, or the stream closing under it.

    An exhausted `StringIO` cannot stand in — Rich returns the prompt's default before
    `next_message`'s handler is reached. Copied from `test_prompt.py`'s `ClosedStdin`
    rather than shared: one module's edge case, and the copy is three lines (§2.3).
    """

    def __init__(self, failure: type[Exception]) -> None:
        self._failure = failure

    def readline(self) -> str:
        raise self._failure


class FakeLive:
    """A `Live` reduced to the two calls the region makes, recording them in order.

    A real one would take the terminal the suite must never touch (§12).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self, refresh: bool = False) -> None:
        # The flag is recorded too: a restart without a repaint leaves the region blank
        # until the next event redraws it.
        self.calls.append(f"start(refresh={refresh})")

    def stop(self) -> None:
        self.calls.append("stop")


def chat(stream: TextIO | None = None) -> ConsoleChat:
    """A chat over a fresh region, reading `stream` instead of stdin."""
    return ConsoleChat(LiveRegion(), stream=stream)


# ------------------------------------------------------------------ the pause's prompt


@pytest.mark.asyncio
async def test_next_message_returns_the_line_typed() -> None:
    assert await chat(io.StringIO("replan without the chart\n")).next_message() == (
        "replan without the chart"
    )


@pytest.mark.asyncio
async def test_next_message_empty_line_declines() -> None:
    """Enter is how the user resumes: the engine reads `""` as "nothing to act on"."""
    assert await chat(io.StringIO("\n")).next_message() == DECLINED


@pytest.mark.asyncio
@pytest.mark.parametrize("command", sorted(RESUME_COMMANDS))
@pytest.mark.parametrize("spelling", [str.lower, str.upper], ids=["lower", "upper"])
async def test_next_message_resume_command_declines(
    command: str, spelling: Callable[[str], str]
) -> None:
    """Typed by anyone who expects a command; they all mean the bare Enter, in any case."""
    assert await chat(io.StringIO(f"  {spelling(command)}  \n")).next_message() == DECLINED


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [EOFError, OSError])
async def test_next_message_dead_stream_declines(failure: type[Exception]) -> None:
    """A closed stdin costs the interaction, never the run (§5, §8 — no traceback for a
    foreseeable condition)."""
    assert await chat(cast("TextIO", DeadStdin(failure))).next_message() == DECLINED


@pytest.mark.asyncio
async def test_next_message_exhausted_stream_declines() -> None:
    """The same for a stream that ends quietly rather than raising, as a pipe does."""
    assert await chat(io.StringIO("")).next_message() == DECLINED


@pytest.mark.asyncio
async def test_next_message_prompts_on_stderr_and_nothing_on_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§5: stdout carries the run's result alone, so a caret on it would corrupt
    `--output json`. Both streams asserted separately, as §12 requires."""
    await chat(io.StringIO("go on\n")).next_message()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "you" in captured.err


# ------------------------------------------------------------------ the reply


def test_say_writes_to_stderr_and_nothing_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """The conversation is a diagnostic; only the run's result belongs on stdout (§5)."""
    chat().say("replanning now")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "replanning now" in captured.err


def test_say_keeps_a_bracketed_token_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    """`markup=False`: a reply naming a token like this would otherwise be eaten as a
    style tag and vanish from the screen."""
    chat().say("the search returned [not_found_error]")

    assert "[not_found_error]" in capsys.readouterr().err


# ------------------------------------------------------------------ the key and the session


@pytest.mark.asyncio
async def test_requested_without_a_terminal_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """No tty, no key: the run simply never pauses, rather than showing a prompt nobody
    can see. A `StringIO` stdin has no `fileno`, so a `_listen` that tried would raise."""
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    async with chat() as opened:
        assert opened.requested() is False


def test_requested_is_consuming() -> None:
    """One press, one pause: the engine calls this between steps, so a flag left set
    would reopen the chat on every step until the user typed again."""
    opened = chat()
    opened._requested = True

    assert opened.requested() is True
    assert opened.requested() is False


def test_session_announces_the_pause_on_stderr_and_suspends_the_region(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The region has to be down for the block: a dashboard frame redrawn over the prompt
    is exactly the corruption §5 forbids."""
    region = LiveRegion()

    with ConsoleChat(region).session():
        # Read into a local: asserting the property twice narrows its type for the rest
        # of the function, and mypy then reads the second assertion as unreachable.
        inside = region.is_suspended

    assert inside is True
    assert region.is_suspended is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert BANNER in captured.err
    assert RESUMING in captured.err


@pytest.mark.asyncio
async def test_console_chat_context_manager_without_a_terminal_installs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing installed means nothing to leak: no reader on the loop and no saved line
    discipline, so the unwind is a no-op however the run ends (§8)."""
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    loop = asyncio.get_running_loop()
    # A delta, not a count: the loop registers its own self-pipe reader before any test runs.
    registered = len(loop._selector.get_map())  # type: ignore[attr-defined]

    async with chat() as opened:
        assert opened._fd is None
        assert len(loop._selector.get_map()) == registered  # type: ignore[attr-defined]

    assert opened._cooked_mode is None
    assert len(loop._selector.get_map()) == registered  # type: ignore[attr-defined]


# ------------------------------------------------------------------ the live region


def test_live_region_suspended_with_nothing_attached_is_a_no_op() -> None:
    """`--quiet`, a pipe, or a run not yet drawing: the chat needs no branch of its own,
    and the depth still tracks so the spinner knows to stay quiet."""
    region = LiveRegion()

    with region.suspended():
        inside = region.is_suspended  # a local, for the reason `test_session_...` gives

    assert inside is True
    assert region.is_suspended is False


def test_live_region_suspended_stops_the_live_and_starts_it_again() -> None:
    """Restarted with a repaint, or the terminal keeps the frame the prompt scrolled."""
    region, live = LiveRegion(), FakeLive()

    with region.attached(cast("Live", live)), region.suspended():
        assert live.calls == ["stop"]

    assert live.calls == ["stop", "start(refresh=True)"]


def test_live_region_suspended_restarts_the_live_after_an_exception() -> None:
    """A Ctrl-C through a pause must still give the terminal back (§8)."""
    region, live = LiveRegion(), FakeLive()

    attached, suspended = region.attached(cast("Live", live)), region.suspended()
    with pytest.raises(KeyboardInterrupt), attached, suspended:
        raise KeyboardInterrupt

    assert live.calls == ["stop", "start(refresh=True)"]
    assert region.is_suspended is False


def test_live_region_nested_suspends_stop_and_start_once() -> None:
    """Counted, not nested for real: an inner `start` would repaint the dashboard over
    the prompt that asked for the terminal."""
    region, live = LiveRegion(), FakeLive()

    with region.attached(cast("Live", live)), region.suspended():
        with region.suspended():
            assert live.calls == ["stop"]
        assert live.calls == ["stop"]  # the inner exit must not bring the region back up

    assert live.calls == ["stop", "start(refresh=True)"]


def test_live_region_attached_forgets_the_live_on_exit() -> None:
    """A stopped `Live` must not outlive the consumer that owns it — a later pause would
    restart a dead region."""
    region, live = LiveRegion(), FakeLive()

    with region.attached(cast("Live", live)):
        pass
    with region.suspended():
        pass

    assert live.calls == []


# --------------------------------------------------------------------------------------
# The key itself
# --------------------------------------------------------------------------------------
#
# Everything above runs the no-terminal path, where `_listen` never touches a descriptor.
# The keypress is the ticket's first criterion, so it is exercised against `conftest`'s
# `fake_terminal` — a pty of the suite's own, never the developer's (§12).


@pytest.mark.asyncio
async def test_the_interrupt_key_is_seen_and_other_keys_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ticket's first criterion. Any other keystroke is swallowed rather than pausing —
    stdin is unbuffered for the run, so a pass-through would land in the shell afterwards.

    Counting the reads, not polling `requested`: "it has not been pressed" is true before
    the byte arrives as well as after it is ignored, so a poll would pass either way.
    """
    with fake_terminal(monkeypatch) as typed:
        async with ConsoleChat(LiveRegion()) as chat:
            reads: list[None] = []

            def spy() -> None:
                ConsoleChat._on_key(chat)
                reads.append(None)

            monkeypatch.setattr(chat, "_on_key", spy)
            assert chat.requested() is False  # the first call is what arms the reader

            os.write(typed, b"x")
            await wait_until(lambda: len(reads) == 1, what="the stray key to be read")
            assert chat.requested() is False  # read, and ignored

            os.write(typed, INTERRUPT_KEY.encode())
            await wait_until(lambda: len(reads) == 2, what="the interrupt key to be read")
            assert chat.requested() is True


# Only the flags anyone sets are compared — the kernel owns the rest of `lflag`.
_LINE_FLAGS = {"ECHO": termios.ECHO, "ICANON": termios.ICANON}
CBREAK = {"ECHO": False, "ICANON": False}


def line_modes() -> dict[str, bool]:
    """stdin's echo and canonical flags, as a shell would find them."""
    lflag = termios.tcgetattr(sys.stdin.fileno())[3]
    return {name: bool(lflag & bit) for name, bit in _LINE_FLAGS.items()}


@pytest.mark.asyncio
async def test_the_terminal_is_handed_back_in_the_mode_it_was_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§8: leaving stdin in cbreak leaves the user's shell with no echo."""
    with fake_terminal(monkeypatch):
        before = line_modes()
        async with ConsoleChat(LiveRegion()) as chat:
            chat.requested()  # arms the reader, which is what changes the mode
            during = line_modes()
        after = line_modes()

    assert during == CBREAK
    assert after == before


@pytest.mark.asyncio
async def test_the_terminal_is_handed_back_when_an_exception_unwinds_the_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Ctrl-C arrives with cbreak armed and unwinds *through* `__aexit__` (§8,
    #16). The test above exits the block normally, so only this one proves the restore
    happens while an exception is in flight."""
    with fake_terminal(monkeypatch):
        before = line_modes()
        with pytest.raises(KeyboardInterrupt):
            async with ConsoleChat(LiveRegion()) as chat:
                chat.requested()
                armed = line_modes()
                raise KeyboardInterrupt
        after = line_modes()

    assert armed == CBREAK  # the mode really was taken before the unwind
    assert after == before


@pytest.mark.asyncio
async def test_nothing_is_armed_until_the_run_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The planner's clarification prompt (#10) reads a line through `input()` while the run
    is still planning. Arming at construction would clear `ECHO` across it and the user
    would type the answer blind."""
    with fake_terminal(monkeypatch):
        echoing = bool(termios.tcgetattr(sys.stdin.fileno())[3] & termios.ECHO)
        async with ConsoleChat(LiveRegion()):
            still_echoing = bool(termios.tcgetattr(sys.stdin.fileno())[3] & termios.ECHO)

    assert echoing and still_echoing


@pytest.mark.asyncio
async def test_a_second_press_before_the_chat_opens_does_not_queue_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Someone whose first `i` appears to do nothing presses again. Both are spent on the
    pause they opened, or they would be thrown back into a second chat on resuming."""
    with fake_terminal(monkeypatch) as typed:
        async with ConsoleChat(LiveRegion()) as chat:
            chat.requested()
            os.write(typed, INTERRUPT_KEY.encode() * 2)
            await wait_until(chat.requested, what="the interrupt key")

            with chat.session():
                pass

            assert chat.requested() is False
