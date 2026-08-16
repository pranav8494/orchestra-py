"""The terminal end of a mid-run interrupt: watch for `i`, then host the chat (#12).

- **The one module that takes stdin out of line mode.** A single keystroke has to arrive
  without an Enter, so the run holds the terminal in cbreak and hands the line discipline
  straight back for the conversation itself.
- **The conversation is a diagnostic** — stderr out, stdin in, as `cli/prompt.py`'s
  clarification is, so a piped run still has only its result on stdout (§5). No `Console`
  is constructed here.
- **No terminal, no key.** Without a TTY, or on a platform with no `termios`, `requested`
  is permanently false and the run simply never pauses — better than a prompt nobody can
  see (§5).
"""

import asyncio
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from types import TracebackType
from typing import Any, TextIO

from rich.prompt import Prompt
from rich.text import Text

from orchestra.cli.console import err_console
from orchestra.cli.prompt import DECLINED
from orchestra.cli.render import LiveRegion

# POSIX only. Guarded by `sys.platform` rather than `try: import`, which mypy reads as an
# untyped name from there on.
if sys.platform != "win32":
    import termios
    import tty

# One letter, and one Rich would not swallow: `q` and `c` already mean quit and cancel to
# anyone at a terminal.
INTERRUPT_KEY = "i"

HINT = f"Press {INTERRUPT_KEY} to interrupt the run and talk to the orchestrator."
BANNER = "Paused. Type a message, or press Enter to resume."
RESUMING = "Resuming."

# What the caret says, and who the replies come from.
_YOU = "you"
_ORCHESTRATOR = "orchestrator"

# Typed instead of a bare Enter, for anyone who expects a command. `DECLINED` is what they
# all mean, so `cli/prompt.py`'s one spelling for "nothing to act on" is reused (§1.5).
RESUME_COMMANDS = frozenset({"/resume", "/exit", "/quit"})


class ConsoleChat:
    """Puts a run's pause on the terminal. Implements `core.interrupt.Chat` (§7).

    An async context manager: the key reader is installed for the run and removed on the
    way out, however the run ends. Ctrl-C unwinds through it, so the terminal is always
    given back (§8).
    """

    def __init__(self, region: LiveRegion, *, stream: TextIO | None = None) -> None:
        """Take the live region to suspend during a pause.

        `stream` is Rich's own seam for scripted input (`Prompt.ask(stream=...)`). `None` —
        the default, and the only value production passes — reads real stdin; tests hand in
        a `StringIO`, so the suite never touches a terminal (§12).
        """
        self._region = region
        self._stream = stream
        self._requested = False
        # Set together, and only on a terminal: both `None` means there is no key to press.
        self._fd: int | None = None
        # `Any` because the element type is `termios`' own — an opaque handle we only ever
        # hand back to `tcsetattr`. Permitted here; §7 bans it inward of `cli/`.
        self._cooked_mode: list[Any] | None = None

    async def __aenter__(self) -> "ConsoleChat":
        """Start watching for the key, if there is a terminal to watch."""
        self._listen()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop watching and give the line discipline back. Never raises."""
        self._deafen()

    def requested(self) -> bool:
        """Has the key been pressed since this was last called? Consuming."""
        pressed, self._requested = self._requested, False
        return pressed

    @contextmanager
    def session(self) -> Iterator[None]:
        """Hold the terminal for one pause: region down, stdin back in line mode."""
        with self._region.suspended(), self._line_mode():
            self._announce(BANNER)
            try:
                yield
            finally:
                # Best-effort, and in a `finally`: stderr dying here must cost the notice,
                # never unwind the run it was resuming (§5).
                self._announce(RESUMING)

    async def next_message(self) -> str:
        """The user's next line, or `""` when they resume.

        Blocking, for the reason `cli/prompt.ConsoleAsker.ask` gives: `asyncio.to_thread`
        runs it on a non-daemon thread, and a Ctrl-C at the prompt could then no longer end
        the process. Safe here because the engine drains before it pauses — the only other
        task on the loop is the dashboard consumer, parked on its queue with its region down.
        """
        try:
            answer = Prompt.ask(
                Text(_YOU),
                console=err_console,
                default=DECLINED,
                show_default=False,
                stream=self._stream,
            )
        except (EOFError, OSError):
            # stdin closed — a pipe, or Ctrl-D — or stderr went away mid-prompt. Resuming,
            # not a crash: a dead stream costs the interaction, never the run (§5).
            return DECLINED
        entry = answer.strip()
        return DECLINED if entry.lower() in RESUME_COMMANDS else entry

    def say(self, text: str) -> None:
        """Show the orchestrator's reply."""
        self._announce(f"{_ORCHESTRATOR}: {text}")

    @staticmethod
    def _announce(text: str) -> None:
        """Write one line of the conversation to stderr.

        markup/highlight off as elsewhere: a reply naming a bracketed token must survive
        Rich's markup parser. Best-effort — a dead stream costs the line, never the run (§5).
        """
        with suppress(OSError):
            err_console.print(text, markup=False, highlight=False)

    def _listen(self) -> None:
        """Put stdin in cbreak and read it from the event loop, on a terminal only.

        Both streams have to be a terminal, as for `cli/app._asker`: the key is read from
        stdin and everything the pause shows is written to stderr.
        """
        if sys.platform == "win32" or not sys.stdin.isatty() or not err_console.is_terminal:
            return
        fd = sys.stdin.fileno()
        self._cooked_mode = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._fd = fd
        asyncio.get_running_loop().add_reader(fd, self._on_key)

    def _deafen(self) -> None:
        """Undo `_listen`. Idempotent, so an unwind through a pause is harmless."""
        fd, cooked = self._fd, self._cooked_mode
        if fd is None or cooked is None:
            return
        self._fd, self._cooked_mode = None, None
        asyncio.get_running_loop().remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, cooked)

    def _on_key(self) -> None:
        """Read the waiting byte and note an interrupt if it is the key. Never raises.

        Every other keystroke is swallowed deliberately: stdin is unbuffered for the run's
        duration, so anything not consumed here would land in the shell after it exits.
        """
        fd = self._fd
        if fd is None:
            return
        with suppress(OSError):
            key = os.read(fd, 1).decode(errors="ignore")
            self._requested = self._requested or key.lower() == INTERRUPT_KEY

    @contextmanager
    def _line_mode(self) -> Iterator[None]:
        """Give stdin its line discipline back for the block, and take it away after.

        The reader goes with it: `Prompt.ask` reads the same descriptor, and two readers on
        one terminal split the user's line between them. `setcbreak` flushes on the way
        back, so what was typed during the chat is not replayed as a second interrupt.
        """
        fd, cooked = self._fd, self._cooked_mode
        if fd is None or cooked is None:
            yield
            return
        loop = asyncio.get_running_loop()
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, cooked)
        try:
            yield
        finally:
            tty.setcbreak(fd)
            loop.add_reader(fd, self._on_key)
