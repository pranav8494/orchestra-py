"""The ports for pausing a run and talking to the orchestrator mid-flight (§3.3, #12).

Two, because two layers answer them. `Chat` is the user's end, implemented by
`cli/chat.py`, which owns the terminal; `Interrupter` is the orchestrator's, implemented
by `agents/interrupt.py`, which owns what a message does to the plan. The engine holds
only the second and never learns there is a keyboard.

The decision itself is model output, so its enum lives with the agent that parses it —
as `PlannerAction` does.
"""

from contextlib import AbstractContextManager
from typing import Protocol

from orchestra.core.state import TaskState


class Chat(Protocol):
    """The user's end of a mid-run conversation. Implemented by `cli/chat.py` (§7).

    A port, so `agents/` never learns the messages come from a terminal — and a test can
    answer from a list.
    """

    def requested(self) -> bool:
        """Has the user asked to interrupt since this was last called?

        Consuming: a request is honoured once. Cheap and non-blocking — the engine asks
        on every lap of its scheduler loop.
        """
        ...

    def session(self) -> AbstractContextManager[None]:
        """Hold the terminal for one pause, releasing it on exit.

        A scope rather than a pair of calls: a live region has to be put down and picked
        back up, and a conversation that raises must not leave it down.
        """
        ...

    async def next_message(self) -> str:
        """The user's next message, or `""` when they are done and the run should resume.

        Never raises for an unusable answer — only cancellation leaves it.
        """
        ...

    def say(self, text: str) -> None:
        """Show the orchestrator's reply. Best-effort: a dead stream costs the reply only."""
        ...


class Interrupter(Protocol):
    """Whoever can pause the engine and act on what the user says (§7).

    Implemented by `agents/interrupt.py`. The engine holds this and nothing else, so
    replanning stays out of the scheduler.
    """

    def pending(self) -> bool:
        """Is a pause waiting to be honoured? Consuming, like `Chat.requested`."""
        ...

    async def handle(self, state: TaskState) -> None:
        """Run one pause to its end, applying whatever the user settled on to `state`.

        Called with nothing in flight, so the ledger is settled before it is reshaped.
        """
        ...
