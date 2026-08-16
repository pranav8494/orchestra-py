"""Detects a worker stuck repeating itself: the same call, the same result, again (§3.3).

The output is part of the signature on purpose — the same call returning something new is
progress, and only a call whose result never changes is a loop. A sliding window rather
than a running tally, so a tool legitimately polled early in a subtask cannot trip a step
later on.

Pure counting (§3.2): what a trip *means* is the caller's — `agents/workers/tool_loop.py`
turns it into a `TaskFailure`.
"""

import hashlib
import json
from collections import deque
from collections.abc import Mapping

DEFAULT_WINDOW = 10
DEFAULT_MAX_REPEATS = 5


def step_signature(name: str, arguments: Mapping[str, object], output: str) -> str:
    """Identify one tool step by what was called, with what, and what came back.

    Args:
        name: the tool called.
        arguments: the call's arguments; key order does not affect the result.
        output: what the tool returned, as the model saw it.

    Returns:
        The SHA-256 hex digest of the three parts.
    """
    # One JSON document over the triple: quoting separates the parts, so ("a", "b") and
    # ("ab", "") differ, and `sort_keys` makes argument order irrelevant. `default=str`
    # covers argument values the provider decoded into something JSON has no form for.
    canonical = json.dumps([name, dict(arguments), output], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RepetitionDetector:
    """Counts repeated step signatures over a sliding window. One per subtask.

    Mutable by nature — it is the count — so a class, not a frozen value object (§7).
    """

    def __init__(
        self, *, window: int = DEFAULT_WINDOW, max_repeats: int = DEFAULT_MAX_REPEATS
    ) -> None:
        """Set how far back to look and how much repetition to tolerate.

        Args:
            window: how many recent steps are counted; older ones age out.
            max_repeats: occurrences within the window that are still acceptable.

        Raises:
            ValueError: a non-positive bound — a wiring bug, caught at construction like
                the engine's and the tool loop's.
        """
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        if max_repeats < 1:
            raise ValueError(f"max_repeats must be at least 1, got {max_repeats}")
        self._window = window
        self._max_repeats = max_repeats
        self._recent: deque[str] = deque(maxlen=window)

    @property
    def window(self) -> int:
        """How many recent steps are counted."""
        return self._window

    @property
    def max_repeats(self) -> int:
        """Occurrences of one signature within the window that are still acceptable."""
        return self._max_repeats

    def record(self, signature: str) -> bool:
        """Record one step and report whether it is now repetition.

        Returns:
            True when `signature` occurs more than `max_repeats` times in the window.
            The detector keeps counting either way, so a caller that ignores a trip still
            gets the next one.
        """
        self._recent.append(signature)
        return self._recent.count(signature) > self._max_repeats
