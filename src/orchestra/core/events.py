"""The typed broker: one publisher, many subscribers, two delivery modes (§3.3, §6).

Progress may be dropped — blocking the agent loop on a slow renderer costs more than a
lost frame — so `publish_progress` is sync and no caller can await it into a stall. A
lifecycle event is a state transition: a dropped `subtask_completed` strands the
dashboard on a spinner, so those block, bounded by a timeout. Drops are counted.

Generic over the event type so the renderer (#11) can publish its own without a second
broker growing beside this one (§1.5).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

DEFAULT_QUEUE_SIZE = 128
DEFAULT_LIFECYCLE_TIMEOUT = 5.0


class Broker[EventT]:
    """Fan-out of events to every current subscriber.

    One instance per run, constructed in `app.py` and injected — not a singleton (§3.3).
    """

    def __init__(
        self,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        lifecycle_timeout: float = DEFAULT_LIFECYCLE_TIMEOUT,
    ) -> None:
        """Configure the fan-out.

        Args:
            queue_size: per-subscriber buffer. Full means progress is dropped.
            lifecycle_timeout: seconds to wait for one subscriber before giving up on it.
        """
        self._queue_size = queue_size
        self._lifecycle_timeout = lifecycle_timeout
        self._subscribers: list[asyncio.Queue[EventT]] = []
        self._dropped_progress = 0
        self._dropped_lifecycle = 0

    @property
    def subscriber_count(self) -> int:
        """How many subscribers are currently attached."""
        return len(self._subscribers)

    @property
    def dropped_progress(self) -> int:
        """Progress events discarded because a subscriber's queue was full."""
        return self._dropped_progress

    @property
    def dropped_lifecycle(self) -> int:
        """Lifecycle events a subscriber failed to accept within the timeout.

        Non-zero means a missed state transition — a defect, not the routine
        backpressure `dropped_progress` counts.
        """
        return self._dropped_lifecycle

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[EventT]]:
        """Attach a bounded queue for the duration of the block.

        A context manager, not a pair of calls: §6 requires unsubscribing on
        cancellation, and every later lifecycle publish would pay the full timeout
        waiting on an abandoned queue nobody drains.
        """
        queue: asyncio.Queue[EventT] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(queue)
        try:
            yield queue
        finally:
            self._subscribers.remove(queue)

    def publish_progress(self, event: EventT) -> None:
        """Offer `event` to every subscriber, dropping it for any that is full.

        Synchronous by design (§6): no await to "fix", so a slow subscriber can never
        stall the agent loop reporting progress.
        """
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._dropped_progress += 1

    async def publish_lifecycle(self, event: EventT) -> None:
        """Deliver `event` to every subscriber, waiting up to the timeout for each.

        A subscriber still full at the deadline is passed over and counted: one wedged
        dashboard must not hang the run.
        """
        # Snapshot: a subscriber may detach while we are awaiting another's queue.
        for queue in list(self._subscribers):
            try:
                async with asyncio.timeout(self._lifecycle_timeout):
                    await queue.put(event)
            except TimeoutError:
                self._dropped_lifecycle += 1
