"""Tests for the typed broker (§6).

The contract is what happens when a subscriber stops draining: progress is dropped,
lifecycle waits then gives up, and neither stalls the publisher.
"""

import asyncio

import pytest

from orchestra.core.events import Broker
from orchestra.core.state import EventKind, TaskEvent


def _event(kind: EventKind = EventKind.SUBTASK_STARTED) -> TaskEvent:
    return TaskEvent(kind=kind, message="fetch")


@pytest.mark.asyncio
async def test_publish_lifecycle_reaches_every_subscriber() -> None:
    broker: Broker[TaskEvent] = Broker()
    event = _event()

    async with broker.subscribe() as first, broker.subscribe() as second:
        await broker.publish_lifecycle(event)

        assert first.get_nowait() is event
        assert second.get_nowait() is event
    assert broker.dropped_lifecycle == 0


@pytest.mark.asyncio
async def test_publish_progress_reaches_every_subscriber() -> None:
    broker: Broker[TaskEvent] = Broker()
    event = _event()

    async with broker.subscribe() as queue:
        broker.publish_progress(event)

        assert queue.get_nowait() is event


@pytest.mark.asyncio
async def test_publish_progress_drops_for_a_full_subscriber_instead_of_blocking() -> None:
    """§6: progress is lossy on purpose — a slow renderer must not stall the agent loop."""
    broker: Broker[TaskEvent] = Broker(queue_size=1)

    async with broker.subscribe() as queue:
        broker.publish_progress(_event())
        broker.publish_progress(_event())  # returns; no await here to stall on

        assert queue.qsize() == 1
        assert broker.dropped_progress == 1


@pytest.mark.asyncio
async def test_publish_lifecycle_gives_up_on_a_wedged_subscriber() -> None:
    """A dashboard that stopped draining must not hang the run — bounded blocking (§6)."""
    broker: Broker[TaskEvent] = Broker(queue_size=1, lifecycle_timeout=0.01)

    async with broker.subscribe() as queue:
        await broker.publish_lifecycle(_event())
        await broker.publish_lifecycle(_event(EventKind.SUBTASK_COMPLETED))

        assert queue.qsize() == 1
        # Counted, not silent: a missed transition is a defect, unlike a dropped frame.
        assert broker.dropped_lifecycle == 1


@pytest.mark.asyncio
async def test_publish_lifecycle_still_reaches_the_subscribers_after_a_wedged_one() -> None:
    broker: Broker[TaskEvent] = Broker(queue_size=1, lifecycle_timeout=0.01)
    event = _event(EventKind.RUN_FINISHED)

    # Subscribed first, so the publish waits on the wedged queue first.
    async with broker.subscribe() as wedged, broker.subscribe() as healthy:
        await broker.publish_lifecycle(_event())  # fills both, capacity being 1
        healthy.get_nowait()  # only this one keeps up

        await broker.publish_lifecycle(event)

        assert healthy.get_nowait() is event
        assert wedged.qsize() == 1
        assert broker.dropped_lifecycle == 1


@pytest.mark.asyncio
async def test_subscribe_detaches_on_cancellation() -> None:
    """§6: an abandoned queue costs every later publish the full lifecycle timeout."""
    broker: Broker[TaskEvent] = Broker()
    attached = asyncio.Event()

    async def listen() -> None:
        async with broker.subscribe() as queue:
            attached.set()
            await queue.get()  # never arrives

    task = asyncio.create_task(listen())
    await attached.wait()
    assert broker.subscriber_count == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert broker.subscriber_count == 0


@pytest.mark.asyncio
async def test_publishing_without_subscribers_is_a_no_op() -> None:
    """The engine publishes whether or not a dashboard is attached."""
    broker: Broker[TaskEvent] = Broker()

    broker.publish_progress(_event())
    await broker.publish_lifecycle(_event())

    assert (broker.dropped_progress, broker.dropped_lifecycle) == (0, 0)
