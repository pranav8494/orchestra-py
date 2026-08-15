"""The Phase A stand-in: echo the instruction into an artifact, return the pointer.

Its job is to make the engine's contract testable before any real agent exists — the
DAG walk, the state slice, the pointer write-back and the event stream are all
exercised end to end with no model, no network and no cost. Real workers (#5-#7)
implement the same `Worker` port and drop in one at a time.

Not a prompt, so nothing here belongs in `prompts/` (§11): the text is the artifact
payload a downstream step would read, written in the shape a real worker's output takes.
"""

import asyncio

from orchestra.artifacts import ArtifactStore
from orchestra.core.state import ArtifactPointer, SubtaskContext


class EchoWorker:
    """Writes its instruction and inputs to the artifact store and returns the pointer."""

    def __init__(self, store: ArtifactStore) -> None:
        """Store the injected artifact store (§3.3 — nothing is constructed here).

        Args:
            store: the run's artifact store, from `app.py`.
        """
        self._store = store

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Record the subtask as text. See `Worker.run`."""
        subtask = context.subtask
        lines = [
            f"role: {subtask.role.value}",
            f"instruction: {subtask.instruction}",
            *(f"input {name}: {pointer}" for name, pointer in sorted(context.inputs.items())),
        ]
        # `to_thread` because the store is synchronous filesystem I/O and blocking the
        # event loop would serialise the concurrent dispatch this worker exists to prove.
        return await asyncio.to_thread(self._store.put_text, f"{subtask.id}.txt", "\n".join(lines))
