"""The Phase A stand-in: echo the instruction into an artifact, return the pointer.

Makes the engine's contract testable with no model, network or cost. Real workers (#5-#7)
implement the same `Worker` port and drop in one at a time.

The text is artifact payload, not a prompt, so it does not belong in `prompts/` (§11).
"""

import asyncio

from orchestra.artifacts import ArtifactStore
from orchestra.core.state import ArtifactPointer, SubtaskContext


class EchoWorker:
    """Writes its instruction and inputs to the artifact store and returns the pointer."""

    def __init__(self, store: ArtifactStore) -> None:
        """Store the injected artifact store (§3.3 — nothing is constructed here)."""
        self._store = store

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Record the subtask as text. See `Worker.run`."""
        subtask = context.subtask
        lines = [
            f"role: {subtask.role.value}",
            f"instruction: {subtask.instruction}",
            *(f"input {name}: {pointer}" for name, pointer in sorted(context.inputs.items())),
        ]
        # `to_thread` because the store is blocking I/O; blocking the loop would serialise
        # the concurrent dispatch this worker exists to prove (§10).
        return await asyncio.to_thread(self._store.put_text, f"{subtask.id}.txt", "\n".join(lines))
