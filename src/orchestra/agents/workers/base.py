"""The `Worker` port every role agent implements (CONVENTIONS.md §6, §7).

One method, and it takes a `SubtaskContext` rather than the ledger: a worker sees its
subtask and its declared inputs, never the plan or another agent's artifacts (§6).

**Pointers out, not payloads.** A worker writes its output to the artifact store and
returns the pointer; the engine records that pointer in `TaskState.artifacts`. Returning
the payload instead would put it in state, and state is serialised into a prompt.
"""

from typing import Protocol

from orchestra.core.state import ArtifactPointer, SubtaskContext


class Worker(Protocol):
    """A role agent. Implemented once per role, plus the Phase A `EchoWorker`."""

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Perform one subtask and return a pointer to what it produced.

        Args:
            context: the worker's slice of the ledger, from `TaskState.state_slice()`.

        Returns:
            The pointer to the stored output, which the engine writes back to state.

        Raises:
            TaskFailure: the subtask could not be completed. The engine records the
                subtask as failed and continues with the rest of the plan, so this
                ends one step, not the run. Tool-level failures never get this far —
                they come back as `ToolResponse(is_error=True)` for the model to read
                and retry (§6).
            asyncio.CancelledError: the run was cancelled; propagated, never swallowed.
        """
        ...
