"""The `Worker` port every role agent implements (§6, §7).

Takes a `SubtaskContext`, not the ledger: a worker sees its subtask and its declared
inputs, never the plan or another agent's artifacts (§6).

**Pointers out, not payloads.** Returning the payload would put it in state, and state is
serialised into a prompt.
"""

from typing import Protocol

from orchestra.core.state import ArtifactPointer, SubtaskContext


class Worker(Protocol):
    """A role agent. Implemented once per role, plus the Phase A `EchoWorker`."""

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Perform one subtask and return a pointer to what it produced.

        Args:
            context: the worker's slice of the ledger, from `TaskState.state_slice()`.

        Raises:
            TaskFailure: the subtask could not be completed. Ends one step, not the run —
                the engine marks it failed and continues. Tool-level failures never get
                this far; they come back as `ToolResponse(is_error=True)` (§6).
            asyncio.CancelledError: propagated, never swallowed.
        """
        ...
