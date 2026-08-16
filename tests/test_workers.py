"""Tests for the Phase A stub worker.

Its job is to make the engine's contract observable, so that is what is asserted: a
pointer out, the payload in the store, and nothing but the slice used.
"""

import pytest

from orchestra.agents.workers.stub import EchoWorker
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.state import AgentRole, Subtask, TaskState

REQUEST = "Summarize the last 3 quarters' financial trends"


def _state() -> TaskState:
    return TaskState(user_request=REQUEST, artifacts={"fetch": "artifact:fetch.txt"})


def _subtask(**overrides: object) -> Subtask:
    fields: dict[str, object] = {
        "id": "analyse",
        "role": AgentRole.ANALYTICS,
        "instruction": "Compute the quarter-over-quarter trend",
        "inputs": ["fetch"],
        "depends_on": ["fetch"],
    }
    return Subtask.model_validate(fields | overrides)


@pytest.mark.asyncio
async def test_echo_worker_stores_the_instruction_and_returns_its_pointer(
    store: ArtifactStore,
) -> None:
    state = _state()
    subtask = _subtask()

    pointer = await EchoWorker(store).run(state.state_slice(subtask))

    assert pointer == "artifact:analyse.txt"
    stored = store.get_text(pointer)
    assert subtask.instruction in stored
    assert "role: analytics" in stored
    # Inputs arrive as pointers; payloads are never copied into state.
    assert "input fetch: artifact:fetch.txt" in stored


@pytest.mark.asyncio
async def test_echo_worker_rejects_a_subtask_id_that_is_not_a_safe_artifact_name(
    store: ArtifactStore,
) -> None:
    """Ids come from model output, so the name is a trust boundary the store enforces."""
    state = TaskState(user_request=REQUEST)
    subtask = _subtask(id="../escape", inputs=[], depends_on=[])

    with pytest.raises(TaskFailure, match="Unsafe artifact name"):
        await EchoWorker(store).run(state.state_slice(subtask))
