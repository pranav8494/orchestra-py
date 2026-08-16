"""Tests for the Analytics agent's subprocess Python executor (CONVENTIONS.md §12).

These really do start interpreters — the thing under test *is* the process boundary, and
a mocked `create_subprocess_exec` would prove nothing about the kill, the scrubbed
environment or the network guard. To stay fast every script is stdlib-only (importing
pandas would cost a second per test), the timeout test injects its own clock rather than
waiting the default 15 seconds out, and nothing here can reach the network: the guard
under test is what stops it.

The cancellation test has the child touch a marker file before it sleeps, so the cancel
lands while `communicate()` is genuinely in flight rather than during staging.
"""

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from orchestra.artifacts import ArtifactStore
from orchestra.tools.base import BaseTool, ToolCall
from orchestra.tools.python_exec import (
    MAX_OUTPUT_CHARS,
    TOOL_NAME,
    RunPythonTool,
)

# Ceiling on every wait in this file. Long enough that a loaded machine does not flake,
# short enough that a swallowed cancellation fails the suite instead of hanging it.
TIMEOUT = 5.0

# The clock the timeout test holds the child to. Above interpreter start-up on a loaded
# machine, far below the 15 s default — the point is to prove the kill, quickly.
SHORT_TIMEOUT = 1.0


def call(name: str, **arguments: object) -> ToolCall:
    """One tool call as a provider would decode it."""
    return ToolCall(id="call-1", name=name, arguments=arguments)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def tool(store: ArtifactStore) -> RunPythonTool:
    return RunPythonTool(store)


def wait_for(path: Path) -> bool:
    """Poll until the child creates `path`. Blocking — await it through `to_thread`.

    Polling because the signal crosses a process boundary, where an `asyncio.Event` does
    not reach; in a thread so the loop stays free to run the child's transport.
    """
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return False


if TYPE_CHECKING:
    # Conformance to the port is mypy's job, not `isinstance`'s: `BaseTool` is a plain
    # Protocol, and a runtime check would compare attribute names only (§7).
    _IS_A_TOOL: BaseTool = RunPythonTool(ArtifactStore(Path("artifacts")))


def test_info_advertises_its_params_schema(tool: RunPythonTool) -> None:
    """The published shape, not the expression `info()` returns — this crosses to the model."""
    spec = tool.info()

    assert spec.name == TOOL_NAME
    properties = spec.input_schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"code", "inputs"}


def test_run_python_rejects_a_non_positive_timeout(store: ArtifactStore) -> None:
    """A wiring bug fails at construction, not four minutes into a run (§9)."""
    with pytest.raises(ValueError, match="timeout"):
        RunPythonTool(store, timeout=0)


@pytest.mark.asyncio
async def test_run_python_returns_what_the_script_printed(tool: RunPythonTool) -> None:
    response = await tool.run(call(TOOL_NAME, code="print(6 * 7)"))

    assert not response.is_error
    assert not response.is_empty
    assert response.content.strip() == "42"


@pytest.mark.asyncio
async def test_run_python_writes_each_input_into_the_working_directory(
    tool: RunPythonTool, store: ArtifactStore
) -> None:
    """The contract the description promises: `artifact:sales.csv` is readable as `./sales.csv`."""
    pointer = store.put_text("sales.csv", "quarter,revenue\n2025Q1,120\n")

    response = await tool.run(
        call(
            TOOL_NAME,
            code="print(open('sales.csv').read().strip().splitlines()[-1])",
            inputs=[pointer],
        )
    )

    assert not response.is_error
    assert response.content.strip() == "2025Q1,120"


@pytest.mark.parametrize(
    "pointer", ["artifact:absent.csv", "sales.csv"], ids=["missing", "malformed"]
)
@pytest.mark.asyncio
async def test_run_python_bad_pointer_returns_an_error_not_an_exception(
    tool: RunPythonTool, pointer: str
) -> None:
    """The store raises `TaskFailure`; §6 says the model gets to read it and retry."""
    response = await tool.run(call(TOOL_NAME, code="print(1)", inputs=[pointer]))

    assert response.is_error
    assert pointer in response.content


@pytest.mark.asyncio
async def test_run_python_failing_script_returns_its_own_traceback(tool: RunPythonTool) -> None:
    """`run_path` in the child is what makes this name the model's file and line, not ours."""
    response = await tool.run(call(TOOL_NAME, code="total = 1 / 0\n"))

    assert response.is_error
    assert "ZeroDivisionError" in response.content
    assert "analysis.py" in response.content


@pytest.mark.asyncio
async def test_run_python_infinite_loop_is_killed_at_the_timeout(store: ArtifactStore) -> None:
    """The ticket's acceptance criterion, on an injected clock so the suite stays fast."""
    tool = RunPythonTool(store, timeout=SHORT_TIMEOUT)

    response = await asyncio.wait_for(
        tool.run(call(TOOL_NAME, code="while True:\n    pass\n")), timeout=TIMEOUT
    )

    assert response.is_error
    assert f"{SHORT_TIMEOUT:g} seconds" in response.content


@pytest.mark.asyncio
async def test_run_python_cannot_open_a_socket(tool: RunPythonTool) -> None:
    """The guard `sandbox.py` installs, proven from inside the child (§12: no network)."""
    response = await tool.run(call(TOOL_NAME, code="import socket\nsocket.socket()\n"))

    assert response.is_error
    assert "network access is disabled" in response.content


@pytest.mark.asyncio
async def test_run_python_child_environment_carries_no_parent_secret(
    tool: RunPythonTool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9: the parent holds the run's API keys and model-written code never sees them."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leaky")

    response = await tool.run(
        call(TOOL_NAME, code="import os\nprint(os.environ.get('ANTHROPIC_API_KEY'))\n")
    )

    assert not response.is_error
    assert "leaky" not in response.content
    assert response.content.strip() == "None"


@pytest.mark.asyncio
async def test_run_python_unknown_argument_is_an_error(tool: RunPythonTool) -> None:
    """`extra="forbid"`: an invented argument is corrected, not silently dropped (§7)."""
    response = await tool.run(call(TOOL_NAME, code="print(1)", timeout=1))

    assert response.is_error
    assert "timeout" in response.content


@pytest.mark.asyncio
async def test_run_python_missing_code_is_an_error(tool: RunPythonTool) -> None:
    response = await tool.run(call(TOOL_NAME, inputs=[]))

    assert response.is_error
    assert "code" in response.content


@pytest.mark.asyncio
async def test_run_python_script_that_prints_nothing_is_empty_not_failed(
    tool: RunPythonTool,
) -> None:
    """Ran, produced nothing: the third outcome, so nobody retries a script that worked."""
    response = await tool.run(call(TOOL_NAME, code="total = 1 + 1\n"))

    assert response.is_empty
    assert not response.is_error
    assert "print" in response.content


@pytest.mark.asyncio
async def test_run_python_truncates_output_that_would_flood_the_prompt(tool: RunPythonTool) -> None:
    """This text goes into the next prompt, so the cap is part of the contract, not a nicety."""
    response = await tool.run(call(TOOL_NAME, code=f"print('x' * {MAX_OUTPUT_CHARS * 2})"))

    assert not response.is_error
    assert "[elided," in response.content
    assert len(response.content) < MAX_OUTPUT_CHARS * 2


@pytest.mark.asyncio
async def test_run_python_propagates_cancellation(tool: RunPythonTool, tmp_path: Path) -> None:
    """§10/§12: `CancelledError` is the only thing that may leave `run`.

    The marker puts the cancel inside `communicate()` rather than in staging, so what is
    under test is the handler that kills the process group and re-raises. A handler that
    returned a `ToolResponse` instead fails here rather than hanging.
    """
    marker = tmp_path / "child-started"
    code = (
        f"import pathlib, time\npathlib.Path({str(marker)!r}).touch()\ntime.sleep({TIMEOUT * 4})\n"
    )
    task = asyncio.create_task(tool.run(call(TOOL_NAME, code=code)))

    assert await asyncio.to_thread(wait_for, marker)  # the child is inside `communicate()`
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TIMEOUT)
