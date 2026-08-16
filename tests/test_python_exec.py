"""Tests for the Analytics agent's subprocess Python executor (CONVENTIONS.md §12).

These really do start interpreters — the thing under test *is* the process boundary, and
a mocked `create_subprocess_exec` would prove nothing about the kill, the scrubbed
environment or the network guard. To stay fast every script is stdlib-only (importing
pandas would cost a second per test), the timeout test injects its own clock rather than
waiting the default 15 seconds out, and nothing here can reach the network: the guard
under test is what stops it.

The two tests that mock do so *at* the boundary rather than across it — `OSError` from
the spawn and from a copy is what the machine does under fan-out, and there is no way to
run a real one out of descriptors on purpose.

The cancellation test has the child touch a marker file before it sleeps, so the cancel
lands while the reads are genuinely in flight rather than during staging.
"""

import asyncio
import os
import shutil
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

# A script that writes far more than the tool will ever keep, as fast as the pipe takes
# it, and then never stops. Both halves matter: the flood is what used to leave the read
# transport paused, and the loop is what used to make the reap after the kill wait for a
# pipe disconnect that could no longer happen.
FLOOD_THEN_LOOP = "import sys\nchunk = 'x' * 100000\nwhile True:\n    sys.stdout.write(chunk)\n"


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


def wait_until_gone(pid: int) -> bool:
    """Poll until `pid` can no longer be signalled. Blocking — await it in a thread.

    Signal 0 is the liveness question without the side effect. Polled rather than waited
    on because a grandchild is not this process's child to reap: once the group dies it
    is init's, and init's reap is what finally frees the pid.
    """
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
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
async def test_run_python_infinite_loop_that_floods_stdout_is_stopped(tool: RunPythonTool) -> None:
    """The case the kill test above cannot reach: `while True: pass` prints nothing.

    A script that writes faster than the parent reads leaves the stream over its limit
    and the read transport paused, and the reap that followed the kill then waited on a
    pipe disconnect nothing could deliver — the child died, the parent did not, and the
    subtask, its `TaskGroup` and the run hung with it. On the *default* clock, so a
    regression fails here in seconds instead of quietly returning at the timeout.
    """
    response = await asyncio.wait_for(
        tool.run(call(TOOL_NAME, code=FLOOD_THEN_LOOP)), timeout=TIMEOUT
    )

    assert response.is_error
    assert "printed more than" in response.content


@pytest.mark.asyncio
async def test_run_python_output_past_the_cap_is_never_buffered_whole(tool: RunPythonTool) -> None:
    """Reading to EOF cost the parent the script's whole output, per concurrent subtask.

    300 MB printed drove peak RSS up by nearly a gigabyte in a quarter of a second, and
    the cap only trimmed what was already in memory. 20 MB here, which the read now stops
    a few kilobytes into: the response costs the same whether the script printed 20 KB or
    20 GB.
    """
    code = "import sys\nfor _ in range(200):\n    sys.stdout.write('x' * 100000)\n"

    response = await asyncio.wait_for(tool.run(call(TOOL_NAME, code=code)), timeout=TIMEOUT)

    assert response.is_error
    assert "[elided," in response.content
    assert len(response.content) < MAX_OUTPUT_CHARS * 2


@pytest.mark.asyncio
async def test_run_python_kills_a_background_process_the_script_left_running(
    tool: RunPythonTool,
) -> None:
    """The group is killed on success too, not only on the timeout and the cancel.

    A grandchild with its own stdio does not hold the pipes open, so the script returns
    cleanly — and used to leave the grandchild writing into a working directory that is
    deleted the moment `run` returns.
    """
    code = (
        "import subprocess\nimport sys\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\nprint(child.pid)\n"
    )

    response = await asyncio.wait_for(tool.run(call(TOOL_NAME, code=code)), timeout=TIMEOUT)

    assert not response.is_error
    assert await asyncio.to_thread(wait_until_gone, int(response.content))


@pytest.mark.asyncio
async def test_run_python_cannot_connect_a_socket(tool: RunPythonTool) -> None:
    """The guard `sandbox.py` installs, proven from inside the child (§12: no network)."""
    response = await tool.run(
        call(TOOL_NAME, code="import socket\nsocket.socket().connect(('127.0.0.1', 9))\n")
    )

    assert response.is_error
    assert "network access is disabled" in response.content


@pytest.mark.asyncio
async def test_run_python_can_import_urllib_and_still_cannot_reach_anything(
    tool: RunPythonTool,
) -> None:
    """The guard patches the connect methods, not the `socket.socket` class.

    `ssl` does `class SSLSocket(socket)`, so a class replaced by a function turned the
    next `import urllib.request` into `TypeError: function() argument 'code' must be
    code, not str` — a fault the model did not cause, cannot fix, and would spend its one
    self-correction turn on. The address is the discard port on loopback: if the guard
    ever fails, this test refuses a connection rather than making one.
    """
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://127.0.0.1:9/', timeout=1)\n"
        "except OSError as exc:\n"
        "    print(f'refused: {exc}')\n"
    )

    response = await tool.run(call(TOOL_NAME, code=code))

    assert not response.is_error
    assert "network access is disabled" in response.content


@pytest.mark.asyncio
async def test_run_python_child_has_no_search_path_to_shell_out_with(tool: RunPythonTool) -> None:
    """`PATH=os.defpath` resolved `curl`, which is the network the description denies."""
    code = "import os\nimport shutil\nprint(os.environ['PATH'] or 'empty', shutil.which('curl'))\n"

    response = await tool.run(call(TOOL_NAME, code=code))

    assert not response.is_error
    assert response.content.strip() == "empty None"


@pytest.mark.asyncio
async def test_run_python_spawn_failure_is_returned_as_data(
    tool: RunPythonTool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6: EAGAIN or EMFILE under fan-out is the model's to retry, not the loop's to unwind."""

    async def refuse(*_args: object, **_kwargs: object) -> asyncio.subprocess.Process:
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)

    response = await tool.run(call(TOOL_NAME, code="print(1)"))

    assert response.is_error
    assert "Too many open files" in response.content


@pytest.mark.asyncio
async def test_run_python_staging_failure_is_returned_as_data(
    tool: RunPythonTool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same for the filesystem under the scratch directory: data, not an exception."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copyfile", refuse)

    response = await tool.run(call(TOOL_NAME, code="print(1)"))

    assert response.is_error
    assert "No space left on device" in response.content


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

    The marker puts the cancel inside the reads rather than in staging, so what is under
    test is the cleanup that kills the process group and lets the error through. A
    handler that returned a `ToolResponse` instead fails here rather than hanging.
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
