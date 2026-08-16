"""The Analytics agent's primary tool: run a model-written Python script and read its output.

- **A subprocess, not `exec`** — analysis is open-ended, and enumerating its operations as
  tools would be a worse pandas. In-process would put a stray `while True` or `sys.exit()`
  inside the orchestrator; a separate process is the only place a wall-clock kill works.
- **Isolation, not a sandbox** — a scrubbed environment (the parent holds the API keys and
  neither may reach model-written code, §9), a throwaway working directory, `-I`, an empty
  `PATH`, `sandbox.py`'s socket guard, and a SIGKILL to the process group however the call
  ends. That bounds accidents, not malice; containing hostile code needs a container.
- **Every failure is content, not an exception** (§6) — a bad pointer, a traceback and a
  timeout all come back as `ToolResponse(is_error=True)`, as does the environment failing
  underneath (staging and spawning both raise `OSError` once the engine's fan-out exhausts
  descriptors). Only `CancelledError` leaves `run`, and the process group dies with it.
"""

import asyncio
import contextlib
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.tools.base import (
    ToolCall,
    ToolResponse,
    ToolSpec,
    format_validation_error,
)

TOOL_NAME = "run_python"

# Analysis over a handful of artifacts is seconds of pandas; past this it is a loop that
# will not end, not a slow query.
DEFAULT_TIMEOUT = 15.0

# Captured output goes straight into the next prompt, so it is capped like any other
# prompt input. ~1k tokens: enough for a table or a full traceback, small enough that a
# script printing a whole DataFrame cannot evict the conversation.
MAX_OUTPUT_CHARS = 4000

# What is ever read, as opposed to what survives into the prompt. Four bytes is UTF-8's
# widest character, so reaching this always means the script outran the prompt cap; the
# extra byte distinguishes "hit the cap" from "reached EOF just under it".
READ_LIMIT = MAX_OUTPUT_CHARS * 4 + 1

# Generous: after a SIGKILL, EOF is the next thing on the pipe.
DRAIN_TIMEOUT = 1.0

# A killed child's pipes hold at most a buffer each, so discarding them a chunk at a time
# spends none of the memory the cap just saved.
DISCARD_CHUNK = 65536

# Named here because staging writes both files and `sandbox.py`'s argv contract depends on
# `SCRIPT_NAME`.
RUNNER_NAME = "_run.py"
SCRIPT_NAME = "analysis.py"

# A prompt (§6). It leads with the rule that costs a turn when missed — only printed
# output comes back — and states the limits the tool enforces silently, so the model meets
# them by design rather than by retry.
DESCRIPTION = (
    "Run a Python script to compute something: aggregate, join, compare, derive. pandas "
    "and the standard library are available. The script runs in a scratch directory with "
    "the artifacts you named in `inputs` written beside it as plain files, each named "
    "after its pointer — 'artifact:sales.csv' is readable as './sales.csv'. **Only what "
    "the script prints comes back**: end it with print() of the numbers you need, because "
    "the directory and everything left in memory are discarded. Keep that output small: a "
    "script that prints more than a few thousand characters is stopped and loses its "
    "result. There is no network, and the script is killed after 15 seconds, so bound your "
    "loops. If it raises, you get the traceback back — read it, fix the script, and run "
    "again rather than repeating the same call."
)


class RunPythonParams(BaseModel):
    """The arguments the model may send, and the check applied to what it sent.

    Published verbatim as `input_schema`, so every `description` here is prompt text.
    `extra` is forbidden so an invented `timeout` or `packages` comes back as a readable
    validation error instead of being silently dropped (§7).
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        description="The Python script to run. Print everything you want returned.",
    )
    # `list[str]`, not a pointer type: the store already decides what a pointer is and
    # rejects an unsafe name, and a second pattern here would be a second answer (§1.5).
    inputs: list[str] = Field(
        default_factory=list,
        description="Artifact pointers the script needs, e.g. ['artifact:sales.csv']. "
        "Each is written into the working directory under its own name. Empty means none.",
    )


class RunPythonTool:
    """Runs one model-written script per call in a throwaway subprocess. `BaseTool` (§6).

    Stateless past the store and the timeout: every call gets a fresh directory and a
    fresh interpreter, so one script cannot leave state where the next one finds it.
    """

    def __init__(self, store: ArtifactStore, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Take the run's artifact store and the wall clock the child is held to.

        Both injected from `agents/toolsets.py`; nothing here reads config (§3.2).
        `timeout` is injectable so a test can prove the kill without waiting the default
        out, and a non-positive one raises `ValueError` — a wiring bug, so it fails at
        construction like the workers' bounds do.
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be greater than 0, got {timeout}")
        self._store = store
        self._timeout = timeout

    def info(self) -> ToolSpec:
        """See `BaseTool.info`. Pure: no disk is touched, it runs on every turn."""
        return ToolSpec(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=RunPythonParams.model_json_schema(),
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        """Run the script in a scratch directory and return what it printed. See `BaseTool.run`."""
        try:
            params = RunPythonParams.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {format_validation_error(exc)}",
                is_error=True,
            )

        try:
            # The directory is the isolation *and* the cleanup: the child's cwd, HOME and
            # TMPDIR, gone with the call, so a script that writes files leaves nothing for
            # the next one to read.
            with tempfile.TemporaryDirectory(prefix="orchestra-run-python-") as name:
                workdir = Path(name)
                # In a thread: subtasks are dispatched concurrently and copying an
                # artifact of any size must not stall the other agents (§10).
                await asyncio.to_thread(self._stage, workdir, params)
                return await self._execute(workdir)
        except (TaskFailure, OSError) as exc:
            # `TaskFailure` is the store rejecting a pointer — the model's own argument,
            # so correctable. `OSError` is everything underneath (the copies, the spawn
            # hitting EAGAIN or EMFILE under fan-out). Both are the model's to retry, not
            # the agent loop's to unwind (§6).
            return ToolResponse(
                content=f"Could not run {TOOL_NAME}: {exc}. Pass only pointers this run has "
                "already produced, or run the script with no inputs.",
                is_error=True,
            )

    def _stage(self, workdir: Path, params: RunPythonParams) -> None:
        """Lay out the child's working directory. Blocking — call it in a thread.

        Raises:
            TaskFailure: an input is not a pointer, or names nothing in the store.
            OSError: the copies or the write failed. Caught with the above in `run`,
                because to the model they are the same event.
        """
        for pointer in params.inputs:
            # `path_for` validates the pointer and returns the store's own sanitised
            # filename, so the copy cannot be aimed outside `workdir`.
            source = self._store.path_for(pointer)
            shutil.copyfile(source, workdir / source.name)

        # After the artifacts, not before: artifact names are model-chosen and `_run.py`
        # is a legal one, so writing these first would let a data file be executed as the
        # runner. This way a collision costs the input instead, and raises where the
        # script tries to parse itself. No artifact this run can produce hits either name.
        shutil.copyfile(Path(__file__).parent / "sandbox.py", workdir / RUNNER_NAME)
        (workdir / SCRIPT_NAME).write_text(params.code, encoding="utf-8")

    async def _execute(self, workdir: Path) -> ToolResponse:
        """Start the child, hold it to the clock, and turn its exit into a response.

        Raises:
            OSError: the interpreter could not be started, out of processes or descriptors
                under fan-out. Turned into a response by `run` (§6).
        """
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            # No PYTHONPATH, user site-packages or script directory on sys.path. The
            # venv's own site-packages still resolve via sys.prefix, so pandas is
            # importable and nothing of ours is.
            "-I",
            RUNNER_NAME,
            SCRIPT_NAME,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env(workdir),
            # Its own process group, so a script that spawned children dies with them
            # under one `killpg` instead of orphaning them past the timeout.
            start_new_session=True,
        )
        try:
            return await self._collect(process)
        finally:
            # Every path, not only the timeout and the cancellation §10 requires: a script
            # that backgrounded a process with redirected stdio exits cleanly, and the
            # grandchild would go on writing into a directory deleted when `run` returns.
            _kill_group(process)
            await _drain(process)

    async def _collect(self, process: asyncio.subprocess.Process) -> ToolResponse:
        """Read both streams under the clock and turn what came back into a response.

        Bounded reads, not `communicate()`, which has no cap: it pulls each stream into
        parent memory until EOF, so a script printing a frame in a loop costs gigabytes
        for output trimmed to `MAX_OUTPUT_CHARS` anyway.
        """

        async def read(stream: asyncio.StreamReader | None) -> bytes:
            """Read one stream to the cap, killing the group the moment it is hit.

            The kill is here rather than after both reads: the other stream would have no
            reason to reach EOF while the child still holds it, so the pair would sit
            until the timeout.
            """
            raw = await _read_capped(stream)
            if len(raw) >= READ_LIMIT:
                _kill_group(process)
            return raw

        try:
            async with asyncio.timeout(self._timeout):
                stdout, stderr = await asyncio.gather(read(process.stdout), read(process.stderr))
                if len(stdout) >= READ_LIMIT or len(stderr) >= READ_LIMIT:
                    # Stopped mid-script, so the rest of its output is gone: a failure
                    # with a head attached, not a trimmed result. Below the cap the script
                    # finished and `_text` elides only for the prompt, so that stays a
                    # success.
                    return ToolResponse(
                        content="The script printed more than the "
                        f"{MAX_OUTPUT_CHARS} characters this tool can return, so it was "
                        "stopped and the rest of its output is lost. Print the figures you "
                        "need, not whole frames. What it printed first:\n"
                        f"{_text(stdout) or _text(stderr)}",
                        is_error=True,
                    )
                # Both streams are at EOF, which a child only reaches by exiting — the fds
                # are the kernel's to close. So this waits on the exit status alone and
                # cannot block on a pipe nobody is draining.
                await process.wait()
        except TimeoutError:
            return ToolResponse(
                content=f"The script was killed after {self._timeout:g} seconds and its "
                "output was lost — nothing printed before the timeout is recoverable. "
                "Bound every loop, work over the data you were given rather than "
                "generating more, and print only the result.",
                is_error=True,
            )

        if process.returncode != 0:
            # The traceback, not a summary: the model wrote the script, so its own line
            # numbers are what let it fix the thing in one turn.
            detail = _text(stderr) or f"it printed nothing to stderr (exit {process.returncode})"
            return ToolResponse(content=f"The script failed:\n{detail}", is_error=True)

        printed = _text(stdout)
        if not printed.strip():
            # Ran, produced nothing: neither a failure to retry blindly nor a result (§6).
            return ToolResponse(
                content="The script ran without error but printed nothing. Only printed "
                "output is captured — add print() for the values you need.",
                is_empty=True,
            )
        return ToolResponse(content=printed)


def _child_env(workdir: Path) -> dict[str, str]:
    """The child's whole environment, built up rather than filtered down.

    An allow-list, because the parent holds the run's API keys and a deny-list only
    excludes the secrets someone remembered (§9). Nothing is read from `os.environ`.

    - `PATH` empty: the interpreter is started by absolute path, and a `PATH` resolving
      bare names would hand a script `curl` — the network it is told it does not have.
    - `HOME`/`TMPDIR` at the scratch directory, so a library's cache dies with the call.
    - Locale pinned: `-I` implies `-E`, so `PYTHONIOENCODING` is ignored and non-ASCII
      output would otherwise depend on the ambient locale.
    """
    return {
        "PATH": "",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }


def _kill_group(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group, tolerating one that already exited.

    The group, not the process: `start_new_session=True` made the child a group leader, so
    this reaches anything it spawned. SIGKILL, not SIGTERM — a script that ignored the
    clock will not be asked politely.

    Addressed by the child's pid, which `setsid` made the group id, rather than via
    `os.getpgid`: the loop's watcher reaps the child on exit, and a lookup on a reaped pid
    answers for whoever holds it now — likely this orchestrator's own group. Naming the
    group directly narrows a stray signal to a pid both reused *and* made a session leader
    again; short of that `killpg` just says the group is gone, which is why
    `PermissionError` (the same reuse under another uid) is suppressed too.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


async def _drain(process: asyncio.subprocess.Process) -> None:
    """Read the killed child's pipes to EOF and throw the bytes away. Bounded.

    A read that stopped at the cap left its transport paused, holding the descriptor open
    until something reads again — and `Process.wait()` will not, because the disconnect it
    waits for is the one this produces. That is the hang this exists to prevent. Bounded
    anyway: the only thing riding on it is a descriptor the interpreter closes in the end.
    """

    async def discard(stream: asyncio.StreamReader | None) -> None:
        while stream is not None and await stream.read(DISCARD_CHUNK):
            pass

    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(DRAIN_TIMEOUT):
            await asyncio.gather(discard(process.stdout), discard(process.stderr))


async def _read_capped(stream: asyncio.StreamReader | None) -> bytes:
    """Read one stream up to `READ_LIMIT` bytes, and never a byte past it.

    `readexactly`, not `read`, which returns whatever is buffered and would have to be
    looped: here returning *is* the cap being hit and `IncompleteReadError` *is* EOF
    first, so the caller can tell the two apart without counting anything twice.

    `None` cannot happen — both pipes are requested at spawn — but allowing it keeps the
    call site free of a cast.
    """
    if stream is None:
        return b""
    try:
        return await stream.readexactly(READ_LIMIT)
    except asyncio.IncompleteReadError as exc:
        return exc.partial


def _text(raw: bytes) -> str:
    """Decode and cap one captured stream for a prompt.

    `errors="replace"`: a stray byte should cost a mojibake character, not the response.
    """
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    # Same marker shape as `ArtifactStore.preview`, deliberately not shared (§2.3): that
    # elides a file by bytes without reading it, this a stream in memory by characters.
    return (
        f"{text[:MAX_OUTPUT_CHARS]}\n... [elided, {len(text) - MAX_OUTPUT_CHARS} more characters]"
    )
