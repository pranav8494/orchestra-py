"""The Analytics agent's primary tool: run a model-written Python script and read its output.

**Why a subprocess and not `exec`.** Analysis is open-ended — the model has to be able to
reshape, join and aggregate whatever the retrieval step produced, and enumerating those
operations as tools would be a worse pandas. Running the code in-process instead would
put a stray `while True` or a `sys.exit()` inside the orchestrator, so the script gets its
own interpreter: a separate process is the only place a wall-clock kill actually works.

**Isolation, not a sandbox.** The child gets a scrubbed environment (the parent holds
`ANTHROPIC_API_KEY` and `TAVILY_API_KEY`, and neither may reach model-written code — §9),
a throwaway working directory, `-I`, a socket guard from `sandbox.py`, and a SIGKILL to
its whole process group when it overruns. That bounds accidents, not malice; containing
hostile code is out of scope for this ticket and would need a container, not a monkeypatch.

**Every failure is content, not an exception** (§6). A bad pointer, a traceback and a
timeout all come back as `ToolResponse(is_error=True)` with the next step spelled out —
the model's own traceback is the most useful thing this tool ever returns, because the
next turn is the fix. The only thing that leaves `run` is `CancelledError`, and the
process group dies with it so a cancelled run leaves nothing behind.
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

# The ticket's wall clock. Analysis over a handful of artifacts is seconds of pandas;
# anything past this is a loop that will not end, not a slow query.
DEFAULT_TIMEOUT = 15.0

# Captured output goes straight into the next prompt, so it is capped like any other
# prompt input. ~1k tokens: enough for a table or a full traceback, small enough that a
# script printing a whole DataFrame cannot evict the conversation.
MAX_OUTPUT_CHARS = 4000

# What the child is started as, and what it is pointed at. Named here because staging
# writes both files and `sandbox.py`'s argv contract depends on the second.
RUNNER_NAME = "_run.py"
SCRIPT_NAME = "analysis.py"

# The description is a prompt (§6), not a docstring. It leads with the one rule that
# costs a turn when missed — only printed output comes back — and states the limits the
# tool enforces silently, so the model meets them by design rather than by retry.
DESCRIPTION = (
    "Run a Python script to compute something: aggregate, join, compare, derive. pandas "
    "and the standard library are available. The script runs in a scratch directory with "
    "the artifacts you named in `inputs` written beside it as plain files, each named "
    "after its pointer — 'artifact:sales.csv' is readable as './sales.csv'. **Only what "
    "the script prints comes back**: end it with print() of the numbers you need, because "
    "the directory and everything left in memory are discarded. There is no network, and "
    "the script is killed after 15 seconds, so bound your loops. If it raises, you get the "
    "traceback back — read it, fix the script, and run again rather than repeating the "
    "same call."
)


class RunPythonParams(BaseModel):
    """The arguments the model may send, and the check applied to what it sent.

    Published verbatim as the tool's `input_schema`, so every `description` here is prompt
    text read while the model chooses arguments. `extra` is forbidden so an invented
    `timeout` or `packages` argument comes back as a readable validation error instead of
    being silently dropped (§7).
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        description="The Python script to run. Print everything you want returned.",
    )
    # `list[str]`, not `list[ArtifactPointer]`: the store already decides what a pointer
    # is and rejects an unsafe name, and a second pattern here would be a second answer to
    # that question (§1.5). A malformed pointer comes back from `run` naming itself.
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

        Args:
            store: the run's artifact store, used to resolve `inputs`. Injected from
                `agents/toolsets.py`; nothing here reads config (§3.2).
            timeout: seconds a script may run before its process group is killed.
                Injectable so a test can prove the kill without waiting the default out.

        Raises:
            ValueError: a non-positive timeout — a wiring bug, not a user-facing error,
                so it fails at construction like the workers' bounds do.
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

        # The directory is the isolation *and* the cleanup: it is the child's cwd, its
        # HOME and its TMPDIR, and it goes away with the call, so a script that writes
        # files leaves nothing for the next one to read.
        with tempfile.TemporaryDirectory(prefix="orchestra-run-python-") as name:
            workdir = Path(name)
            try:
                # In a thread: the engine dispatches subtasks concurrently, and copying
                # an artifact of any size must not stall the other agents (§10).
                await asyncio.to_thread(self._stage, workdir, params)
            except TaskFailure as exc:
                # The store raises this for a malformed or missing pointer. It ends the
                # run when the engine sees it, but here it is the model's own argument,
                # so it comes back as data for it to correct (§6).
                return ToolResponse(
                    content=f"Could not read the inputs for {TOOL_NAME}: {exc}. Pass only "
                    "pointers this run has already produced, or run the script with no "
                    "inputs.",
                    is_error=True,
                )
            return await self._execute(workdir)

    def _stage(self, workdir: Path, params: RunPythonParams) -> None:
        """Lay out the child's working directory. Blocking — call it in a thread.

        Raises:
            TaskFailure: an input is not a pointer, or names nothing in the store.
        """
        for pointer in params.inputs:
            # `path_for` both validates the pointer and gives back the store's own
            # sanitised filename, so the copy cannot be aimed outside `workdir`.
            source = self._store.path_for(pointer)
            shutil.copyfile(source, workdir / source.name)

        # After the artifacts, not before: artifact names are model-chosen, and one that
        # collides with the runner would otherwise replace it. This way the collision
        # costs the input, which the script reports, rather than the interpreter.
        shutil.copyfile(Path(__file__).parent / "sandbox.py", workdir / RUNNER_NAME)
        (workdir / SCRIPT_NAME).write_text(params.code, encoding="utf-8")

    async def _execute(self, workdir: Path) -> ToolResponse:
        """Start the child, hold it to the clock, and turn its exit into a response."""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            # Isolated mode: no PYTHONPATH, no user site-packages, no script directory on
            # sys.path. The venv's own site-packages still resolve — they come from
            # sys.prefix — so pandas is importable and nothing of ours is.
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
            async with asyncio.timeout(self._timeout):
                stdout, stderr = await process.communicate()
        except TimeoutError:
            _kill_group(process)
            await process.wait()  # reap it, now that it cannot outlive the kill
            return ToolResponse(
                content=f"The script was killed after {self._timeout:g} seconds and its "
                "output was lost — nothing printed before the timeout is recoverable. "
                "Bound every loop, work over the data you were given rather than "
                "generating more, and print only the result.",
                is_error=True,
            )
        except asyncio.CancelledError:
            # §10: clean up and re-raise, never swallow. Not awaited here — the task is
            # already cancelled, so a second await would be interrupted and the child
            # left running; SIGKILL to the group is what guarantees no orphan, and the
            # loop's child watcher reaps it.
            _kill_group(process)
            raise

        if process.returncode != 0:
            # The traceback, not a summary of it: the model wrote the script, so its own
            # filename and line numbers are what let it fix the thing in one turn.
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

    An allow-list, because the parent's environment holds the run's API keys and a
    deny-list only excludes the secrets someone remembered (§9). Nothing here is read
    from `os.environ`: `PATH` is the stdlib's own default rather than ours, which keeps
    the "no `os.environ` outside `config.py`" rule intact and is all the child needs — it
    is started by absolute path and has no business shelling out.

    `HOME` and `TMPDIR` point at the scratch directory so a library writing a cache or a
    temporary file puts it somewhere that is deleted with the call, not in the user's
    home. The locale is set because `-I` implies `-E`, so `PYTHONIOENCODING` would be
    ignored and printing a non-ASCII label would otherwise depend on the ambient locale.
    """
    return {
        "PATH": os.defpath,
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }


def _kill_group(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group, tolerating one that already exited.

    The group, not the process: `start_new_session=True` made the child a group leader,
    so this reaches anything it spawned. SIGKILL rather than SIGTERM because a script
    that ignored the clock has already shown it will not be asked politely.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def _text(raw: bytes) -> str:
    """Decode and cap one captured stream for a prompt.

    `errors="replace"`: a script printing a stray byte should cost a mojibake character,
    not the whole response. Capped here rather than by reading less, because the
    interesting end of a traceback is its last line and a short read would drop it —
    a smarter cap can wait for evidence that the head is the wrong half to keep.
    """
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    # Same marker shape as `ArtifactStore.preview`, deliberately not shared (§2.3): that
    # one elides a file by bytes without reading it, this one elides a captured stream
    # already in memory by characters. Different axis; they will not move together.
    return (
        f"{text[:MAX_OUTPUT_CHARS]}\n... [elided, {len(text) - MAX_OUTPUT_CHARS} more characters]"
    )
