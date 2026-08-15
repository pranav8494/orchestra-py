"""Error taxonomy: every deliberate failure carries the exit code it exits with.

`ExitCode` is an `IntEnum`, not the `StrEnum` §7 prescribes for closed string sets —
an exit code *is* a number the shell compares (`$?`), so `int` is its domain type and
members pass straight to `typer.Exit`.

Errors carry a message and a code; they never format themselves. Rendering happens
once, at the CLI boundary. See CONVENTIONS.md §8.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes (CONVENTIONS.md §8)."""

    SUCCESS = 0
    UNHANDLED = 1
    USAGE = 2  # Typer's default for a bad invocation
    CONFIG = 3
    PROVIDER = 4
    TASK_FAILURE = 5
    INTERRUPTED = 130  # 128 + SIGINT


class OrchestraError(Exception):
    """Base for every error this application raises on purpose.

    Anything else reaching the CLI boundary is a bug and exits `UNHANDLED`.
    """

    exit_code: ExitCode = ExitCode.UNHANDLED


class ConfigError(OrchestraError):
    """Configuration is missing or invalid. Raised at startup, before work begins (§9)."""

    exit_code = ExitCode.CONFIG


class ProviderError(OrchestraError):
    """The model provider failed and retries were exhausted."""

    exit_code = ExitCode.PROVIDER


# N818: named for the domain concept (a failed task), not the Python convention —
# the taxonomy in CONVENTIONS.md §8 and the task state both call it a task failure.
class TaskFailure(OrchestraError):  # noqa: N818
    """The run completed but the task did not succeed."""

    exit_code = ExitCode.TASK_FAILURE
