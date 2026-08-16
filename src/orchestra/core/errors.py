"""Error taxonomy: every deliberate failure carries the exit code it exits with (§8).

`IntEnum`, not the `StrEnum` §7 prescribes: an exit code *is* a number the shell
compares (`$?`), and members pass straight to `typer.Exit`.

Errors never format themselves — rendering happens once, at the CLI boundary.
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
    """Base for every deliberate error. Anything else at the CLI boundary is a bug."""

    exit_code: ExitCode = ExitCode.UNHANDLED


class ConfigError(OrchestraError):
    """Configuration is missing or invalid. Raised at startup, before work begins (§9)."""

    exit_code = ExitCode.CONFIG


class ProviderError(OrchestraError):
    """The model provider failed and retries were exhausted."""

    exit_code = ExitCode.PROVIDER


# N818: named for the domain concept — §8 and the task state both say "task failure".
class TaskFailure(OrchestraError):  # noqa: N818
    """The run completed but the task did not succeed."""

    exit_code = ExitCode.TASK_FAILURE
