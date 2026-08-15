"""Tests for the Typer entrypoint and the single error boundary (§4, §5, §8).

Exit code, stdout, and stderr are asserted separately throughout: asserting only
stdout is how stream-contract regressions ship (§12). Assertions are on collapsed
whitespace or substrings because Rich pads help output to the console width — the
contract is the text, not Rich's rendering of it.
"""

import pytest
import typer
from typer.testing import CliRunner

import orchestra.cli.app as cli_app
from orchestra import __version__
from orchestra.cli.app import app, error_boundary
from orchestra.cli.console import console
from orchestra.config import load_config
from orchestra.core.errors import ConfigError, ExitCode, ProviderError, TaskFailure
from orchestra.core.state import AgentRole, Plan, Subtask, SubtaskStatus, TaskState

runner = CliRunner()

# Without this the runner reports the program as "root", which would make the
# usage-line assertions test the harness rather than the CLI.
PROG = "orchestra"


def _squash(text: str) -> str:
    """Collapse the width padding Rich adds, so `in` compares the words only."""
    return " ".join(text.split())


def test_cli_help_returns_zero_and_prints_usage_to_stdout() -> None:
    result = runner.invoke(app, ["--help"], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    stdout = _squash(result.stdout)
    assert f"Usage: {PROG} [OPTIONS] COMMAND [ARGS]..." in stdout
    assert "--version" in stdout
    assert result.stderr == ""  # help is a result, not a diagnostic (§5)


def test_cli_help_without_valid_config_still_succeeds() -> None:
    """§4: `--help` must work in a fresh checkout, so no command may load config at import.

    The `pytest.raises` is the precondition, not the subject: it proves this process
    genuinely has no usable configuration before the CLI is invoked.
    """
    with pytest.raises(ConfigError):
        load_config()

    result = runner.invoke(app, ["--help"], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert "Usage:" in _squash(result.stdout)
    assert result.stderr == ""


def test_cli_no_args_exits_usage_with_help_on_stdout() -> None:
    """`no_args_is_help=True` (§4). Click >=8.2 treats it as a usage error, hence 2."""
    result = runner.invoke(app, [], prog_name=PROG)

    assert result.exit_code == ExitCode.USAGE
    assert f"Usage: {PROG} [OPTIONS] COMMAND [ARGS]..." in _squash(result.stdout)
    assert result.stderr == ""


def test_cli_unknown_option_exits_usage_with_error_on_stderr() -> None:
    result = runner.invoke(app, ["--nope"], prog_name=PROG)

    assert result.exit_code == ExitCode.USAGE
    assert "No such option: --nope" in _squash(result.stderr)
    assert result.stdout == ""  # nothing a pipe would pick up (§5)


def test_cli_version_prints_version_to_stdout() -> None:
    result = runner.invoke(app, ["--version"], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert result.stdout.strip() == __version__
    assert result.stderr == ""


# --------------------------------------------------------------------------
# `run` — parse, delegate, map to an exit code (§4). The run itself is stubbed:
# what the engine does is `test_engine.py`'s subject, not the command's.
# --------------------------------------------------------------------------

PROMPT = "Summarize the last 3 quarters' financial trends"


def _finished_state(*statuses: SubtaskStatus) -> TaskState:
    plan = Plan(
        subtasks=[
            Subtask(id=f"step_{index}", role=AgentRole.ANALYTICS, instruction="Do the thing")
            for index, _ in enumerate(statuses)
        ]
    )
    for subtask, status in zip(plan.subtasks, statuses, strict=True):
        subtask.status = status
        if status is SubtaskStatus.DONE:
            subtask.output_pointer = f"artifact:{subtask.id}.txt"
    return TaskState(user_request=PROMPT, plan=plan)


def _stub_run_once(monkeypatch: pytest.MonkeyPatch, outcome: TaskState | BaseException) -> None:
    """Replace the delegation target, leaving the command's own behaviour under test."""

    async def fake_run_once(prompt: str) -> TaskState:
        assert prompt == PROMPT  # the command passes the argument through unchanged
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(cli_app, "run_once", fake_run_once)


def test_run_succeeds_and_prints_the_summary_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, SubtaskStatus.DONE))

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert "done     step_0  artifact:step_0.txt" in result.stdout
    assert result.stderr == ""


def test_run_with_a_failed_subtask_still_reports_and_exits_task_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial results beat no results: the artifacts still print, the code says it failed."""
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, SubtaskStatus.FAILED))

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.TASK_FAILURE
    assert "done     step_0  artifact:step_0.txt" in result.stdout
    assert "failed   step_1" in result.stdout
    assert result.stderr == ""


def test_run_maps_a_task_failure_to_its_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_once(monkeypatch, TaskFailure("The planner returned an unusable plan twice."))

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.TASK_FAILURE
    assert "unusable plan" in result.stderr
    assert result.stdout == ""  # nothing half-written for a pipe to parse (§5)


def test_run_interrupted_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    """§8: Ctrl-C cancels the run and exits 130 rather than dumping a traceback."""
    _stub_run_once(monkeypatch, KeyboardInterrupt())

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.INTERRUPTED
    assert "Interrupted." in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


# --------------------------------------------------------------------------
# error_boundary — the single §8 mapping point, testable without a command.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (ConfigError("ANTHROPIC_API_KEY is not set."), ExitCode.CONFIG),
        (ProviderError("provider failed after retries"), ExitCode.PROVIDER),
        (KeyboardInterrupt(), ExitCode.INTERRUPTED),
        (ValueError("a bug, not a condition the user can fix"), ExitCode.UNHANDLED),
    ],
    ids=["config", "provider", "interrupt", "unexpected"],
)
def test_error_boundary_failure_maps_to_its_exit_code(
    raised: BaseException,
    expected: ExitCode,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exc_info, error_boundary():
        raise raised

    assert exc_info.value.exit_code == expected
    captured = capsys.readouterr()
    assert captured.err != ""  # every failure is rendered, once, here
    assert captured.out == ""  # diagnostics never reach stdout (§5)


def test_error_boundary_typer_exit_passes_through_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`typer.Exit` is control flow, not a failure — re-wrapping it would lose the code."""
    with pytest.raises(typer.Exit) as exc_info, error_boundary():
        raise typer.Exit(ExitCode.TASK_FAILURE)

    assert exc_info.value.exit_code == ExitCode.TASK_FAILURE
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_error_boundary_without_debug_omits_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§8: give the message, the cause, and the fix — never a traceback by default."""
    with pytest.raises(typer.Exit), error_boundary():
        raise ConfigError("ANTHROPIC_API_KEY is not set.")

    stderr = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY is not set." in stderr
    assert "Traceback" not in stderr


def test_error_boundary_with_debug_includes_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit), error_boundary(debug=True):
        raise ConfigError("ANTHROPIC_API_KEY is not set.")

    stderr = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY is not set." in stderr
    assert "Traceback" in stderr


@pytest.mark.parametrize("raised", [ConfigError, ValueError], ids=["known", "unexpected"])
def test_error_boundary_preserves_bracketed_text_in_message(
    raised: type[Exception],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: Rich markup is on by default, so `[not_found_error]` was deleted.

    §8 promises the message, the cause, and the fix. A provider error naming a
    bracketed token silently lost it — on both boundary arms.
    """
    with pytest.raises(typer.Exit), error_boundary():
        raise raised("model [claude-x] rejected: [not_found_error]")

    stderr = capsys.readouterr().err
    assert "[claude-x]" in stderr
    assert "[not_found_error]" in stderr


def test_console_stdout_does_not_mangle_markup_in_results() -> None:
    """§5: stdout carries documents a script parses — Rich must not eat part of one."""
    payload = '{"note": "see [link=http://x]here[/link]", "n": 1}'
    with console.capture() as captured:
        console.print(payload)

    assert captured.get().strip() == payload
