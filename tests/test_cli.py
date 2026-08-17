"""Tests for the Typer entrypoint and the single error boundary (§4, §5, §8).

Exit code, stdout, and stderr are asserted separately throughout: asserting only stdout is
how stream-contract regressions ship (§12). Assertions collapse whitespace because Rich
pads help output to the console width — the contract is the text, not the rendering.

`run` is stubbed here; the un-stubbed CLI cases, which drive the real agents, are in
`test_end_to_end.py`.
"""

import json
import sys
from functools import partial

import pytest
import typer
from typer.testing import CliRunner

import orchestra.cli.app as cli_app
from conftest import force_terminal
from orchestra import __version__
from orchestra.app import RunObserver
from orchestra.cli.app import app, error_boundary
from orchestra.cli.chat import HINT, ConsoleChat
from orchestra.cli.console import console
from orchestra.cli.prompt import ConsoleAsker
from orchestra.cli.render import RenderMode
from orchestra.config import load_config
from orchestra.core.errors import ConfigError, ExitCode, ProviderError, TaskFailure
from orchestra.core.interrupt import Chat
from orchestra.core.question import Asker
from orchestra.core.state import (
    AgentRole,
    FinalReport,
    KeyFigure,
    Plan,
    Subtask,
    SubtaskStatus,
    TaskState,
)

runner = CliRunner()

# Without this the runner reports the program as "root", making the usage-line assertions
# test the harness rather than the CLI.
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
    The `pytest.raises` is the precondition: this process has no usable config."""
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
SUMMARY = "Revenue grew in each of the last three quarters."


def _finished_state(
    *statuses: SubtaskStatus,
    summary: str = SUMMARY,
    failure_reason: str | None = None,
) -> TaskState:
    """A ledger shaped the way `run_task` hands one back: statuses, artifacts, a report."""
    plan = Plan(
        subtasks=[
            Subtask(id=f"step_{index}", role=AgentRole.ANALYTICS, instruction="Do the thing")
            for index, _ in enumerate(statuses)
        ]
    )
    artifacts: dict[str, str] = {}
    for subtask, status in zip(plan.subtasks, statuses, strict=True):
        subtask.status = status
        if status is SubtaskStatus.DONE:
            subtask.output_pointer = f"artifact:{subtask.id}.txt"
            artifacts[subtask.id] = subtask.output_pointer
    return TaskState(
        user_request=PROMPT,
        plan=plan,
        artifacts=artifacts,
        # The aggregator always leaves one behind, even on the paths that fell short.
        final_result=FinalReport(
            executive_summary=summary,
            key_figures=[
                KeyFigure(label="Q3 revenue", value="145", source=pointer)
                for pointer in list(artifacts.values())[:1]
            ],
        ),
        failure_reason=failure_reason,
    )


def _stub_run_once(
    monkeypatch: pytest.MonkeyPatch,
    outcome: TaskState | BaseException,
    *,
    askers: list[Asker | None] | None = None,
    chats: list[Chat | None] | None = None,
) -> list[RunObserver | None]:
    """Replace the delegation target, leaving the command's own behaviour under test.

    Returns the list the command's observer is recorded in, so a test can assert which
    dashboard the flags asked for without running one. `askers` is the same seam for
    whoever the command offered to answer clarifying questions (#10), and `chats` for
    whoever it offered to interrupt the run (#12).
    """
    observers: list[RunObserver | None] = []

    async def fake_run_once(
        prompt: str,
        *,
        observer: RunObserver | None = None,
        asker: Asker | None = None,
        chat: Chat | None = None,
    ) -> TaskState:
        assert prompt == PROMPT  # the command passes the argument through unchanged
        observers.append(observer)
        if askers is not None:
            askers.append(asker)
        if chats is not None:
            chats.append(chat)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(cli_app, "run_once", fake_run_once)
    return observers


def _requested_mode(observers: list[RunObserver | None]) -> RenderMode:
    """The `RenderMode` baked into the observer the command handed `run_once`, read off the
    partial so the assertion is on the renderer's input, not on what Rich drew (§12)."""
    assert len(observers) == 1
    observer = observers[0]
    assert isinstance(observer, partial)
    mode = observer.keywords["mode"]
    assert isinstance(mode, RenderMode)
    return mode


def test_run_succeeds_and_prints_the_summary_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, SubtaskStatus.DONE))

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert SUMMARY in result.stdout
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


def test_run_output_json_puts_one_document_on_stdout_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5: `json.loads` is the assertion — it rejects a stray banner or a second document
    as trailing data."""
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, SubtaskStatus.DONE))

    result = runner.invoke(app, ["run", PROMPT, "--output", "json"], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    document = json.loads(result.stdout)
    assert document["request"] == PROMPT
    assert document["status"] == "completed"
    assert document["report"]["executive_summary"] == SUMMARY
    assert [subtask["id"] for subtask in document["subtasks"]] == ["step_0", "step_1"]
    assert result.stderr == ""


def test_run_quiet_omits_the_step_lines_and_keeps_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5: `--quiet` suppresses progress, never the result. The trace is the progress."""
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, SubtaskStatus.DONE))

    result = runner.invoke(app, ["run", PROMPT, "--quiet"], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert SUMMARY in result.stdout
    assert "Steps:" not in result.stdout
    assert "done     step_0" not in result.stdout
    assert result.stderr == ""


def test_run_does_not_frame_the_report_when_a_forced_terminal_meets_a_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#49: `FORCE_COLOR`/`CLICOLOR_FORCE` in the shell make Rich report a forced terminal
    even when stdout is redirected, so `console.is_terminal` is `True` down a pipe. Framing
    must gate on the real `stdout.isatty()` instead — a `Panel`'s box characters in a
    redirect break the first `json.loads` that meets them (§5)."""
    # What `FORCE_COLOR=3` does at construction: the stdout console is forced to a terminal,
    # so `console.is_terminal` is `True` though `CliRunner` supplies a pipe.
    monkeypatch.setattr(cli_app.console, "_force_terminal", True, raising=False)
    assert cli_app.console.is_terminal is True  # precondition: the forced-terminal state
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, SubtaskStatus.DONE))

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert SUMMARY in result.stdout
    # No `Panel` box-drawing glyphs: redirected stdout must stay pipeable (§5).
    assert not any(glyph in result.stdout for glyph in "─│╭╮╰╯"), result.stdout
    assert result.stderr == ""


# --------------------------------------------------------------------------
# Which dashboard the flags ask for (#11). The command builds the observer and
# hands it to `run_once`; what it draws is `test_render.py`'s subject, so these
# assert on the mode chosen, never on Rich's output (§12).
# --------------------------------------------------------------------------


class _Stdin:
    """A stdin that is, or is not, a terminal. `CliRunner` always supplies a pipe, so the
    interactive arm is unreachable through `invoke` — as with `conftest.force_terminal`."""

    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_run_with_a_piped_stdin_offers_nobody_to_answer_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#10: a script or CI run has no one at the keyboard, so the command hands `run_once`
    no asker. Anything else hangs."""
    askers: list[Asker | None] = []
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE), askers=askers)

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert askers == [None]


@pytest.mark.parametrize(
    ("stdin_tty", "stderr_tty", "offered"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_asker_needs_both_streams_to_be_a_terminal(
    monkeypatch: pytest.MonkeyPatch, stdin_tty: bool, stderr_tty: bool, offered: bool
) -> None:
    """Both streams, or nobody: `2>log` in particular would put the question in the file
    and leave the user staring at a terminal that has stopped.

    Called directly: `CliRunner` replaces `sys.stdin` inside `invoke`.
    """
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=stdin_tty))
    force_terminal(monkeypatch, value=stderr_tty)

    asker = cli_app._asker()

    assert isinstance(asker, ConsoleAsker) is offered
    # The interrupt key needs the same two streams (#12): the key comes off stdin and
    # everything the pause shows goes to stderr.
    assert cli_app._interactive() is offered


def test_run_with_a_piped_stdin_offers_nobody_to_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#12: no terminal, no key to press — so the command hands `run_once` no chat and the
    run simply never pauses."""
    chats: list[Chat | None] = []
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE), chats=chats)

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert chats == [None]
    assert HINT not in result.stderr  # nothing to advertise


def test_run_on_a_terminal_offers_a_chat_and_says_how_to_reach_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other arm: an interruptible run has to say so, and the affordance is a
    diagnostic, so it goes to stderr and never near the result (§5).

    `_interactive` is patched rather than the streams, because `CliRunner` replaces
    `sys.stdin` inside `invoke`; which streams make a run interactive is the parametrized
    test above.
    """
    chats: list[Chat | None] = []
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE), chats=chats)
    monkeypatch.setattr(cli_app, "_interactive", lambda: True)
    force_terminal(monkeypatch, value=True)

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert isinstance(chats[0], ConsoleChat)
    assert HINT in result.stderr
    assert HINT not in result.stdout


def test_run_on_a_terminal_asks_for_the_live_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    observers = _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE))
    force_terminal(monkeypatch, value=True)

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert _requested_mode(observers) is RenderMode.LIVE


def test_run_piped_falls_back_to_plain_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-TTY fallback: it has to work piped, in CI, and in a recording."""
    observers = _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE))
    force_terminal(monkeypatch, value=False)

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert _requested_mode(observers) is RenderMode.PLAIN


def test_run_with_stdout_redirected_still_draws_live_on_a_tty_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`orchestra run x > report.txt`: stdout is a file, stderr is still the terminal. Pins
    which console each decision reads — mode follows stderr, framing follows stdout (§5)."""
    observers = _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE))
    force_terminal(monkeypatch, value=True)  # stderr only; `console` stays a pipe

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert _requested_mode(observers) is RenderMode.LIVE
    # Unframed: stdout is redirected, so a panel's box would land in the file.
    assert SUMMARY in result.stdout
    assert "╭" not in result.stdout


def test_run_quiet_asks_for_no_dashboard_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """§5: `--quiet` suppresses progress, including the `Live` region, never the result."""
    observers = _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE))
    force_terminal(monkeypatch, value=True)

    result = runner.invoke(app, ["run", PROMPT, "--quiet"], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert _requested_mode(observers) is RenderMode.NONE
    assert SUMMARY in result.stdout  # the report survives the flag


def test_run_json_never_starts_a_live_region_even_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose stdout is being parsed is one nobody is watching redraw."""
    observers = _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE))
    force_terminal(monkeypatch, value=True)

    result = runner.invoke(app, ["run", PROMPT, "-o", "json"], prog_name=PROG)

    assert result.exit_code == ExitCode.SUCCESS
    assert _requested_mode(observers) is RenderMode.PLAIN
    json.loads(result.stdout)  # still exactly one document, unframed


def test_run_that_stopped_short_prints_the_report_and_the_reason_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The artifacts are on disk, so they are reported; why the run stopped is a diagnostic
    and goes to stderr (§5, §8)."""
    reason = "Step cap of 1 exceeded; the plan is too large to run."
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, failure_reason=reason))

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.TASK_FAILURE
    assert SUMMARY in result.stdout
    assert "done     step_0  artifact:step_0.txt" in result.stdout
    assert reason not in result.stdout  # never on the stream a script parses
    assert reason in _squash(result.stderr)


def test_run_whose_synthesis_failed_prints_the_ledger_report_and_exits_task_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#9's trade at the boundary: a provider outage during synthesis exits 5 with a report
    rather than 4 with an empty stdout. The operator tells the two apart by stderr (§8)."""
    reason = "The report could not be synthesised: 503 overloaded_error"
    ledger_summary = "No synthesis was available for this run"
    _stub_run_once(
        monkeypatch,
        _finished_state(SubtaskStatus.DONE, summary=ledger_summary, failure_reason=reason),
    )

    result = runner.invoke(app, ["run", PROMPT], prog_name=PROG)

    assert result.exit_code == ExitCode.TASK_FAILURE
    assert ledger_summary in _squash(result.stdout)
    assert "done     step_0  artifact:step_0.txt" in result.stdout
    assert "overloaded_error" not in result.stdout  # never on the stream a script parses
    assert reason in _squash(result.stderr)


def test_run_that_stopped_short_reports_the_reason_on_both_streams_in_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which stream says what does not change with `-o`: the document carries the reason,
    stderr repeats it for the person watching the pipe."""
    reason = "No worker is registered for roles: [<AgentRole.VISUALIZATION: 'visualization'>]"
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE, failure_reason=reason))

    result = runner.invoke(app, ["run", PROMPT, "-o", "json"], prog_name=PROG)

    assert result.exit_code == ExitCode.TASK_FAILURE
    assert json.loads(result.stdout)["failure_reason"] == reason
    # Rich markup would read the bracketed role list as a style tag. Substring, because
    # stderr is for eyes and wraps at the console width.
    assert "[<AgentRole.VISUALIZATION:" in _squash(result.stderr)


def test_run_rejects_an_unknown_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The enum makes this a usage error instead of a run that prints nonsense."""
    _stub_run_once(monkeypatch, _finished_state(SubtaskStatus.DONE))

    result = runner.invoke(app, ["run", PROMPT, "--output", "yaml"], prog_name=PROG)

    assert result.exit_code == ExitCode.USAGE
    assert "yaml" in _squash(result.stderr)
    assert result.stdout == ""


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
    """`typer.Exit` is control flow; re-wrapping it would lose the code."""
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
    """Regression: Rich markup is on by default, so a provider error naming
    `[not_found_error]` silently lost it — on both boundary arms (§8)."""
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
