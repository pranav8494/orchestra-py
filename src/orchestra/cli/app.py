"""The Typer application: parse, delegate, map to an exit code (§4).

The one place a failure becomes a message and a status (§8): `error_boundary` maps an
`OrchestraError` to its own exit code, `KeyboardInterrupt` to 130, anything else to 1.

Config is loaded inside the command, not at import — `--help` must work in a checkout
with no `.env`, and a `ConfigError` has to land under the boundary that makes it exit 3.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Annotated

import typer

from orchestra import __version__
from orchestra.app import run_once
from orchestra.cli.console import console, err_console
from orchestra.cli.format import OutputFormat
from orchestra.cli.render import RenderMode, dashboard, result_renderable
from orchestra.core.errors import ExitCode, OrchestraError

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help=(
        "Break a plain-language request into subtasks, route them to role-specialised "
        "agents, and stream live progress to a structured result."
    ),
)


@contextmanager
def error_boundary(*, debug: bool = False) -> Iterator[None]:
    """Render a failure once, here, and exit with its code (§8).

    `typer.Exit`/`typer.Abort` are control flow, not failures, so they pass through.
    `debug` also prints the traceback, which §8 forbids otherwise.
    """
    try:
        yield
    except (typer.Exit, typer.Abort):
        raise
    except OrchestraError as exc:
        # markup=False: an error naming a bracketed token ("[not_found_error]") would
        # otherwise be parsed as a style tag and deleted from the message.
        err_console.print(str(exc), markup=False, highlight=False)
        if debug:
            err_console.print_exception()
        raise typer.Exit(exc.exit_code) from exc
    except KeyboardInterrupt as exc:
        err_console.print("Interrupted.")
        raise typer.Exit(ExitCode.INTERRUPTED) from exc
    except Exception as exc:
        # Outside the taxonomy, so a bug rather than something the user can fix.
        err_console.print(f"Unexpected error: {exc}", markup=False, highlight=False)
        if debug:
            err_console.print_exception()
        raise typer.Exit(ExitCode.UNHANDLED) from exc


# invoke_without_command: the callback owns --version, so it must run with no subcommand.
@app.callback(invoke_without_command=True)
def main(
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    """Global options. Commands are registered in later phases."""
    with error_boundary():
        if version:
            console.print(__version__)  # a result, not a diagnostic (§5)
            raise typer.Exit(ExitCode.SUCCESS)


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="The task to solve.")],
    output: Annotated[
        OutputFormat,
        typer.Option("--output", "-o", help="Shape of the result on stdout."),
    ] = OutputFormat.TEXT,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress live progress. The report still prints."),
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show the traceback on failure.")] = False,
) -> None:
    """Plan the request, run the subtasks, and report on what they produced."""
    with error_boundary(debug=debug):
        # Ctrl-C unwinds the TaskGroup inside `asyncio.run` and re-raises here, so the
        # dashboard is always torn down and the boundary maps it to 130 (§8).
        observer = partial(dashboard, mode=_render_mode(quiet=quiet, output=output))
        state = asyncio.run(run_once(prompt, observer=observer))
        if state.failure_reason is not None:
            # Diagnostic, so stderr even under `-o json` and `--quiet` (§5, §8): which
            # stream says what must not depend on the format or the noise level.
            err_console.print(state.failure_reason, markup=False, highlight=False)
        # `is_terminal` is read here and passed down so framing stays in `render.py`.
        console.print(
            result_renderable(state, output=output, quiet=quiet, terminal=console.is_terminal)
        )
        # A run that fell short still prints its report; only the code says it failed.
        raise typer.Exit(ExitCode.TASK_FAILURE if state.failed else ExitCode.SUCCESS)


def _render_mode(*, quiet: bool, output: OutputFormat) -> RenderMode:
    """Which dashboard the flags ask for. The one place that policy is decided.

    `--quiet` suppresses progress entirely (§5). `--output json` keeps the scrolling
    lines but never a `Live` region — nobody watches a run whose stdout is being parsed.
    """
    if quiet:
        return RenderMode.NONE
    if output is OutputFormat.JSON or not err_console.is_terminal:
        return RenderMode.PLAIN
    return RenderMode.LIVE
