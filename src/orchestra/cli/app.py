"""The Typer application: parse, delegate, map to an exit code (CONVENTIONS.md §4).

Also the single place a failure is turned into a message and a status (§8) —
`error_boundary` maps an `OrchestraError` to its own exit code, `KeyboardInterrupt`
to 130, and anything else to 1. Nothing below `cli/` formats an error for a human.

`load_config()` is deliberately not called here or at import — `--help` must work in a
checkout with no `.env`, so configuration is loaded inside the command, under the
boundary that turns a `ConfigError` into exit 3.
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

    Every command body wraps its delegation in this. `typer.Exit`/`typer.Abort` are
    control flow rather than failures, so they pass through untouched.

    Args:
        debug: also print the traceback. §8 forbids showing one otherwise; the
            `--debug` flag that sets it lands with the first command that can fail.
    """
    try:
        yield
    except (typer.Exit, typer.Abort):
        raise
    except OrchestraError as exc:
        # Already user-facing — errors carry the message, they never format themselves.
        # markup=False: an error naming a bracketed token ("[not_found_error]") would
        # otherwise be parsed as a style tag and silently deleted from the message.
        err_console.print(str(exc), markup=False, highlight=False)
        if debug:
            err_console.print_exception()
        raise typer.Exit(exc.exit_code) from exc
    except KeyboardInterrupt as exc:
        err_console.print("Interrupted.")
        raise typer.Exit(ExitCode.INTERRUPTED) from exc
    except Exception as exc:
        # Outside the taxonomy, so it is a bug, not a condition the user can fix.
        err_console.print(f"Unexpected error: {exc}", markup=False, highlight=False)
        if debug:
            err_console.print_exception()
        raise typer.Exit(ExitCode.UNHANDLED) from exc


# invoke_without_command: the callback owns --version, so it has to run when no
# subcommand follows. `no_args_is_help` still wins for a bare `orchestra`.
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
        # Ctrl-C cancels the run inside `asyncio.run`, which unwinds the TaskGroup and
        # re-raises KeyboardInterrupt here; the boundary maps it to 130 (§8). The
        # dashboard is torn down on that unwind, so the `Live` region is always exited.
        observer = partial(dashboard, mode=_render_mode(quiet=quiet, output=output))
        state = asyncio.run(run_once(prompt, observer=observer))
        if state.failure_reason is not None:
            # A diagnostic, so stderr (§5, §8) — and an error rather than progress, so
            # `--quiet` keeps it. Printed under `-o json` too: which stream says what must
            # not depend on the format. markup/highlight off for `error_boundary`'s
            # reason — the message can name a bracketed token Rich would eat as a tag.
            err_console.print(state.failure_reason, markup=False, highlight=False)
        # The result (§5). `is_terminal` is read here and passed down, so `render.py`
        # decides how to frame the report and this command decides nothing but where the
        # answer comes from.
        console.print(
            result_renderable(state, output=output, quiet=quiet, terminal=console.is_terminal)
        )
        # A run that fell short still prints its report; only the code says it failed.
        raise typer.Exit(ExitCode.TASK_FAILURE if state.failed else ExitCode.SUCCESS)


def _render_mode(*, quiet: bool, output: OutputFormat) -> RenderMode:
    """Which dashboard the flags ask for. The one place that policy is decided.

    `--quiet` suppresses progress entirely (§5). `--output json` never gets a `Live`
    region: the document goes to stdout and the region to stderr, but a run whose stdout
    is being parsed is one nobody is watching redraw, so the scrolling fallback is the
    honest shape — and it keeps working when the whole invocation is piped. Otherwise a
    terminal gets the live table and a pipe, a CI log or a recording gets plain lines.
    """
    if quiet:
        return RenderMode.NONE
    if output is OutputFormat.JSON or not err_console.is_terminal:
        return RenderMode.PLAIN
    return RenderMode.LIVE
