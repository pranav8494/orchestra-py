"""THE two `Console` objects. No other module in `src/` may construct one (§5).

Two objects rather than one with a `stderr=` argument, so a forgotten keyword cannot
put a progress line on stdout. Rich already handles `NO_COLOR`, `TERM=dumb` and
non-tty streams; call sites gate *colour* on `console.is_terminal` and *layout* on
`stdout_is_tty()`, which are not the same question.
"""

from rich.console import Console

# stdout must survive a pipe intact (§5): soft_wrap so a long line or JSON document is
# not hard-wrapped at rich's assumed 80 columns, markup/highlight off so a bracketed
# token is not eaten as a style tag and a parseable document is not colourised.
console = Console(soft_wrap=True, markup=False, highlight=False)

# Diagnostics, progress, errors — for eyes, never a parser, so wrapping is wanted.
err_console = Console(stderr=True)


def stdout_is_tty() -> bool:
    """Is stdout really a terminal — whatever the colour variables say (#49)?

    `console.is_terminal` answers "may I emit escape sequences", and `FORCE_COLOR`,
    `TTY_COMPATIBLE` and friends legitimately force it to `True` on a pipe. Layout is a
    different question: a `Panel`'s box characters corrupt a redirect no matter how the
    user feels about colour (§5). Ask the stream itself, so forcing colour through a pipe
    keeps working and framing follows the file descriptor.

    `console.file`, not `sys.stdout`, so the answer is about the stream this console
    actually writes to — Rich resolves and unwraps it for us.
    """
    try:
        return console.file.isatty()
    except ValueError:
        # A closed stream, which pytest can leave behind at teardown. Not a terminal.
        return False
