"""THE two `Console` objects. No other module in `src/` may construct one (§5).

Two objects rather than one with a `stderr=` argument, so a forgotten keyword cannot
put a progress line on stdout. Rich already handles `NO_COLOR`, `TERM=dumb` and
non-tty streams; call sites still gate *layout* on `console.is_terminal`.
"""

from rich.console import Console

# stdout must survive a pipe intact (§5): soft_wrap so a long line or JSON document is
# not hard-wrapped at rich's assumed 80 columns, markup/highlight off so a bracketed
# token is not eaten as a style tag and a parseable document is not colourised.
console = Console(soft_wrap=True, markup=False, highlight=False)

# Diagnostics, progress, errors — for eyes, never a parser, so wrapping is wanted.
err_console = Console(stderr=True)
