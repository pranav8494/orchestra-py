"""THE two `Console` objects. No other module in `src/` may construct one (§5).

Two objects rather than one with a `stderr=` argument: a caller cannot then send a
progress line to stdout by forgetting a keyword. stdout carries results a script can
parse; stderr carries diagnostics, progress, and errors.

Rich already does the environment handling §5 asks for, so none of it is repeated
here: a non-empty `NO_COLOR` sets `Console.no_color`, `TERM=dumb`/`unknown` drops the
colour system to `None`, and a non-tty file leaves `is_terminal` False so nothing
styled is emitted down a pipe. Call sites still gate *layout* on `console.is_terminal`.
"""

from rich.console import Console

# soft_wrap: with no tty, rich assumes 80 columns and hard-wraps — which would break a
# long result line or a JSON document mid-token. stdout has to survive a pipe (§5).
console = Console(soft_wrap=True)

# Diagnostics, progress, and errors. Wrapping is wanted here: this stream is for eyes,
# never for a parser.
err_console = Console(stderr=True)
