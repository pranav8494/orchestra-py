"""The child process's entry point: install the guards, then run the model's script.

**Copied, never imported.** `RunPythonTool` copies this file into the run's scratch
directory as `_run.py` and starts it there with the script as `sys.argv[1]`; nothing in
the package imports it. It lives under `src/` anyway so ruff and `mypy --strict` see it
— an entry point the gates skip is the one that breaks in production.

**This is isolation, not a sandbox.** A scrubbed environment, a throwaway working
directory, a wall-clock kill and the socket guard below raise the cost of an *accident*
— a script that loops forever, or reaches for the network mid-analysis. They do not
contain hostile code, and containing it is out of scope here: the script is written by
our own model from our own prompt, and real containment is a container, a seccomp
profile or a separate uid, not a monkeypatch a script could undo.
"""

import runpy
import socket
import sys
from typing import NoReturn


def _no_network(*_args: object, **_kwargs: object) -> NoReturn:
    """Refuse every connection attempt, in the child, before a packet is sent.

    Raised as `OSError` because that is what a real failed connection raises, so a
    script with its own `except OSError` degrades instead of dying — and the message
    reaches the model through the traceback either way.
    """
    raise OSError("network access is disabled in this environment")


# The methods, never the class. `ssl` does `class SSLSocket(socket)`, so replacing
# `socket.socket` with a function turns the next `import ssl` — or `http.client`, or
# `urllib.request`, or anything importing them — into `TypeError: function() argument
# 'code' must be code, not str`, which is not the `OSError` above and costs the model a
# turn on a fault it did not cause. Connecting is the thing worth refusing; constructing
# a socket object is not. `create_connection` is patched as well as the methods it calls,
# because it resolves the name first: refusing it here means `http.client` and everything
# above it fail before a DNS query leaves the machine, not after.
socket.socket.connect = _no_network  # type: ignore[method-assign]
socket.socket.connect_ex = _no_network  # type: ignore[method-assign]
socket.create_connection = _no_network

# `run_path`, not `exec(source)`: it compiles with the real filename, so the traceback
# the tool hands back names `analysis.py` and the model's own line numbers. `__main__`
# so a script guarded by `if __name__ == "__main__"` still runs.
runpy.run_path(sys.argv[1], run_name="__main__")
