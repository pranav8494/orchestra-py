"""The child process's entry point: install the guards, then run the model's script.

- **Copied, never imported** — `RunPythonTool` copies this into the scratch directory as
  `_run.py` and starts it with the script as `sys.argv[1]`. It lives under `src/` anyway
  so ruff and `mypy --strict` see it; an entry point the gates skip breaks in production.
- **Isolation, not a sandbox** — the guard below raises the cost of an *accident*, not of
  hostile code. Real containment is a container, a seccomp profile or a separate uid, not
  a monkeypatch the script could undo.
"""

import runpy
import socket
import sys
from typing import NoReturn


def _no_network(*_args: object, **_kwargs: object) -> NoReturn:
    """Refuse every connection attempt, in the child, before a packet is sent.

    `OSError` because that is what a real failed connection raises, so a script with its
    own `except OSError` degrades instead of dying.
    """
    raise OSError("network access is disabled in this environment")


# The methods, never the class: `ssl` does `class SSLSocket(socket)`, so replacing
# `socket.socket` turns any `import ssl`/`http.client` into a `TypeError` the model would
# spend a turn on. Connecting is what is worth refusing, not constructing a socket.
# `create_connection` is patched too because it resolves the name first — refusing it here
# fails before a DNS query leaves the machine, not after.
socket.socket.connect = _no_network  # type: ignore[method-assign]
socket.socket.connect_ex = _no_network  # type: ignore[method-assign]
socket.create_connection = _no_network

# `run_path`, not `exec(source)`: it compiles with the real filename, so the traceback
# names `analysis.py` and the model's own line numbers. `__main__` so a script guarded by
# `if __name__ == "__main__"` still runs.
runpy.run_path(sys.argv[1], run_name="__main__")
