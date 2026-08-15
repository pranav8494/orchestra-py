"""Pointer-based artifact store: payloads on disk, `artifact:<name>` strings in state.

Agents exchange pointers, never blobs — see `core/state.py` for why. This module is the
other end of that string: it writes the payload and resolves the pointer back.

**Why here and not in `core/`.** §1.3 keeps `core/` free of I/O so the ledger stays
portable behind another front end, and this store is precisely the part that changes
when it does — a local directory today, object storage the moment the run leaves one
process. `config.py` is the existing precedent: impure infrastructure sits at the top
level and is injected downward from `app.py`.

Payloads are `bytes`, `str`, or an existing file. Serialising a DataFrame is the job of
the agent that owns one; the store stays ignorant of what it holds, which is what makes
it swappable.

Synchronous by design (§10 keeps blocking I/O off the event loop): the async engine
calls these through `asyncio.to_thread`, so every method here must be safe to run in
parallel with itself. `_reserve` is the reason that holds.
"""

import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from orchestra.core.errors import ConfigError, TaskFailure
from orchestra.core.state import ARTIFACT_NAME_PATTERN, ARTIFACT_PREFIX


class ArtifactStore:
    """Stores payloads under `root` and hands back pointer keys.

    One instance per run, constructed in `app.py` and injected — one of each thing (§1.5),
    but not a singleton, which would hide the dependency from its users (§3.3).
    """

    def __init__(self, root: Path) -> None:
        """Create the store, making `root` if it does not exist.

        Args:
            root: absolute directory to write into. Comes from `Config.artifact_dir`.

        Raises:
            ConfigError: `root` cannot be created. Raised when the store is constructed
                rather than at the first write, so `app.py` surfaces a bad path before
                any agent runs (§9).
        """
        try:
            root.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            # ValueError, not just OSError: a NUL byte in the path fails in the encoder
            # before the syscall, and it must not escape the taxonomy as an exit-1 bug.
            raise ConfigError(f"Cannot use artifact directory {root}: {exc}") from exc
        self._root = root

    @property
    def root(self) -> Path:
        """The directory payloads are written to."""
        return self._root

    def put_bytes(self, name: str, data: bytes) -> str:
        """Store `data` under `name` (or the next free variant) and return its pointer."""
        path = self._reserve(name)
        with _as_task_failure(f"write {path}"):
            path.write_bytes(data)
        return ARTIFACT_PREFIX + path.name

    def put_text(self, name: str, text: str) -> str:
        """Store `text` as UTF-8 and return its pointer."""
        # Explicit encoding: `Path.write_text` defaults to the locale's, so the same CSV
        # would round-trip differently on a machine that is not UTF-8.
        return self.put_bytes(name, text.encode("utf-8"))

    def put_file(self, path: Path) -> str:
        """Copy a file another tool already wrote — a Plotly chart — into the store."""
        target = self._reserve(path.name)
        with _as_task_failure(f"copy {path}"):
            shutil.copyfile(path, target)
        return ARTIFACT_PREFIX + target.name

    def get_bytes(self, pointer: str) -> bytes:
        """Read the payload behind `pointer`."""
        path = self._resolve(pointer)
        with _as_task_failure(f"read {path}"):
            return path.read_bytes()

    def get_text(self, pointer: str) -> str:
        """Read the payload behind `pointer` as UTF-8 text."""
        return self.get_bytes(pointer).decode("utf-8")

    def path_for(self, pointer: str) -> Path:
        """Filesystem path behind `pointer`, for handing a chart file to the renderer.

        Raises:
            TaskFailure: the pointer is malformed or names nothing.
        """
        return self._resolve(pointer)

    def _resolve(self, pointer: str) -> Path:
        """Turn a pointer back into a path inside `root`.

        Raises:
            TaskFailure: malformed pointer or missing file. A pointer in state that
                resolves to nothing means the run has lost data it claims to hold, which
                ends the run (exit 5) rather than surfacing as an empty payload. Callers
                that are tools convert this to `ToolResponse(is_error=True)` at their own
                boundary (§6) — the store does not know it is being called by one.
        """
        if not pointer.startswith(ARTIFACT_PREFIX):
            raise TaskFailure(f"Not an artifact pointer: {pointer!r}")
        path = self._root / _safe_name(pointer.removeprefix(ARTIFACT_PREFIX))
        if not path.is_file():
            raise TaskFailure(f"Artifact not found: {pointer!r}")
        return path

    def _reserve(self, name: str) -> Path:
        """Atomically claim the first unused of `name`, `name-1`, `name-2`, …

        Two subtasks both naming their output `chart.png` must not clobber each other,
        and the pointer returned by the earlier `put` must keep resolving to what it
        stored. `touch(exist_ok=False)` is `O_CREAT|O_EXCL`, so the check and the claim
        are one syscall: a plain `exists()` test would let two threads — the engine
        dispatches subtasks concurrently — pick the same name between the check and the
        write, and one payload would be lost. The loop terminates because each pass
        proposes a name it has not tried.
        """
        safe = _safe_name(name)
        stem, suffix = Path(safe).stem, Path(safe).suffix
        attempt = 0
        while True:
            candidate = self._root / (safe if attempt == 0 else f"{stem}-{attempt}{suffix}")
            try:
                candidate.touch(exist_ok=False)
            except FileExistsError:
                attempt += 1
            except OSError as exc:
                raise TaskFailure(f"Artifact store could not create {candidate}: {exc}") from exc
            else:
                return candidate


def _safe_name(name: str) -> str:
    """Check a name against the artifact allow-list in `core/state.py`.

    Names reach the store from model output — a planner naming an output file — so this
    is a trust boundary, not a sanity check. Because the pattern admits no separator, no
    colon, no leading dot, and no control character, `root / name` cannot leave `root`;
    containment is a property of the name rather than a second check that could drift
    from this one.

    Raises:
        TaskFailure: the name is not a valid artifact name.
    """
    if not re.fullmatch(ARTIFACT_NAME_PATTERN, name):
        raise TaskFailure(f"Unsafe artifact name: {name!r}")
    return name


@contextmanager
def _as_task_failure(action: str) -> Iterator[None]:
    """Turn a filesystem error into the taxonomy's `TaskFailure` (§8), never a bare `OSError`."""
    try:
        yield
    except OSError as exc:
        raise TaskFailure(f"Artifact store could not {action}: {exc}") from exc
