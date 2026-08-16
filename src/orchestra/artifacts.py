"""Pointer-based artifact store: payloads on disk, `artifact:<name>` strings in state.

**Why here and not in `core/`.** §1.3 keeps `core/` free of I/O so the ledger stays
portable behind another front end, and this store is the part that changes when it does
— a local directory today, object storage once the run leaves one process. `config.py`
is the precedent: impure infrastructure at the top level, injected from `app.py`.

Payloads are `bytes`, `str`, or an existing file; serialising a DataFrame belongs to the
agent that owns one. Synchronous, so the async engine calls it through `asyncio.to_thread`
(§10) — which is why every method has to be safe to run in parallel with itself.
"""

import codecs
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from orchestra.core.errors import ConfigError, TaskFailure
from orchestra.core.state import ARTIFACT_NAME_PATTERN, ARTIFACT_PREFIX

# Characters of an artifact a preview may carry into a prompt — roughly 200 tokens, so a
# whole plan's previews still leave the model room to answer.
DEFAULT_PREVIEW_LIMIT = 800

# UTF-8 is at most 4 bytes per character, so this many bytes always yields `limit`
# characters when the file has them, whatever the payload size.
_BYTES_PER_CHARACTER = 4


class ArtifactStore:
    """Stores payloads under `root` and hands back pointer keys.

    One instance per run, constructed in `app.py` and injected — not a singleton (§3.3).
    """

    def __init__(self, root: Path) -> None:
        """Create the store, making `root` if it does not exist.

        Args:
            root: absolute directory to write into, from `Config.artifact_dir`.

        Raises:
            ConfigError: `root` is unusable — at construction, so `app.py` surfaces a bad
                path before any agent runs (§9).
        """
        try:
            root.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            # ValueError too: a NUL byte fails in the encoder before the syscall, and
            # must not escape the taxonomy as an exit-1 bug.
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
        # Explicit encoding: `write_text` defaults to the locale's.
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

    def preview(self, pointer: str, *, limit: int = DEFAULT_PREVIEW_LIMIT) -> str:
        """A compact, prompt-safe rendering of the payload behind `pointer`.

        The aggregator (#8) is shown previews, never payloads — a report over five
        artifacts would otherwise put five whole files into one prompt. Only the head of
        the file is read, so a large artifact costs what a small one does.

        Undecodable payloads are described, not decoded: `errors="replace"` turns a PNG
        into a screenful of U+FFFD that costs tokens and says nothing, where the size and
        the fact that it is binary are the whole informative content.

        Args:
            pointer: the artifact to preview.
            limit: how many characters of text to keep before eliding.

        Returns:
            The whole payload when it is short text, the first `limit` characters plus an
            elision marker naming the omitted bytes when it is longer, or
            `<binary, N bytes>` when it is not UTF-8 text.

        Raises:
            TaskFailure: the pointer is malformed or names nothing.
        """
        path = self._resolve(pointer)
        with _as_task_failure(f"read {path}"):
            size = path.stat().st_size
            with path.open("rb") as handle:
                head = handle.read(limit * _BYTES_PER_CHARACTER)

        # Incremental, so a multi-byte character straddling the end of the read is held
        # back instead of reported as a decode error — otherwise a long UTF-8 document
        # would be called binary whenever the cut landed mid-character.
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            text = decoder.decode(head, len(head) == size)
        except UnicodeDecodeError:
            return f"<binary, {size} bytes>"

        if len(head) == size and len(text) <= limit:
            return text
        kept = text[:limit]
        # In bytes against the file's own size: a character count needs the whole file
        # decoded, which is the read this method exists to avoid.
        omitted = size - len(kept.encode("utf-8"))
        return f"{kept}\n... [elided, {omitted} more bytes]"

    def path_for(self, pointer: str) -> Path:
        """Filesystem path behind `pointer`, for handing a chart to the renderer.

        Raises:
            TaskFailure: the pointer is malformed or names nothing.
        """
        return self._resolve(pointer)

    def _resolve(self, pointer: str) -> Path:
        """Turn a pointer back into a path inside `root`.

        Raises:
            TaskFailure: malformed pointer or missing file — the run has lost data it
                claims to hold, so it ends (exit 5) rather than yielding an empty payload.
                Tools convert this to `ToolResponse(is_error=True)` at their own boundary
                (§6); the store does not know it is being called by one.
        """
        if not pointer.startswith(ARTIFACT_PREFIX):
            raise TaskFailure(f"Not an artifact pointer: {pointer!r}")
        path = self._root / _safe_name(pointer.removeprefix(ARTIFACT_PREFIX))
        if not path.is_file():
            raise TaskFailure(f"Artifact not found: {pointer!r}")
        return path

    def _reserve(self, name: str) -> Path:
        """Atomically claim the first unused of `name`, `name-1`, `name-2`, …

        `touch(exist_ok=False)` is `O_CREAT|O_EXCL`, so the check and the claim are one
        syscall. An `exists()` test would let two threads — the engine dispatches
        subtasks concurrently — pick the same name and lose a payload.
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
    """Check a name against the allow-list in `core/state.py`.

    Names come from model output, so this is a trust boundary. The pattern admits no
    separator, colon, leading dot or control character, so `root / name` cannot leave
    `root` — containment is a property of the name, not a second check that could drift.

    Raises:
        TaskFailure: not a valid artifact name.
    """
    if not re.fullmatch(ARTIFACT_NAME_PATTERN, name):
        raise TaskFailure(f"Unsafe artifact name: {name!r}")
    return name


@contextmanager
def _as_task_failure(action: str) -> Iterator[None]:
    """Turn a filesystem error into the taxonomy's `TaskFailure` (§8)."""
    try:
        yield
    except OSError as exc:
        raise TaskFailure(f"Artifact store could not {action}: {exc}") from exc
