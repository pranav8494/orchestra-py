"""Tests for the pointer-based artifact store (CONVENTIONS.md §12).

Every test writes under `tmp_path`; nothing here touches the real `~/.orchestra`.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import ConfigError, ExitCode, TaskFailure
from orchestra.core.state import ARTIFACT_PREFIX

CSV = "quarter,revenue\nQ1,120\nQ2,131\n"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def test_put_text_returns_pointer_that_reads_back(store: ArtifactStore) -> None:
    pointer = store.put_text("revenue.csv", CSV)

    assert pointer == f"{ARTIFACT_PREFIX}revenue.csv"
    assert store.get_text(pointer) == CSV


def test_put_bytes_round_trips_unchanged(store: ArtifactStore) -> None:
    payload = b"\x89PNG\r\n\x1a\n binary chart"

    assert store.get_bytes(store.put_bytes("chart.png", payload)) == payload


def test_put_file_copies_an_existing_file_into_the_store(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """How a Plotly chart, written by the library to its own path, enters the store."""
    written_elsewhere = tmp_path / "plotly-output.html"
    written_elsewhere.write_text("<html>chart</html>", encoding="utf-8")

    pointer = store.put_file(written_elsewhere)

    assert store.get_text(pointer) == "<html>chart</html>"
    assert store.path_for(pointer).parent == store.root


def test_put_same_name_twice_does_not_clobber_the_first(store: ArtifactStore) -> None:
    """Two subtasks both naming their output `chart.png` must both survive."""
    first = store.put_text("chart.png", "first")
    second = store.put_text("chart.png", "second")

    assert first == f"{ARTIFACT_PREFIX}chart.png"
    assert second == f"{ARTIFACT_PREFIX}chart-1.png"
    assert store.get_text(first) == "first"
    assert store.get_text(second) == "second"


def test_concurrent_puts_of_one_name_all_survive(store: ArtifactStore) -> None:
    """Regression: a check-then-write `exists()` loop hands two threads the same name and
    loses a payload. The engine dispatches subtasks concurrently through `to_thread` (§10)."""
    payloads = [f"payload-{index}" for index in range(16)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        pointers = list(pool.map(lambda text: store.put_text("chart.png", text), payloads))

    assert len(set(pointers)) == len(payloads)
    assert sorted(store.get_text(pointer) for pointer in pointers) == sorted(payloads)


def test_path_for_returns_the_file_on_disk(store: ArtifactStore) -> None:
    """The renderer needs a path to show the user, not the bytes."""
    path = store.path_for(store.put_text("revenue.csv", CSV))

    assert path.is_file()
    assert path.read_text(encoding="utf-8") == CSV


def test_preview_of_short_text_returns_it_whole(store: ArtifactStore) -> None:
    """Under the limit there is nothing to elide, and a marker would be noise in a prompt."""
    pointer = store.put_text("revenue.csv", CSV)

    assert store.preview(pointer) == CSV


def test_preview_of_oversized_text_elides_and_names_the_omitted_size(store: ArtifactStore) -> None:
    """What keeps a hundred-thousand-row CSV out of the aggregator's prompt (#8)."""
    pointer = store.put_text("rows.csv", "x" * 1_000)

    preview = store.preview(pointer, limit=100)

    assert preview.startswith("x" * 100)
    assert preview.endswith("[elided, 900 more bytes]")
    assert "x" * 101 not in preview


def test_preview_of_text_exactly_at_the_limit_is_not_elided(store: ArtifactStore) -> None:
    """The boundary of the elision rule, pinned from both sides: `limit` characters are
    the whole payload, and one more is what buys the marker."""
    pointer = store.put_text("exact.csv", "x" * 100)

    assert store.preview(pointer, limit=100) == "x" * 100
    assert store.preview(pointer, limit=99) == "x" * 99 + "\n... [elided, 1 more bytes]"


def test_preview_of_an_empty_artifact_is_empty(store: ArtifactStore) -> None:
    """A worker that wrote nothing shows the aggregator nothing — not a marker claiming
    bytes were held back, and not `<binary, 0 bytes>`."""
    pointer = store.put_text("empty.csv", "")

    assert store.preview(pointer) == ""


def test_preview_of_multibyte_text_is_still_read_as_text(store: ArtifactStore) -> None:
    """Regression: the head read cuts mid-character, which a strict decode would call
    binary. The bytes are counted, so the omitted size stays honest for non-ASCII too."""
    pointer = store.put_text("prices.csv", "€" * 500)  # 3 bytes each

    preview = store.preview(pointer, limit=100)

    assert preview.startswith("€" * 100)
    assert preview.endswith("[elided, 1200 more bytes]")


def test_preview_of_a_binary_payload_reports_its_size_instead_of_its_bytes(
    store: ArtifactStore,
) -> None:
    """A chart is opened, not read: replacement characters would cost tokens and say
    nothing."""
    pointer = store.put_bytes("chart.png", b"\x89PNG\r\n\x1a\n\xff\xd8\xff")

    assert store.preview(pointer) == "<binary, 11 bytes>"


def test_preview_of_unknown_pointer_raises_task_failure(store: ArtifactStore) -> None:
    """One error path: the same `_resolve` every other read goes through."""
    with pytest.raises(TaskFailure, match="Artifact not found") as exc_info:
        store.preview(f"{ARTIFACT_PREFIX}never-written.csv")

    assert exc_info.value.exit_code == ExitCode.TASK_FAILURE


def test_get_unknown_pointer_raises_task_failure(store: ArtifactStore) -> None:
    with pytest.raises(TaskFailure) as exc_info:
        store.get_text(f"{ARTIFACT_PREFIX}never-written.csv")

    assert exc_info.value.exit_code == ExitCode.TASK_FAILURE


def test_get_string_that_is_not_a_pointer_raises_task_failure(store: ArtifactStore) -> None:
    """A raw blob where a pointer belongs is a contract breach, not a cache miss."""
    with pytest.raises(TaskFailure, match="Not an artifact pointer"):
        store.get_text(CSV)


def test_path_for_rejects_a_malformed_pointer(store: ArtifactStore) -> None:
    with pytest.raises(TaskFailure, match="Not an artifact pointer"):
        store.path_for("/etc/passwd")


def test_pointer_escaping_the_root_is_rejected(store: ArtifactStore, tmp_path: Path) -> None:
    """Names come from model output, so traversal is a trust boundary (§7)."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do not read me", encoding="utf-8")

    with pytest.raises(TaskFailure, match="Unsafe artifact name") as exc_info:
        store.get_text(f"{ARTIFACT_PREFIX}../secret.txt")

    assert "do not read me" not in str(exc_info.value)  # nothing outside root was read


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.csv",  # traversal
        "nested/chart.png",  # separator
        "back\\slash.png",  # separator, Windows
        "C:drive-relative.csv",  # drive-relative, resolves outside root on Windows
        "nul\x00byte.csv",  # fails inside open(), not as an OSError
        "",  # empty
        ".",
        "..",
    ],
)
def test_put_with_an_unsafe_name_is_rejected(store: ArtifactStore, name: str) -> None:
    with pytest.raises(TaskFailure, match="Unsafe artifact name"):
        store.put_text(name, CSV)


def test_put_file_with_a_missing_source_raises_task_failure(
    store: ArtifactStore, tmp_path: Path
) -> None:
    """A filesystem error leaves as a `TaskFailure`, never a bare `OSError`."""
    with pytest.raises(TaskFailure, match="could not copy") as exc_info:
        store.put_file(tmp_path / "never-written.html")

    assert exc_info.value.exit_code == ExitCode.TASK_FAILURE


def test_store_creates_its_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "artifacts"

    assert ArtifactStore(root).root.is_dir()


def test_unusable_root_raises_config_error_at_construction(tmp_path: Path) -> None:
    """§9 fail-fast: a bad artifact path is a startup problem, not a mid-run surprise."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        ArtifactStore(blocker / "artifacts")

    assert exc_info.value.exit_code == ExitCode.CONFIG
