"""Tests for the Data Retrieval agent's two tools.

Nothing here touches the network. The happy paths read the committed files under `data/`,
so the shipped dataset and corpus are exercised rather than a fixture that agrees with the
code; the error paths build their own files under `tmp_path`.

Paths are resolved from `__file__`: `conftest._isolated_env` chdirs every test into
`tmp_path`, so a relative `data/` would resolve to nothing.

Both tools are async over `asyncio.to_thread`, so both get a cancellation test. Those swap
the blocking read for a `BlockedRead` that parks in the worker thread — a real eight-row
read is over too fast for the cancellation to land in flight.
"""

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

from conftest import tool_call
from orchestra.artifacts import ArtifactStore
from orchestra.tools import fetch_data as fetch_data_module
from orchestra.tools import search as search_module
from orchestra.tools.base import BaseTool
from orchestra.tools.fetch_data import (
    INLINE_MAX_BYTES,
    INLINED_KEY,
    POINTER_KEY,
    Dataset,
    FetchDataTool,
    discover_datasets,
)
from orchestra.tools.search import MAX_RESULTS, SearchTool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNIPPETS = DATA_DIR / "search_snippets.json"

# Ceiling on every wait in this file. Long enough that a loaded machine does not flake,
# short enough that a swallowed cancellation fails the suite instead of hanging it.
TIMEOUT = 5.0


def rows(content: str) -> list[list[str]]:
    """Split a tool's CSV answer into fields, header included."""
    return [line.split(",") for line in content.strip().splitlines()]


def properties(schema: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """A JSON Schema's `properties` block, narrowed so a test can index into it.

    `ToolSpec.input_schema` is `Mapping[str, object]` because it crosses into
    `providers/` where `Any` is banned (§7); mypy needs the narrowing spelled out.
    """
    block = schema["properties"]
    assert isinstance(block, dict)
    return block


@dataclass(frozen=True, slots=True)
class BlockedRead:
    """A stand-in for a tool's blocking file read that parks in its worker thread.

    Cancellation can only land on `await asyncio.to_thread(...)` while that read is still
    running, so a cancellation test has to hold one there.
    """

    started: threading.Event
    release: threading.Event
    result: object

    def __call__(self, *args: object) -> object:
        """`*args`: it stands in for reads of different arities — one path, or a store
        and a dataset."""
        self.started.set()
        self.release.wait(timeout=TIMEOUT)
        return self.result


@pytest_asyncio.fixture
async def blocked_read() -> AsyncIterator[Callable[[object], BlockedRead]]:
    """Hand out `BlockedRead`s and release every one of them on teardown.

    Released unconditionally: cancelling the task does not stop the thread, and the default
    executor's threads are joined when the loop closes, so a read left parked stalls the
    suite for its whole timeout. Async purely for ordering — an async fixture is finalised
    while the loop is still open, so the release lands before that join.
    """
    reads: list[BlockedRead] = []

    def make(result: object) -> BlockedRead:
        read = BlockedRead(threading.Event(), threading.Event(), result)
        reads.append(read)
        return read

    yield make
    for read in reads:
        read.release.set()


@pytest.fixture
def bundled(store: ArtifactStore) -> FetchDataTool:
    """The tool over the committed `data/`, minus the corpus `agents/toolsets.py` hides."""
    return FetchDataTool(
        store, [dataset for dataset in discover_datasets(DATA_DIR) if dataset.path != SNIPPETS]
    )


@pytest.fixture
def search_tool() -> SearchTool:
    return SearchTool(SNIPPETS)


if TYPE_CHECKING:
    # Conformance is mypy's job, not `isinstance`'s: `BaseTool` is a plain Protocol, and a
    # runtime check would compare attribute names only (§7).
    _FETCH_IS_A_TOOL: BaseTool = FetchDataTool(ArtifactStore(DATA_DIR), ())
    _SEARCH_IS_A_TOOL: BaseTool = SearchTool(SNIPPETS)


# -------------------------------------------------------------------------- fetch_data


def test_fetch_data_info_advertises_its_params_schema(bundled: FetchDataTool) -> None:
    """Asserted field by field rather than against `FetchDataParams.model_json_schema()`,
    which is the expression `info()` returns — that comparison passes whatever the schema
    says, including nothing."""
    spec = bundled.info()

    assert spec.name == "fetch_data"
    assert set(properties(spec.input_schema)) == {"name"}
    # `extra="forbid"` reaches the model, so the provider rejects an invented `columns`
    # client-side as well as `run` does.
    assert spec.input_schema["additionalProperties"] is False


def test_fetch_data_info_names_every_bundled_dataset(bundled: FetchDataTool) -> None:
    """The description is the only place the model learns what it may ask for."""
    description = bundled.info().description

    assert "quarterly_financials" in description and "expense_breakdown" in description
    assert "quarter, revenue, costs, profit" in description
    # A two-tool agent only stays a two-tool agent if each prompt names the other.
    assert "search" in description


def test_fetch_data_info_is_pure(bundled: FetchDataTool, store: ArtifactStore) -> None:
    """It runs every turn, so it must not touch disk: the same object, and no artifact."""
    assert bundled.info() is bundled.info()
    assert list(store.root.iterdir()) == []


def test_fetch_data_provides_names_the_datasets_and_keeps_its_boundary(
    bundled: FetchDataTool,
) -> None:
    """The planner plans against this (#10). Without the boundary a request for data
    nobody holds is planned as three steps that then retrieve nothing."""
    provides = bundled.info().provides

    assert "expense_breakdown" in provides and "quarter, category, amount" in provides
    assert "nothing beyond those files" in provides


def test_fetch_data_with_an_empty_catalogue_provides_nothing(store: ArtifactStore) -> None:
    """`retrievable_data` skips a tool that supplies none, so the planner is correctly
    told the team can fetch nothing — an empty string, not a sentence about no files."""
    tool = FetchDataTool(store, ())

    assert tool.info().provides == ""
    assert tool.info().description.strip()  # still a prompt: it has to say why


@pytest.mark.asyncio
async def test_fetch_data_returns_a_small_file_whole(
    bundled: FetchDataTool, store: ArtifactStore
) -> None:
    """Exercises the committed dataset: eight quarters, and profit that adds up."""
    response = await bundled.run(tool_call("fetch_data", name="quarterly_financials"))

    assert not response.is_error
    header, *data = rows(response.content)
    assert header == ["quarter", "revenue", "costs", "profit"]
    assert [row[0] for row in data] == [
        "2024Q1",
        "2024Q2",
        "2024Q3",
        "2024Q4",
        "2025Q1",
        "2025Q2",
        "2025Q3",
        "2025Q4",
    ]
    assert all(int(row[3]) == int(row[1]) - int(row[2]) for row in data)
    # Registered even when inlined, so the analysis step can open the file itself.
    assert response.metadata[INLINED_KEY] == "true"
    assert store.get_text(response.metadata[POINTER_KEY]) == response.content


@pytest.mark.asyncio
async def test_fetch_data_second_bundled_dataset_reconciles_with_the_first(
    bundled: FetchDataTool,
) -> None:
    """Two datasets is what proves the catalogue: the breakdown sums to the `costs`
    column it was written against, so a join across both is a real analysis."""
    expenses = await bundled.run(tool_call("fetch_data", name="expense_breakdown"))
    financials = await bundled.run(tool_call("fetch_data", name="quarterly_financials"))

    header, *breakdown = rows(expenses.content)
    assert header == ["quarter", "category", "amount"]
    costs = {row[0]: int(row[2]) for row in rows(financials.content)[1:]}
    totals: dict[str, int] = {}
    for quarter, _category, amount in breakdown:
        totals[quarter] = totals.get(quarter, 0) + int(amount)
    assert totals == costs


@pytest.mark.asyncio
async def test_fetch_data_large_file_returns_a_pointer_instead_of_rows(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """#40: a tool result is re-sent every later turn, so past the threshold the file is
    handed on by pointer and the analysis step opens it."""
    big = tmp_path / "big.csv"
    big.write_text("quarter,revenue\n" + "2025Q1,1\n" * INLINE_MAX_BYTES, encoding="utf-8")
    tool = FetchDataTool(store, discover_datasets(tmp_path))

    response = await tool.run(tool_call("fetch_data", name="big"))

    assert not response.is_error
    assert response.metadata[INLINED_KEY] == "false"
    pointer = response.metadata[POINTER_KEY]
    # The schema, the pointer and what to do with it — and none of the rows.
    assert "quarter, revenue" in response.content
    assert pointer in response.content and "analysis step" in response.content
    assert "2025Q1,1" not in response.content
    assert "not UTF-8" not in response.content  # the other reason a file is withheld
    assert store.get_text(pointer) == big.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_fetch_data_small_binary_file_is_withheld_as_binary_not_as_too_large(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """Regression: both reasons hand over a pointer, but "too large" said of a 12-byte
    file is a false claim, and the retrieval summary repeats what the tool said into the
    report. The `.parquet` a real deployment would hold takes this path."""
    (tmp_path / "readings.parquet").write_bytes(b"PAR1\x00\xff\xfe binary")
    tool = FetchDataTool(store, discover_datasets(tmp_path))

    response = await tool.run(tool_call("fetch_data", name="readings"))

    assert not response.is_error and not response.is_empty
    assert response.metadata[INLINED_KEY] == "false"
    assert "not UTF-8 text" in response.content
    assert "too large" not in response.content and str(INLINE_MAX_BYTES) not in response.content
    assert store.path_for(response.metadata[POINTER_KEY]).read_bytes() == b"PAR1\x00\xff\xfe binary"


@pytest.mark.parametrize("content", ["", "\n  \n"], ids=["zero-byte", "whitespace-only"])
@pytest.mark.asyncio
async def test_fetch_data_empty_file_reports_an_empty_result(
    tmp_path: Path, store: ArtifactStore, content: str
) -> None:
    """Regression: an empty file is not a successful fetch of nothing. Empty `content`
    reads to the model as a broken tool, and the worker would store it as rows the
    analysis step then hands to `pd.read_csv`, which raises `EmptyDataError`.

    `is_empty`, not `is_error`, as in `search`: the file will be just as empty next time,
    so there is no retry to invite — but the loop must drop it either way."""
    (tmp_path / "nothing.csv").write_text(content, encoding="utf-8")
    tool = FetchDataTool(store, discover_datasets(tmp_path))

    response = await tool.run(tool_call("fetch_data", name="nothing"))

    assert response.is_empty and not response.is_error
    assert response.content.strip()  # a sentence, not the file's own emptiness
    assert "empty file" in response.content


@pytest.mark.asyncio
async def test_fetch_data_repairs_a_filename_the_artifact_store_would_refuse(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """Regression: `ARTIFACT_NAME_PATTERN` admits no `&` or `(`, so a catalogue built from
    raw filenames advertises datasets whose every call dies in `put_file` — the #10 failure
    the boundary clause exists to prevent. The operator's file works instead."""
    (tmp_path / "Q3 P&L (final).csv").write_text("quarter,profit\n2025Q3,7\n", encoding="utf-8")
    datasets = discover_datasets(tmp_path)
    tool = FetchDataTool(store, datasets)

    # The model still names the file the way the operator did.
    response = await tool.run(tool_call("fetch_data", name="Q3 P&L (final)"))

    assert not response.is_error
    assert "2025Q3,7" in response.content
    # Stored under a repaired name, so the pointer is one the store and `run_python` accept.
    pointer = response.metadata[POINTER_KEY]
    assert pointer == "artifact:Q3 P_L _final_.csv"
    assert store.get_text(pointer) == "quarter,profit\n2025Q3,7\n"


@pytest.mark.asyncio
async def test_fetch_data_registers_one_pointer_however_often_it_is_fetched(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """Two fetches of one dataset are one artifact: re-registering would leave `x.csv` and
    `x-1.csv`, two pointers for one file and two copies of a large one."""
    (tmp_path / "sales.csv").write_text("quarter,revenue\n2025Q1,1\n", encoding="utf-8")
    tool = FetchDataTool(store, discover_datasets(tmp_path))

    first = await tool.run(tool_call("fetch_data", name="sales"))
    second = await tool.run(tool_call("fetch_data", name="sales"))

    assert first.metadata[POINTER_KEY] == second.metadata[POINTER_KEY]
    assert [path.name for path in store.root.iterdir()] == ["sales.csv"]


@pytest.mark.asyncio
async def test_fetch_data_unknown_name_lists_the_datasets(bundled: FetchDataTool) -> None:
    """The error is the model's next prompt, so it has to contain the retry (§6)."""
    response = await bundled.run(tool_call("fetch_data", name="headcount"))

    assert response.is_error
    assert "headcount" in response.content
    assert "quarterly_financials" in response.content


@pytest.mark.asyncio
async def test_fetch_data_unknown_name_with_no_datasets_says_so(store: ArtifactStore) -> None:
    """An installed wheel points at nothing; the model is told that, not given a list."""
    response = await FetchDataTool(store, ()).run(tool_call("fetch_data", name="anything"))

    assert response.is_error
    assert "no data files" in response.content


@pytest.mark.asyncio
async def test_fetch_data_missing_name_reports_the_validation_message(
    bundled: FetchDataTool,
) -> None:
    response = await bundled.run(tool_call("fetch_data"))

    assert response.is_error
    assert "name" in response.content


@pytest.mark.asyncio
async def test_fetch_data_unknown_argument_is_rejected(bundled: FetchDataTool) -> None:
    """`extra="forbid"`: an invented argument is reported, never silently dropped."""
    response = await bundled.run(tool_call("fetch_data", name="quarterly_financials", last_n=3))

    assert response.is_error
    assert "last_n" in response.content


@pytest.mark.asyncio
async def test_fetch_data_file_deleted_after_startup_is_content_not_an_exception(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """The catalogue is probed once; the file can go away afterwards (§6)."""
    dataset = tmp_path / "gone.csv"
    dataset.write_text("quarter,revenue\n2025Q1,1\n", encoding="utf-8")
    tool = FetchDataTool(store, discover_datasets(tmp_path))
    dataset.unlink()

    response = await tool.run(tool_call("fetch_data", name="gone"))

    assert response.is_error
    assert str(dataset) in response.content


@pytest.mark.asyncio
async def test_fetch_data_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    blocked_read: Callable[[object], BlockedRead],
    store: ArtifactStore,
) -> None:
    """§10: `CancelledError` is the only thing that may leave `run`. The read's `except`
    list is what could swallow it — widening to `BaseException`, or to a bare `except`
    (§8), fails this test."""
    read = blocked_read(("artifact:x.csv", "quarter\n2024Q1\n"))
    monkeypatch.setattr(fetch_data_module, "_fetch", read)
    tool = FetchDataTool(store, discover_datasets(DATA_DIR))
    task = asyncio.create_task(tool.run(tool_call("fetch_data", name="quarterly_financials")))

    assert await asyncio.to_thread(read.started.wait, TIMEOUT)  # parked inside the try
    task.cancel()

    # A handler that caught the cancellation returns a `ToolResponse` and fails here
    # rather than hanging: the read is released on fixture teardown either way.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TIMEOUT)


# --------------------------------------------------------------------- discover_datasets


def test_discover_datasets_lists_every_file_sorted(tmp_path: Path) -> None:
    """Sorted, so the roster the planner sees does not depend on directory order."""
    (tmp_path / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "a.json").write_text('{"k": 1}', encoding="utf-8")
    (tmp_path / "sub").mkdir()  # directories are not datasets
    (tmp_path / ".DS_Store").write_text("junk", encoding="utf-8")  # nor is the platform's

    assert [dataset.name for dataset in discover_datasets(tmp_path)] == ["a", "b"]


def test_discover_datasets_missing_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    """Installed as a wheel the default points at nothing; the agent still has `search`."""
    assert discover_datasets(tmp_path / "never-created") == ()


def test_discover_datasets_skips_a_file_it_cannot_probe(tmp_path: Path) -> None:
    """One bad file must not cost the run its other data."""
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "fine.csv").write_text("x\n1\n", encoding="utf-8")

    assert [dataset.name for dataset in discover_datasets(tmp_path)] == ["fine"]


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("t.csv", "quarter,revenue\n2025Q1,1\n", "CSV with columns quarter, revenue"),
        ("t.tsv", "quarter\trevenue\n2025Q1\t1\n", "TSV with columns quarter, revenue"),
        ("t.json", '[{"a": 1, "b": 2}]', "JSON list of 1 entries, an object with keys a, b each"),
        ("t.json", '{"a": 1, "b": 2}', "an object with keys a, b"),
        ("t.jsonl", '{"a": 1}\n{"a": 2}\n', "JSON lines, an object with keys a per line"),
        ("t.md", "# Title\nbody\n", "beginning '# Title'"),
        ("t.parquet", "\x00binary", ".parquet file"),
    ],
)
def test_discover_datasets_probes_by_suffix(
    tmp_path: Path, filename: str, content: str, expected: str
) -> None:
    """The summary is what both the model and the planner are told a file holds, so each
    suffix has to say something more useful than its size."""
    (tmp_path / filename).write_text(content, encoding="utf-8")

    (dataset,) = discover_datasets(tmp_path)
    assert expected in dataset.summary


def test_discover_datasets_does_not_read_a_large_json_body(tmp_path: Path) -> None:
    """The probe runs at startup for every file, so it is bounded: past the cap a JSON
    document is described by size rather than parsed."""
    body = '{"a": ' + "1" * (fetch_data_module.PROBE_MAX_BYTES + 1) + "}"
    (tmp_path / "huge.json").write_text(body, encoding="utf-8")

    (dataset,) = discover_datasets(tmp_path)
    assert "too large to probe" in dataset.summary


def test_discover_datasets_keys_a_colliding_stem_on_the_whole_filename(tmp_path: Path) -> None:
    """Two formats of one export are two datasets. Dropping the second would need a
    warning this module has nowhere to emit, so the loser keeps its extension instead."""
    (tmp_path / "sales.csv").write_text("quarter,revenue\n", encoding="utf-8")
    (tmp_path / "sales.xlsx").write_bytes(b"PK\x03\x04binary")

    assert [dataset.name for dataset in discover_datasets(tmp_path)] == ["sales", "sales.xlsx"]


def test_discover_datasets_skips_a_file_whose_name_cannot_be_repaired(tmp_path: Path) -> None:
    """The other half of the repair: a name with nothing legal left is not offered, rather
    than listed and then refused by the store on every call."""
    (tmp_path / "....csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "fine.csv").write_text("x\n", encoding="utf-8")

    assert [dataset.name for dataset in discover_datasets(tmp_path)] == ["fine"]


def test_discover_datasets_reads_no_more_than_the_probe_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 MB export on one line is one `readline`. The bound applies to every branch,
    not only to the JSON one whose parse made the cost obvious."""
    monkeypatch.setattr(fetch_data_module, "PROBE_MAX_BYTES", 20)
    (tmp_path / "wide.txt").write_text("a" * 10_000, encoding="utf-8")

    (dataset,) = discover_datasets(tmp_path)
    assert "a" * 20 in dataset.summary
    assert "a" * 21 not in dataset.summary


def test_catalogue_caps_the_datasets_it_lists(tmp_path: Path, store: ArtifactStore) -> None:
    """The catalogue rides in every retrieval turn and every planner call, so its size is
    this module's to bound rather than the directory's to decide."""
    for index in range(fetch_data_module.MAX_LISTED_DATASETS + 3):
        (tmp_path / f"set{index:02d}.csv").write_text("quarter,revenue\n", encoding="utf-8")
    tool = FetchDataTool(store, discover_datasets(tmp_path))

    provides = tool.info().provides
    assert provides.count("; ") == fetch_data_module.MAX_LISTED_DATASETS
    assert "and 3 more not listed here" in provides
    # And the boundary survives the elision, or the planner loses what stops it (#10).
    assert "nothing beyond those files" in provides


def test_catalogue_elides_a_very_wide_files_column_list(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """One CSV with 300 columns must not spend the prompt every other dataset needs."""
    (tmp_path / "wide.csv").write_text(
        ",".join(f"column_{index}" for index in range(300)) + "\n", encoding="utf-8"
    )
    tool = FetchDataTool(store, discover_datasets(tmp_path))

    provides = tool.info().provides
    assert "[elided]" in provides
    assert len(provides) < fetch_data_module.MAX_SUMMARY_CHARS + 200


def test_dataset_is_an_immutable_value_object() -> None:
    """§7: an internal value object, so a caller cannot re-point one at another file."""
    dataset = Dataset(name="t", path=Path("t.csv"), summary="CSV", store_name="t.csv")

    with pytest.raises(AttributeError):
        dataset.path = Path("other.csv")  # type: ignore[misc]


# ------------------------------------------------------------------------------ search


def test_search_info_advertises_its_params_schema(search_tool: SearchTool) -> None:
    """The published shape, not the expression `info()` returns."""
    spec = search_tool.info()

    assert spec.name == "search"
    assert set(properties(spec.input_schema)) == {"query", "limit"}
    assert spec.input_schema["additionalProperties"] is False
    # The cap is in the schema so the model reads it before choosing, not in an error.
    assert properties(spec.input_schema)["limit"]["maximum"] == MAX_RESULTS


def test_search_info_description_holds_for_either_backend(search_tool: SearchTool) -> None:
    """One description is shown whichever backend is configured, so it must fit both. It
    used to promise no internet access, which became false the moment a key could be
    configured."""
    description = search_tool.info().description

    assert "fetch_data" in description
    assert "illustrative" in description
    # Neither backend may be promised, because only one of them is ever present.
    assert "NOT reach the internet" not in description


@pytest.mark.asyncio
async def test_search_matching_query_returns_the_bundled_note(search_tool: SearchTool) -> None:
    """Exercises the committed corpus through the real scoring path."""
    response = await search_tool.run(tool_call("search", query="typical gross margin for software"))

    assert not response.is_error and not response.is_empty
    assert "margin ranges" in response.content
    assert "75-85%" in response.content


@pytest.mark.asyncio
async def test_search_result_preamble_disclaims_the_corpus_as_unsourced(
    search_tool: SearchTool,
) -> None:
    """The notes state invented specifics as fact and the model will quote them; the
    preamble is the only thing keeping them out of a report that forbids unsourced
    figures."""
    response = await search_tool.run(tool_call("search", query="growth benchmarks"))

    preamble = response.content.split("\n\n")[0]
    assert "illustrative sample data" in preamble
    assert "not sourced research" in preamble
    assert "do not quote its numbers as fact" in preamble


@pytest.mark.asyncio
async def test_search_uppercase_keyword_in_the_corpus_still_matches(tmp_path: Path) -> None:
    """`_rank` intersects a lowercased token set, so `"Margin"` would never match.
    Normalised on load rather than rejected: a typo must not cost every note."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        '[{"keywords": ["Margin", "GROSS"], "title": "Margins", "snippet": "a"}]',
        encoding="utf-8",
    )

    response = await SearchTool(corpus).run(tool_call("search", query="gross margin"))

    assert not response.is_error and not response.is_empty
    assert "Margins" in response.content


@pytest.mark.asyncio
async def test_search_limit_caps_the_number_of_notes_returned(search_tool: SearchTool) -> None:
    broad = "quarterly cost margin growth seasonality macro benchmarks leverage"

    one = await search_tool.run(tool_call("search", query=broad, limit=1))
    two = await search_tool.run(tool_call("search", query=broad, limit=2))

    assert one.content.count("[1]") == 1 and "[2]" not in one.content
    assert "[2]" in two.content and "[3]" not in two.content


@pytest.mark.asyncio
async def test_search_ranks_the_entry_matching_more_keywords_first(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        '[{"keywords": ["cost"], "title": "One hit", "snippet": "a"},'
        ' {"keywords": ["cost", "margin"], "title": "Two hits", "snippet": "b"}]',
        encoding="utf-8",
    )

    response = await SearchTool(corpus).run(tool_call("search", query="cost and margin"))

    assert response.content.index("Two hits") < response.content.index("One hit")


@pytest.mark.asyncio
async def test_search_no_match_is_empty_not_an_error(search_tool: SearchTool) -> None:
    """Flagging a clean miss as an error reads to the model as "the tool broke" and buys a
    pointless retry. `is_empty` is the other half: without it a caller asking "did this
    step retrieve anything?" records "nothing matched" as provenance."""
    response = await search_tool.run(tool_call("search", query="zzzz nonexistent topic"))

    assert not response.is_error
    assert response.is_empty
    assert "Nothing in the offline corpus matched" in response.content
    assert "Seasonality" in response.content  # the retry is told what is there


@pytest.mark.asyncio
async def test_search_malformed_corpus_is_an_error_not_a_crash(tmp_path: Path) -> None:
    """A JSON file on disk is a trust boundary like any other (§7)."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text('{"notes": []}', encoding="utf-8")

    response = await SearchTool(corpus).run(tool_call("search", query="margin"))

    assert response.is_error
    assert str(corpus) in response.content


@pytest.mark.asyncio
async def test_search_corpus_with_a_bad_entry_is_an_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text('[{"title": "No keywords", "snippet": "a"}]', encoding="utf-8")

    response = await SearchTool(corpus).run(tool_call("search", query="margin"))

    assert response.is_error
    assert "keywords" in response.content


@pytest.mark.asyncio
async def test_search_unparseable_corpus_is_an_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[{", encoding="utf-8")

    response = await SearchTool(corpus).run(tool_call("search", query="margin"))

    assert response.is_error
    assert "malformed" in response.content


@pytest.mark.asyncio
async def test_search_missing_corpus_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "never-written.json"

    response = await SearchTool(missing).run(tool_call("search", query="margin"))

    assert response.is_error
    assert str(missing) in response.content


@pytest.mark.asyncio
async def test_search_empty_query_reports_the_validation_message(search_tool: SearchTool) -> None:
    response = await search_tool.run(tool_call("search", query=""))

    assert response.is_error
    assert "query" in response.content


@pytest.mark.asyncio
async def test_search_limit_above_the_cap_is_rejected(search_tool: SearchTool) -> None:
    """The cap is in the schema, so exceeding it comes back naming the cap."""
    response = await search_tool.run(tool_call("search", query="margin", limit=MAX_RESULTS + 1))

    assert response.is_error
    assert "limit" in response.content


@pytest.mark.asyncio
async def test_search_reads_the_corpus_once_across_calls(tmp_path: Path) -> None:
    """The corpus is cached after the first call — the second must not touch the disk."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        '[{"keywords": ["margin"], "title": "Margins", "snippet": "a"}]', encoding="utf-8"
    )
    tool = SearchTool(corpus)

    first = await tool.run(tool_call("search", query="margin"))
    corpus.unlink()
    second = await tool.run(tool_call("search", query="margin"))

    assert not first.is_error and not second.is_error
    assert first.content == second.content


@pytest.mark.asyncio
async def test_search_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch, blocked_read: Callable[[object], BlockedRead]
) -> None:
    """§10: the corpus load is cancellable and neither handler around it absorbs the
    cancellation. As above, the mutation this catches is a widening to `BaseException`."""
    read = blocked_read([])
    monkeypatch.setattr(search_module, "_read_corpus", read)
    task = asyncio.create_task(SearchTool(SNIPPETS).run(tool_call("search", query="margin")))

    assert await asyncio.to_thread(read.started.wait, TIMEOUT)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TIMEOUT)


# --------------------------------------------------------------------------
# The live search backend. Still no network: `httpx.AsyncClient` is stubbed, so
# what is under test is the request built, the payload validated, and the
# fallback taken (§12).
# --------------------------------------------------------------------------

LIVE_KEY = SecretStr("tvly-test-key")
LIVE_PAYLOAD = json.dumps(
    {
        "query": "saas margins",
        "results": [
            {
                "title": "Gross margin benchmarks",
                "url": "https://example.com/margins",
                "content": "Median gross margin was 74%.",
                "score": 0.9,
                "an_unexpected_field": "the vendor added this",
            }
        ],
    }
)


class StubResponse:
    """Stands in for `httpx.Response`, with the two members the tool reads."""

    def __init__(self, body: str, status_code: int = 200) -> None:
        self.content = body.encode()
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        request = httpx.Request("POST", search_module.TAVILY_URL)
        raise httpx.HTTPStatusError(
            "error", request=request, response=httpx.Response(self.status_code, request=request)
        )


def stub_httpx(monkeypatch: pytest.MonkeyPatch, result: object) -> list[dict[str, object]]:
    """Replace `httpx.AsyncClient` and record what the tool posted.

    Patched on `httpx` itself rather than injected: the tool builds a client per call
    (`BaseTool` has no close hook), so there is no seam to inject through.
    """
    posts: list[dict[str, object]] = []

    class StubClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "StubClient":
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def post(self, url: str, *, headers: Mapping[str, str], json: object) -> object:
            posts.append({"url": url, "headers": dict(headers), "json": json, "init": self.kwargs})
            if isinstance(result, BaseException):
                raise result
            return result

    monkeypatch.setattr(httpx, "AsyncClient", StubClient)
    return posts


def live_tool() -> SearchTool:
    """The same corpus plus a key, so a fallback has somewhere real to land."""
    return SearchTool(SNIPPETS, api_key=LIVE_KEY)


@pytest.mark.asyncio
async def test_search_with_a_key_returns_live_results_marked_as_sourced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live result cites its page, so unlike a corpus note it may be quoted."""
    stub_httpx(monkeypatch, StubResponse(LIVE_PAYLOAD))

    response = await live_tool().run(tool_call("search", query="saas margins"))

    assert not response.is_error and not response.is_empty
    assert "https://example.com/margins" in response.content
    assert "Median gross margin was 74%." in response.content
    assert "live web search" in response.content
    assert "illustrative" not in response.content  # no corpus disclaimer on sourced results


@pytest.mark.asyncio
async def test_search_sends_the_key_as_a_bearer_token_and_bounds_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wire shape, asserted once: auth header, endpoint, body, and a timeout (§10)."""
    posts = stub_httpx(monkeypatch, StubResponse(LIVE_PAYLOAD))

    await live_tool().run(tool_call("search", query="saas margins", limit=5))

    assert posts[0]["url"] == search_module.TAVILY_URL
    assert posts[0]["headers"] == {"Authorization": "Bearer tvly-test-key"}
    assert posts[0]["json"] == {
        "query": "saas margins",
        "max_results": 5,
        "search_depth": "basic",
    }
    assert posts[0]["init"] == {"timeout": search_module.DEFAULT_TIMEOUT}


@pytest.mark.asyncio
async def test_search_ignores_unknown_fields_in_the_live_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vendor owns that schema. A field they add must not break the tool (§7)."""
    stub_httpx(monkeypatch, StubResponse(LIVE_PAYLOAD))

    response = await live_tool().run(tool_call("search", query="saas margins"))

    assert not response.is_error


@pytest.mark.asyncio
async def test_search_live_with_no_results_is_empty_not_a_corpus_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web answered and had nothing — that is an answer, not a reason to fall back."""
    stub_httpx(monkeypatch, StubResponse(json.dumps({"results": []})))

    response = await live_tool().run(tool_call("search", query="nothing at all"))

    assert response.is_empty and not response.is_error
    assert "no usable results" in response.content
    assert "illustrative" not in response.content  # not the corpus, quietly


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (StubResponse("", status_code=401), "HTTP 401"),
        (httpx.ConnectTimeout("timed out"), "ConnectTimeout"),
        (StubResponse("{not json"), "unexpected response shape"),
    ],
    ids=["rejected", "timed_out", "malformed"],
)
async def test_search_falls_back_to_the_corpus_when_the_live_backend_fails(
    monkeypatch: pytest.MonkeyPatch, failure: object, expected: str
) -> None:
    """A search outage degrades the run rather than ending a subtask that has a corpus. The
    notice matters as much as the fallback: without it the model reads bundled sample data
    as the live results it asked for."""
    stub_httpx(monkeypatch, failure)

    response = await live_tool().run(tool_call("search", query="typical gross margin for software"))

    assert not response.is_error
    assert "Live search was unavailable" in response.content
    assert expected in response.content
    assert "illustrative" in response.content  # the corpus really did answer


@pytest.mark.asyncio
async def test_search_failure_notice_never_carries_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§9: an auth failure is the one error most likely to echo the credential."""
    stub_httpx(monkeypatch, StubResponse("", status_code=401))

    response = await live_tool().run(tool_call("search", query="typical gross margin for software"))

    assert "tvly-test-key" not in response.content


@pytest.mark.asyncio
async def test_search_without_a_key_never_opens_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline has to mean no request was attempted, not one that failed."""
    posts = stub_httpx(monkeypatch, StubResponse(LIVE_PAYLOAD))

    response = await SearchTool(SNIPPETS).run(
        tool_call("search", query="typical gross margin for software")
    )

    assert posts == []
    assert "illustrative" in response.content
