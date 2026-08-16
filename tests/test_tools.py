"""Tests for the Data Retrieval agent's two tools (CONVENTIONS.md §12).

Nothing here touches the network — neither tool can. The happy paths read the committed
files under `data/`, so the shipped dataset and corpus are exercised rather than only a
fixture that agrees with the code; the error paths build their own files under
`tmp_path`.

Paths are resolved from `__file__`: the autouse fixture in `conftest.py` chdirs every
test into `tmp_path`, so a relative `data/` would resolve to nothing.

Both tools are async over `asyncio.to_thread`, so both get a cancellation test (§12).
Those swap the blocking read for a `BlockedRead` that parks in the worker thread — the
cancellation has to land while the read is in flight, which a real eight-row read is
over too fast to allow.
"""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from orchestra.tools import query_csv as query_csv_module
from orchestra.tools import search as search_module
from orchestra.tools.base import BaseTool, ToolCall
from orchestra.tools.query_csv import QueryCsvTool
from orchestra.tools.search import MAX_RESULTS, SearchTool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FINANCIALS = DATA_DIR / "quarterly_financials.csv"
SNIPPETS = DATA_DIR / "search_snippets.json"

# Ceiling on every wait in this file. Long enough that a loaded machine does not flake,
# short enough that a swallowed cancellation fails the suite instead of hanging it.
TIMEOUT = 5.0


def call(name: str, **arguments: object) -> ToolCall:
    """One tool call as a provider would decode it."""
    return ToolCall(id="call-1", name=name, arguments=arguments)


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

    Cancellation can only land on `await asyncio.to_thread(...)` while that read is
    still running, so a cancellation test has to hold one there.
    """

    started: threading.Event
    release: threading.Event
    result: object

    def __call__(self, path: Path) -> object:
        self.started.set()
        self.release.wait(timeout=TIMEOUT)
        return self.result


@pytest_asyncio.fixture
async def blocked_read() -> AsyncIterator[Callable[[object], BlockedRead]]:
    """Hand out `BlockedRead`s and release every one of them on teardown.

    Released unconditionally: cancelling the task does not stop the thread, and the
    default executor's threads are joined when the loop closes — a read left parked
    stalls the suite for its whole timeout instead of failing a test.

    Async purely for ordering: an async fixture is finalised while the loop is still
    open, so the release lands before that join rather than after it.
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
def csv_tool() -> QueryCsvTool:
    return QueryCsvTool(FINANCIALS)


@pytest.fixture
def search_tool() -> SearchTool:
    return SearchTool(SNIPPETS)


if TYPE_CHECKING:
    # Conformance to the port is mypy's job, not `isinstance`'s: `BaseTool` is a plain
    # Protocol, and a runtime check would compare attribute names only (§7).
    _CSV_IS_A_TOOL: BaseTool = QueryCsvTool(FINANCIALS)
    _SEARCH_IS_A_TOOL: BaseTool = SearchTool(SNIPPETS)


# --------------------------------------------------------------------------- query_csv


def test_query_csv_info_advertises_its_params_schema(csv_tool: QueryCsvTool) -> None:
    """The schema the model is shown must be the one `run` validates against (§6).

    Asserted field by field rather than against `QueryCsvParams.model_json_schema()`,
    which is the expression `info()` returns — that comparison passes whatever the
    schema says, including nothing.
    """
    spec = csv_tool.info()

    assert spec.name == "query_csv"
    assert set(properties(spec.input_schema)) == {"columns", "quarters", "last_n"}
    # `extra="forbid"` reaches the model, so an invented `where` is rejected client-side
    # by the provider as well as by `run`.
    assert spec.input_schema["additionalProperties"] is False


def test_query_csv_info_description_routes_general_questions_elsewhere(
    csv_tool: QueryCsvTool,
) -> None:
    """A two-tool agent only stays a two-tool agent if each prompt names the other."""
    assert "search" in csv_tool.info().description


@pytest.mark.asyncio
async def test_query_csv_no_filters_returns_the_whole_bundled_dataset(
    csv_tool: QueryCsvTool,
) -> None:
    """Exercises the committed dataset: eight quarters, and profit that actually adds up."""
    response = await csv_tool.run(call("query_csv"))

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


@pytest.mark.asyncio
async def test_query_csv_column_and_quarter_filter_returns_only_those_cells(
    csv_tool: QueryCsvTool,
) -> None:
    response = await csv_tool.run(
        call("query_csv", columns=["quarter", "profit"], quarters=["2025Q1"])
    )

    assert not response.is_error
    assert rows(response.content) == [["quarter", "profit"], ["2025Q1", "915000"]]


@pytest.mark.asyncio
async def test_query_csv_columns_are_returned_in_the_requested_order(
    csv_tool: QueryCsvTool,
) -> None:
    """The model reads its answer more easily in the shape it asked for."""
    response = await csv_tool.run(
        call("query_csv", columns=["profit", "quarter"], quarters=["2024Q1"])
    )

    assert rows(response.content) == [["profit", "quarter"], ["630000", "2024Q1"]]


@pytest.mark.asyncio
async def test_query_csv_last_n_keeps_the_most_recent_rows(csv_tool: QueryCsvTool) -> None:
    """The file is oldest-first, so "most recent" is the tail, not the head."""
    response = await csv_tool.run(call("query_csv", columns=["quarter"], last_n=2))

    assert rows(response.content) == [["quarter"], ["2025Q3"], ["2025Q4"]]


@pytest.mark.asyncio
async def test_query_csv_last_n_larger_than_the_dataset_returns_every_row(
    csv_tool: QueryCsvTool,
) -> None:
    """Asking for more than exists is not an error — it is the whole table."""
    response = await csv_tool.run(call("query_csv", columns=["quarter"], last_n=99))

    assert not response.is_error
    assert len(rows(response.content)) == 9  # header + 8 quarters


@pytest.mark.asyncio
async def test_query_csv_lowercase_quarter_still_matches(csv_tool: QueryCsvTool) -> None:
    """`2025q4` meant `2025Q4`; spending a turn correcting that teaches the model nothing."""
    response = await csv_tool.run(call("query_csv", columns=["quarter"], quarters=["2025q4"]))

    assert rows(response.content) == [["quarter"], ["2025Q4"]]


@pytest.mark.asyncio
async def test_query_csv_unknown_column_names_the_valid_columns(csv_tool: QueryCsvTool) -> None:
    """The error is the model's next prompt: it must contain the retry (§6)."""
    response = await csv_tool.run(call("query_csv", columns=["margin"]))

    assert response.is_error
    assert "margin" in response.content
    assert "quarter, revenue, costs, profit" in response.content


@pytest.mark.asyncio
async def test_query_csv_unknown_quarter_names_the_available_quarters(
    csv_tool: QueryCsvTool,
) -> None:
    response = await csv_tool.run(call("query_csv", quarters=["2023Q4"]))

    assert response.is_error
    assert "2023Q4" in response.content
    assert "2024Q1" in response.content and "2025Q4" in response.content


@pytest.mark.asyncio
async def test_query_csv_missing_dataset_file_names_the_path(tmp_path: Path) -> None:
    """A bad injection surfaces as content, not as an unwound agent loop."""
    missing = tmp_path / "never-written.csv"

    response = await QueryCsvTool(missing).run(call("query_csv"))

    assert response.is_error
    assert str(missing) in response.content


@pytest.mark.asyncio
async def test_query_csv_header_only_dataset_reports_an_empty_result(tmp_path: Path) -> None:
    """The empty-result path: valid file, valid arguments, nothing to return.

    An error here and `is_empty` in `search` (see below) — the asymmetry is deliberate.
    This dataset is fixed, so no rows means the model asked for something it does not
    hold and the message names what it does hold; the retry is the right next move.
    """
    dataset = tmp_path / "empty.csv"
    dataset.write_text("quarter,revenue,costs,profit\n", encoding="utf-8")

    response = await QueryCsvTool(dataset).run(call("query_csv"))

    assert response.is_error and not response.is_empty
    assert "no quarters" in response.content


@pytest.mark.asyncio
async def test_query_csv_dataset_without_a_quarter_column_names_its_columns(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "wrong-shape.csv"
    dataset.write_text("period,revenue\n2024-01,10\n", encoding="utf-8")

    response = await QueryCsvTool(dataset).run(call("query_csv"))

    assert response.is_error
    assert "period, revenue" in response.content


@pytest.mark.asyncio
async def test_query_csv_ragged_dataset_reports_the_line(tmp_path: Path) -> None:
    """Regression guard: a short row indexed blind is an IndexError, which §6 forbids."""
    dataset = tmp_path / "ragged.csv"
    dataset.write_text("quarter,revenue,costs,profit\n2024Q1,10\n", encoding="utf-8")

    response = await QueryCsvTool(dataset).run(call("query_csv"))

    assert response.is_error
    assert "line 2" in response.content


@pytest.mark.asyncio
async def test_query_csv_zero_last_n_reports_the_validation_message(
    csv_tool: QueryCsvTool,
) -> None:
    response = await csv_tool.run(call("query_csv", last_n=0))

    assert response.is_error
    assert "last_n" in response.content


@pytest.mark.asyncio
async def test_query_csv_unknown_argument_is_rejected(csv_tool: QueryCsvTool) -> None:
    """`extra="forbid"`: an invented argument must be reported, never silently dropped."""
    response = await csv_tool.run(call("query_csv", where="profit > 0"))

    assert response.is_error
    assert "where" in response.content


@pytest.mark.asyncio
async def test_query_csv_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch, blocked_read: Callable[[object], BlockedRead]
) -> None:
    """§10/§12: `CancelledError` is the only thing that may leave `run`.

    The read's `except` list is what could swallow it. `CancelledError` is a
    `BaseException`, so `except Exception` would still be safe; widening to
    `BaseException`, or to a bare `except` (§8), fails this test.
    """
    read = blocked_read([["quarter"], ["2024Q1"]])
    monkeypatch.setattr(query_csv_module, "_read_rows", read)
    task = asyncio.create_task(QueryCsvTool(FINANCIALS).run(call("query_csv")))

    assert await asyncio.to_thread(read.started.wait, TIMEOUT)  # parked inside the try
    task.cancel()

    # A handler that caught the cancellation would return a `ToolResponse` and fail here
    # rather than hang: the read is released on fixture teardown either way.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TIMEOUT)


# ------------------------------------------------------------------------------ search


def test_search_info_advertises_its_params_schema(search_tool: SearchTool) -> None:
    """As above: the published shape, not the expression `info()` returns."""
    spec = search_tool.info()

    assert spec.name == "search"
    assert set(properties(spec.input_schema)) == {"query", "limit"}
    assert spec.input_schema["additionalProperties"] is False
    # The cap is in the schema so the model reads it before choosing, not in an error.
    assert properties(spec.input_schema)["limit"]["maximum"] == MAX_RESULTS


def test_search_info_description_disclaims_the_internet_and_company_figures(
    search_tool: SearchTool,
) -> None:
    """The one thing this description must not do is imply it can look things up online."""
    description = search_tool.info().description

    assert "NOT reach the internet" in description
    assert "query_csv" in description


@pytest.mark.asyncio
async def test_search_matching_query_returns_the_bundled_note(search_tool: SearchTool) -> None:
    """Exercises the committed corpus through the real scoring path."""
    response = await search_tool.run(call("search", query="typical gross margin for software"))

    assert not response.is_error and not response.is_empty
    assert "margin ranges" in response.content
    assert "75-85%" in response.content


@pytest.mark.asyncio
async def test_search_result_preamble_disclaims_the_corpus_as_unsourced(
    search_tool: SearchTool,
) -> None:
    """The notes state invented specifics as fact, and the model will quote them.

    The preamble is the only thing between "growth clustered between 25% and 40%" and
    that sentence landing in a report whose design forbids unsourced figures.
    """
    response = await search_tool.run(call("search", query="growth benchmarks"))

    preamble = response.content.split("\n\n")[0]
    assert "illustrative sample data" in preamble
    assert "not sourced research" in preamble
    assert "do not quote its numbers as fact" in preamble


@pytest.mark.asyncio
async def test_search_uppercase_keyword_in_the_corpus_still_matches(tmp_path: Path) -> None:
    """`_rank` intersects a lowercased token set, so `"Margin"` would silently never
    match. Normalised on load rather than rejected — a typo must not cost every note."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        '[{"keywords": ["Margin", "GROSS"], "title": "Margins", "snippet": "a"}]',
        encoding="utf-8",
    )

    response = await SearchTool(corpus).run(call("search", query="gross margin"))

    assert not response.is_error and not response.is_empty
    assert "Margins" in response.content


@pytest.mark.asyncio
async def test_search_limit_caps_the_number_of_notes_returned(search_tool: SearchTool) -> None:
    broad = "quarterly cost margin growth seasonality macro benchmarks leverage"

    one = await search_tool.run(call("search", query=broad, limit=1))
    two = await search_tool.run(call("search", query=broad, limit=2))

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

    response = await SearchTool(corpus).run(call("search", query="cost and margin"))

    assert response.content.index("Two hits") < response.content.index("One hit")


@pytest.mark.asyncio
async def test_search_no_match_is_empty_not_an_error(search_tool: SearchTool) -> None:
    """A search that looked and found nothing answered correctly. Flagging it as an error
    reads to the model as "the tool broke" and buys a pointless retry of the same call.

    `is_empty` is the other half: without it a caller asking "did this step retrieve
    anything?" reads a non-error response and records "nothing matched" as provenance.
    """
    response = await search_tool.run(call("search", query="zzzz nonexistent topic"))

    assert not response.is_error
    assert response.is_empty
    assert "Nothing in the offline corpus matched" in response.content
    assert "Seasonality" in response.content  # the retry is told what is there


@pytest.mark.asyncio
async def test_search_malformed_corpus_is_an_error_not_a_crash(tmp_path: Path) -> None:
    """A JSON file on disk is a trust boundary like any other (§7)."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text('{"notes": []}', encoding="utf-8")

    response = await SearchTool(corpus).run(call("search", query="margin"))

    assert response.is_error
    assert str(corpus) in response.content


@pytest.mark.asyncio
async def test_search_corpus_with_a_bad_entry_is_an_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text('[{"title": "No keywords", "snippet": "a"}]', encoding="utf-8")

    response = await SearchTool(corpus).run(call("search", query="margin"))

    assert response.is_error
    assert "keywords" in response.content


@pytest.mark.asyncio
async def test_search_unparseable_corpus_is_an_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[{", encoding="utf-8")

    response = await SearchTool(corpus).run(call("search", query="margin"))

    assert response.is_error
    assert "malformed" in response.content


@pytest.mark.asyncio
async def test_search_missing_corpus_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "never-written.json"

    response = await SearchTool(missing).run(call("search", query="margin"))

    assert response.is_error
    assert str(missing) in response.content


@pytest.mark.asyncio
async def test_search_empty_query_reports_the_validation_message(search_tool: SearchTool) -> None:
    response = await search_tool.run(call("search", query=""))

    assert response.is_error
    assert "query" in response.content


@pytest.mark.asyncio
async def test_search_limit_above_the_cap_is_rejected(search_tool: SearchTool) -> None:
    """The cap is in the schema, so exceeding it comes back naming the cap."""
    response = await search_tool.run(call("search", query="margin", limit=MAX_RESULTS + 1))

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

    first = await tool.run(call("search", query="margin"))
    corpus.unlink()
    second = await tool.run(call("search", query="margin"))

    assert not first.is_error and not second.is_error
    assert first.content == second.content


@pytest.mark.asyncio
async def test_search_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch, blocked_read: Callable[[object], BlockedRead]
) -> None:
    """§10/§12: the corpus load is cancellable and neither handler around it absorbs the
    cancellation. As above, the mutation this catches is a widening to `BaseException`."""
    read = blocked_read([])
    monkeypatch.setattr(search_module, "_read_corpus", read)
    task = asyncio.create_task(SearchTool(SNIPPETS).run(call("search", query="margin")))

    assert await asyncio.to_thread(read.started.wait, TIMEOUT)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TIMEOUT)
