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
from orchestra.tools import query_csv as query_csv_module
from orchestra.tools import search as search_module
from orchestra.tools.base import BaseTool
from orchestra.tools.query_csv import QueryCsvTool
from orchestra.tools.search import MAX_RESULTS, SearchTool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FINANCIALS = DATA_DIR / "quarterly_financials.csv"
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

    def __call__(self, path: Path) -> object:
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
def csv_tool() -> QueryCsvTool:
    return QueryCsvTool(FINANCIALS)


@pytest.fixture
def search_tool() -> SearchTool:
    return SearchTool(SNIPPETS)


if TYPE_CHECKING:
    # Conformance is mypy's job, not `isinstance`'s: `BaseTool` is a plain Protocol, and a
    # runtime check would compare attribute names only (§7).
    _CSV_IS_A_TOOL: BaseTool = QueryCsvTool(FINANCIALS)
    _SEARCH_IS_A_TOOL: BaseTool = SearchTool(SNIPPETS)


# --------------------------------------------------------------------------- query_csv


def test_query_csv_info_advertises_its_params_schema(csv_tool: QueryCsvTool) -> None:
    """Asserted field by field rather than against `QueryCsvParams.model_json_schema()`,
    which is the expression `info()` returns — that comparison passes whatever the schema
    says, including nothing."""
    spec = csv_tool.info()

    assert spec.name == "query_csv"
    assert set(properties(spec.input_schema)) == {"columns", "quarters", "last_n"}
    # `extra="forbid"` reaches the model, so the provider rejects an invented `where`
    # client-side as well as `run` does.
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
    """Exercises the committed dataset: eight quarters, and profit that adds up."""
    response = await csv_tool.run(tool_call("query_csv"))

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
        tool_call("query_csv", columns=["quarter", "profit"], quarters=["2025Q1"])
    )

    assert not response.is_error
    assert rows(response.content) == [["quarter", "profit"], ["2025Q1", "915000"]]


@pytest.mark.asyncio
async def test_query_csv_columns_are_returned_in_the_requested_order(
    csv_tool: QueryCsvTool,
) -> None:
    """The model reads its answer more easily in the shape it asked for."""
    response = await csv_tool.run(
        tool_call("query_csv", columns=["profit", "quarter"], quarters=["2024Q1"])
    )

    assert rows(response.content) == [["profit", "quarter"], ["630000", "2024Q1"]]


@pytest.mark.asyncio
async def test_query_csv_last_n_keeps_the_most_recent_rows(csv_tool: QueryCsvTool) -> None:
    """The file is oldest-first, so "most recent" is the tail, not the head."""
    response = await csv_tool.run(tool_call("query_csv", columns=["quarter"], last_n=2))

    assert rows(response.content) == [["quarter"], ["2025Q3"], ["2025Q4"]]


@pytest.mark.asyncio
async def test_query_csv_last_n_larger_than_the_dataset_returns_every_row(
    csv_tool: QueryCsvTool,
) -> None:
    """Asking for more than exists is not an error — it is the whole table."""
    response = await csv_tool.run(tool_call("query_csv", columns=["quarter"], last_n=99))

    assert not response.is_error
    assert len(rows(response.content)) == 9  # header + 8 quarters


@pytest.mark.asyncio
async def test_query_csv_lowercase_quarter_still_matches(csv_tool: QueryCsvTool) -> None:
    """`2025q4` meant `2025Q4`; a turn spent correcting that teaches the model nothing."""
    response = await csv_tool.run(tool_call("query_csv", columns=["quarter"], quarters=["2025q4"]))

    assert rows(response.content) == [["quarter"], ["2025Q4"]]


@pytest.mark.asyncio
async def test_query_csv_unknown_column_names_the_valid_columns(csv_tool: QueryCsvTool) -> None:
    """The error is the model's next prompt, so it has to contain the retry (§6)."""
    response = await csv_tool.run(tool_call("query_csv", columns=["margin"]))

    assert response.is_error
    assert "margin" in response.content
    assert "quarter, revenue, costs, profit" in response.content


@pytest.mark.asyncio
async def test_query_csv_unknown_quarter_names_the_available_quarters(
    csv_tool: QueryCsvTool,
) -> None:
    response = await csv_tool.run(tool_call("query_csv", quarters=["2023Q4"]))

    assert response.is_error
    assert "2023Q4" in response.content
    assert "2024Q1" in response.content and "2025Q4" in response.content


@pytest.mark.asyncio
async def test_query_csv_missing_dataset_file_names_the_path(tmp_path: Path) -> None:
    """A bad injection surfaces as content, not as an unwound agent loop."""
    missing = tmp_path / "never-written.csv"

    response = await QueryCsvTool(missing).run(tool_call("query_csv"))

    assert response.is_error
    assert str(missing) in response.content


@pytest.mark.asyncio
async def test_query_csv_header_only_dataset_reports_an_empty_result(tmp_path: Path) -> None:
    """An error here and `is_empty` in `search`: the asymmetry is deliberate. This dataset
    is fixed, so no rows means the model asked for something it does not hold, and a retry
    against the columns the message names is the right next move."""
    dataset = tmp_path / "empty.csv"
    dataset.write_text("quarter,revenue,costs,profit\n", encoding="utf-8")

    response = await QueryCsvTool(dataset).run(tool_call("query_csv"))

    assert response.is_error and not response.is_empty
    assert "no quarters" in response.content


@pytest.mark.asyncio
async def test_query_csv_dataset_without_a_quarter_column_names_its_columns(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "wrong-shape.csv"
    dataset.write_text("period,revenue\n2024-01,10\n", encoding="utf-8")

    response = await QueryCsvTool(dataset).run(tool_call("query_csv"))

    assert response.is_error
    assert "period, revenue" in response.content


@pytest.mark.asyncio
async def test_query_csv_ragged_dataset_reports_the_line(tmp_path: Path) -> None:
    """Regression: a short row indexed blind is an IndexError, which §6 forbids."""
    dataset = tmp_path / "ragged.csv"
    dataset.write_text("quarter,revenue,costs,profit\n2024Q1,10\n", encoding="utf-8")

    response = await QueryCsvTool(dataset).run(tool_call("query_csv"))

    assert response.is_error
    assert "line 2" in response.content


@pytest.mark.asyncio
async def test_query_csv_zero_last_n_reports_the_validation_message(
    csv_tool: QueryCsvTool,
) -> None:
    response = await csv_tool.run(tool_call("query_csv", last_n=0))

    assert response.is_error
    assert "last_n" in response.content


@pytest.mark.asyncio
async def test_query_csv_unknown_argument_is_rejected(csv_tool: QueryCsvTool) -> None:
    """`extra="forbid"`: an invented argument is reported, never silently dropped."""
    response = await csv_tool.run(tool_call("query_csv", where="profit > 0"))

    assert response.is_error
    assert "where" in response.content


@pytest.mark.asyncio
async def test_query_csv_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch, blocked_read: Callable[[object], BlockedRead]
) -> None:
    """§10: `CancelledError` is the only thing that may leave `run`. The read's `except`
    list is what could swallow it — widening to `BaseException`, or to a bare `except`
    (§8), fails this test."""
    read = blocked_read([["quarter"], ["2024Q1"]])
    monkeypatch.setattr(query_csv_module, "_read_rows", read)
    task = asyncio.create_task(QueryCsvTool(FINANCIALS).run(tool_call("query_csv")))

    assert await asyncio.to_thread(read.started.wait, TIMEOUT)  # parked inside the try
    task.cancel()

    # A handler that caught the cancellation returns a `ToolResponse` and fails here
    # rather than hanging: the read is released on fixture teardown either way.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TIMEOUT)


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

    assert "query_csv" in description
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
