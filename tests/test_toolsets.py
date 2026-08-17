"""Tests for the one place that decides an agent's capability (§3.3).

The re-exported constants are the names the Data Retrieval worker branches on. Nothing
else checks that a tool's `info()` advertises the name it declares, so that is checked
here, in the order the worker sees them.
"""

from pathlib import Path

from conftest import FINANCIALS_CSV
from orchestra.agents.toolsets import (
    FETCH_DATA_TOOL,
    SEARCH_CORPUS,
    SEARCH_TOOL,
    analytics_tools,
    data_retrieval_tools,
    retrievable_data,
)
from orchestra.artifacts import ArtifactStore
from orchestra.config import default_data_dir

# The committed catalogue: every file `data/` ships bar the corpus, each with the shape
# the startup probe reports for it. Both the roster test and the "the demo still has its
# data" test read this, so a file added to `data/` is added here once.
BUNDLED_DATASETS = {
    "Q4 Board Pack (final).csv": "CSV with columns metric, value, unit, notes & source",
    "expense_breakdown.csv": "CSV with columns quarter, category, amount",
    "product_lines.json": "an object with keys quarter, product_line, revenue",
    "project_timeline.md": "beginning '# Project timeline 2024-2025'",
    "quarterly_financials.csv": "CSV with columns quarter, revenue, costs, profit",
    "yearly_performance.csv": (
        "CSV with columns year, revenue, costs, profit, headcount_year_end, "
        "customers_year_end, net_revenue_retention_pct"
    ),
}


def test_data_retrieval_tools_advertise_the_names_the_worker_branches_on(
    store: ArtifactStore,
) -> None:
    tools = data_retrieval_tools(default_data_dir(), store)

    assert [tool.info().name for tool in tools] == [FETCH_DATA_TOOL, SEARCH_TOOL]


def test_data_retrieval_tools_describe_themselves_for_the_model(store: ArtifactStore) -> None:
    """§6: `info()`'s description is a prompt. An empty one leaves the model guessing."""
    for tool in data_retrieval_tools(default_data_dir(), store):
        spec = tool.info()
        assert spec.description.strip()
        assert spec.input_schema.get("type") == "object"


def test_retrievable_data_lists_what_each_tool_supplies(store: ArtifactStore) -> None:
    """The planner plans against this: one bullet per source, naming what is on file and
    what is not, so it stops offering data nothing can fetch (#10)."""
    roster = retrievable_data(data_retrieval_tools(default_data_dir(), store))

    assert roster.count("- ") == 2
    # Every bundled dataset, named and shaped, from the tool that holds them: the planner
    # can only plan a step against a file it was told the columns of.
    for filename, shape in BUNDLED_DATASETS.items():
        assert Path(filename).stem in roster
        assert shape in roster
    # And the boundary that made a stock-price run plan three steps and produce nothing.
    assert "nothing beyond those files" in roster


def test_retrievable_data_skips_a_tool_that_supplies_none(tmp_path: Path) -> None:
    """An interpreter is not a source. Listing it would read to the planner as one."""
    assert retrievable_data(analytics_tools(ArtifactStore(tmp_path))) == ""


def test_data_retrieval_tools_read_the_committed_datasets() -> None:
    """The bundled files are the offline guarantee — a missing one is a broken demo."""
    data_dir = default_data_dir()

    assert FINANCIALS_CSV in BUNDLED_DATASETS  # the one the fixtures fetch by name
    for filename in BUNDLED_DATASETS:
        assert (data_dir / filename).is_file()
    assert (data_dir / SEARCH_CORPUS).is_file()


def test_data_retrieval_tools_keep_the_search_corpus_out_of_the_catalogue(
    store: ArtifactStore,
) -> None:
    """One file under two tools would tell the planner there are two sources where there
    is one, and would let retrieval hand over the corpus without `search`'s disclaimer."""
    provides = data_retrieval_tools(default_data_dir(), store)[0].info().provides

    assert Path(SEARCH_CORPUS).stem not in provides


def test_data_retrieval_tools_take_the_directory_they_are_given(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """Injected, not read from config or the environment inside the tools (§6, §9)."""
    tools = data_retrieval_tools(tmp_path, store)

    assert len(tools) == 2  # constructing against an empty directory must not raise
    assert tools[0].info().provides == ""  # and the planner is told there is no data
