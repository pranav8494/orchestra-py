"""Tests for the one place that decides an agent's capability (§3.3).

The re-exported constants are the names the Data Retrieval worker branches on. Nothing
else checks that a tool's `info()` advertises the name it declares, so that is checked
here, in the order the worker sees them.
"""

from pathlib import Path

from orchestra.agents.toolsets import (
    FINANCIALS_CSV,
    QUERY_CSV_TOOL,
    SEARCH_CORPUS,
    SEARCH_TOOL,
    analytics_tools,
    data_retrieval_tools,
    retrievable_data,
)
from orchestra.artifacts import ArtifactStore
from orchestra.config import default_data_dir


def test_data_retrieval_tools_advertise_the_names_the_worker_branches_on() -> None:
    tools = data_retrieval_tools(default_data_dir())

    assert [tool.info().name for tool in tools] == [QUERY_CSV_TOOL, SEARCH_TOOL]


def test_data_retrieval_tools_describe_themselves_for_the_model() -> None:
    """§6: `info()`'s description is a prompt. An empty one leaves the model guessing."""
    for tool in data_retrieval_tools(default_data_dir()):
        spec = tool.info()
        assert spec.description.strip()
        assert spec.input_schema.get("type") == "object"


def test_retrievable_data_lists_what_each_tool_supplies() -> None:
    """The planner plans against this: one bullet per source, naming what is on file and
    what is not, so it stops offering data nothing can fetch (#10)."""
    roster = retrievable_data(data_retrieval_tools(default_data_dir()))

    assert roster.count("- ") == 2
    # The bundled dataset's own subject and range, from the tool that holds it.
    assert "revenue" in roster and "2025Q4" in roster
    # And the boundary that made a stock-price run plan three steps and produce nothing.
    assert "no share price" in roster


def test_retrievable_data_skips_a_tool_that_supplies_none(tmp_path: Path) -> None:
    """An interpreter is not a source. Listing it would read to the planner as one."""
    assert retrievable_data(analytics_tools(ArtifactStore(tmp_path))) == ""


def test_data_retrieval_tools_read_the_committed_dataset() -> None:
    """The bundled files are the offline guarantee — a missing one is a broken demo."""
    data_dir = default_data_dir()

    assert (data_dir / FINANCIALS_CSV).is_file()
    assert (data_dir / SEARCH_CORPUS).is_file()


def test_data_retrieval_tools_take_the_directory_they_are_given(tmp_path: Path) -> None:
    """Injected, not read from config or the environment inside the tools (§6, §9)."""
    tools = data_retrieval_tools(tmp_path)

    assert len(tools) == 2  # constructing against an empty directory must not raise
