"""Tests for the one place that decides an agent's capability (CONVENTIONS.md §3.3).

The constants `toolsets.py` re-exports are the names the Data Retrieval worker branches
on when it decides which tool answered. They come from the tools' own modules, but
nothing checks that a tool's `info()` actually advertises the name it declares — so that
is checked here, in the order the worker will see them.
"""

from pathlib import Path

from orchestra.agents.toolsets import (
    FINANCIALS_CSV,
    QUERY_CSV_TOOL,
    SEARCH_CORPUS,
    SEARCH_TOOL,
    data_retrieval_tools,
)
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


def test_data_retrieval_tools_read_the_committed_dataset() -> None:
    """The bundled files are the offline guarantee — a missing one is a broken demo."""
    data_dir = default_data_dir()

    assert (data_dir / FINANCIALS_CSV).is_file()
    assert (data_dir / SEARCH_CORPUS).is_file()


def test_data_retrieval_tools_take_the_directory_they_are_given(tmp_path: Path) -> None:
    """Injected, not read from config or the environment inside the tools (§6, §9)."""
    tools = data_retrieval_tools(tmp_path)

    assert len(tools) == 2  # constructing against an empty directory must not raise
