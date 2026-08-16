"""Which tools each agent gets — one place, so the answer is never spread out (§3.3).

An agent's capability is the list it is handed here. Constructing a tool inside a worker
would hide that: two workers would drift into two slightly different `query_csv`s, and
answering "what can the retrieval agent actually do?" would mean reading every worker.

Tool names are re-exported rather than restated: a worker branches on which tool
answered, and a second literal would stop matching the day one is renamed (§1.5).
"""

from pathlib import Path

from pydantic import SecretStr

from orchestra.artifacts import ArtifactStore
from orchestra.tools.base import BaseTool
from orchestra.tools.python_exec import RunPythonTool
from orchestra.tools.query_csv import TOOL_NAME as QUERY_CSV_TOOL
from orchestra.tools.query_csv import QueryCsvTool
from orchestra.tools.search import TOOL_NAME as SEARCH_TOOL
from orchestra.tools.search import SearchTool

__all__ = [
    "FINANCIALS_CSV",
    "QUERY_CSV_TOOL",
    "SEARCH_CORPUS",
    "SEARCH_TOOL",
    "analytics_tools",
    "data_retrieval_tools",
]

# Filenames inside `Config.data_dir`. Here, not in the tools: the tools take a path, so
# they stay testable against a fixture and this stays the only module that knows the layout.
FINANCIALS_CSV = "quarterly_financials.csv"
SEARCH_CORPUS = "search_snippets.json"


def data_retrieval_tools(
    data_dir: Path, *, search_api_key: SecretStr | None = None
) -> tuple[BaseTool, ...]:
    """The Data Retrieval agent's toolset: the company's own figures, plus background.

    Two rather than one because the roles differ — `query_csv` is authoritative and
    narrow, `search` is contextual — and choosing between them is the agent's judgement.

    Args:
        data_dir: the directory holding the bundled dataset, from `Config.data_dir`.
        search_api_key: the live search credential, or `None` for the bundled corpus only.

    Returns:
        The tools, in the order the model is shown them.
    """
    return (
        QueryCsvTool(data_dir / FINANCIALS_CSV),
        SearchTool(data_dir / SEARCH_CORPUS, api_key=search_api_key),
    )


def analytics_tools(store: ArtifactStore) -> tuple[BaseTool, ...]:
    """The Analytics agent's toolset: an interpreter, and nothing that fetches data.

    One tool, deliberately. This agent computes over what a previous step retrieved, and
    handing it `query_csv` as well would let it re-retrieve — bypassing the plan's
    dependency, so the artifact it analyses is no longer the artifact the plan says it
    analysed, and the report's figures trace to a source the planner never ordered.

    Args:
        store: the run's artifact store, which the executor resolves `inputs` against.

    Returns:
        The tools, in the order the model is shown them.
    """
    return (RunPythonTool(store),)
