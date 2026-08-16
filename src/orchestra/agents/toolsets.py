"""Which tools each agent gets — one place, so the answer is never spread out (§3.3).

Constructing a tool inside a worker would hide that: two workers would drift into two
slightly different `query_csv`s.

Tool names are re-exported rather than restated: a worker branches on which tool
answered, and a second literal would stop matching the day one is renamed (§1.5).
"""

from collections.abc import Sequence
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
    "retrievable_data",
]

# Filenames inside `Config.data_dir`. Here, not in the tools: the tools take a path, so
# they stay testable against a fixture and this stays the only module knowing the layout.
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


def retrievable_data(tools: Sequence[BaseTool]) -> str:
    """What `tools` put within reach, one bullet each, for the planner's prompt (#10).

    Here because this module already answers "which tools does an agent get"; the answer
    to "so what data can the team obtain" is the same question read from the other end.
    Composed from `ToolSpec.provides` rather than restated, or the day a dataset changes
    the planner would go on offering the old one.

    Tools that supply no data are skipped, so an interpreter or a prompt does not read as
    a source.
    """
    return "\n".join(f"- {provides}" for tool in tools if (provides := tool.info().provides))


def analytics_tools(store: ArtifactStore) -> tuple[BaseTool, ...]:
    """The Analytics agent's toolset: an interpreter, and nothing that fetches data.

    One tool, deliberately. Handing it `query_csv` too would let it re-retrieve, bypassing
    the plan's dependency — the report's figures would then trace to a source the planner
    never ordered.

    Args:
        store: the run's artifact store, which the executor resolves `inputs` against.
    """
    return (RunPythonTool(store),)
