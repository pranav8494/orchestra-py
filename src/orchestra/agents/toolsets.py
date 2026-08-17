"""Which tools each agent gets — one place, so the answer is never spread out (§3.3).

Constructing a tool inside a worker would hide that: two workers would drift into two
slightly different `fetch_data`s.

Tool names are re-exported rather than restated: a worker branches on which tool
answered, and a second literal would stop matching the day one is renamed (§1.5).
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import SecretStr

from orchestra.artifacts import ArtifactStore
from orchestra.tools.base import BaseTool
from orchestra.tools.fetch_data import TOOL_NAME as FETCH_DATA_TOOL
from orchestra.tools.fetch_data import FetchDataTool, discover_datasets
from orchestra.tools.python_exec import RunPythonTool
from orchestra.tools.search import TOOL_NAME as SEARCH_TOOL
from orchestra.tools.search import SearchTool

__all__ = [
    "FETCH_DATA_TOOL",
    "SEARCH_CORPUS",
    "SEARCH_TOOL",
    "analytics_tools",
    "data_retrieval_tools",
    "retrievable_data",
]

# The one filename inside `Config.data_dir` that code still has to know: `search` is given
# its corpus, where `fetch_data` discovers whatever else is there. Here, not in the tool,
# so the tool stays testable against a fixture.
SEARCH_CORPUS = "search_snippets.json"


def data_retrieval_tools(
    data_dir: Path, store: ArtifactStore, *, search_api_key: SecretStr | None = None
) -> tuple[BaseTool, ...]:
    """The Data Retrieval agent's toolset: the team's own files, plus background.

    Two rather than one because the roles differ — `fetch_data` is authoritative and
    narrow, `search` is contextual — and choosing between them is the agent's judgement.

    The corpus is kept out of the catalogue: it is the `search` tool's, and listing one
    file under two tools would tell the planner there are two sources where there is one.

    Args:
        data_dir: the directory holding the bundled data, from `Config.data_dir`.
        store: the run's artifact store, where a fetched file is registered.
        search_api_key: the live search credential, or `None` for the bundled corpus only.

    Returns:
        The tools, in the order the model is shown them.
    """
    datasets = tuple(
        dataset for dataset in discover_datasets(data_dir) if dataset.path.name != SEARCH_CORPUS
    )
    return (
        FetchDataTool(store, datasets),
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

    One tool, deliberately. Handing it `fetch_data` too would let it re-retrieve, bypassing
    the plan's dependency — the report's figures would then trace to a source the planner
    never ordered.

    Args:
        store: the run's artifact store, which the executor resolves `inputs` against.
    """
    return (RunPythonTool(store),)
