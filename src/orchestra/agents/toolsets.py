"""Which tools each agent gets — one place, so the answer is never spread out (§3.3).

An agent's capability is the list it is handed here. Constructing a tool inside a worker
would hide that: two workers would drift into two slightly different `query_csv`s, and
answering "what can the retrieval agent actually do?" would mean reading every worker.

A worker has to recognise which of its tools answered — the dataset comes back from one
and provenance from another. The names are re-exported from the tools themselves rather
than written out again here: each tool already declares its own, and a second literal
would be the parallel definition §1.5 rules out, silently stopping a worker's branch
from matching the day one is renamed.
"""

from pathlib import Path

from pydantic import SecretStr

from orchestra.tools.base import BaseTool
from orchestra.tools.query_csv import TOOL_NAME as QUERY_CSV_TOOL
from orchestra.tools.query_csv import QueryCsvTool
from orchestra.tools.search import TOOL_NAME as SEARCH_TOOL
from orchestra.tools.search import SearchTool

__all__ = [
    "FINANCIALS_CSV",
    "QUERY_CSV_TOOL",
    "SEARCH_CORPUS",
    "SEARCH_TOOL",
    "data_retrieval_tools",
]

# Filenames inside `Config.data_dir`. Here rather than in the tools because the tools
# take a path: what they read is the caller's choice, which makes them testable against
# a fixture and keeps this module the only thing that knows the bundled layout.
FINANCIALS_CSV = "quarterly_financials.csv"
SEARCH_CORPUS = "search_snippets.json"


def data_retrieval_tools(
    data_dir: Path, *, search_api_key: SecretStr | None = None
) -> tuple[BaseTool, ...]:
    """The Data Retrieval agent's toolset: the company's own figures, plus background.

    Two tools rather than one because the roles genuinely differ — `query_csv` is
    authoritative and narrow, `search` is contextual — and choosing between them is the
    judgement the agent exists to make.

    Args:
        data_dir: the directory holding the bundled dataset, from `Config.data_dir`.
        search_api_key: the live search credential, or `None` to search the bundled
            corpus only. Optional here rather than required-and-nullable so a caller
            that does not care about search — every test that is not about it — does not
            have to say so.

    Returns:
        The tools, in the order the model is shown them.
    """
    return (
        QueryCsvTool(data_dir / FINANCIALS_CSV),
        SearchTool(data_dir / SEARCH_CORPUS, api_key=search_api_key),
    )
