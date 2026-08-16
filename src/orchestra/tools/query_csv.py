"""The Data Retrieval agent's primary tool: filtered reads of one financials CSV.

**Why stdlib `csv` and not pandas.** The work here is select-and-filter over eight rows.
pandas is the Analytics agent's dependency (#6) and it costs a second of import time on
first use — paying that inside a data-retrieval tool buys nothing, and importing it here
would also make this module the second place in the codebase that decides what a
DataFrame is. Rows stay `list[str]`; whoever wants arithmetic gets the CSV text.

**Why the dataset is injected.** `agents/toolsets.py` constructs the tool with a path
(§3.3); nothing here reads `os.environ` or `Config` (§9). A tool that resolves its own
data cannot be pointed at a fixture, and the layer rule (§3.2) forbids `tools/` importing
`config.py` in any case.

**Every failure is content, not an exception** (§6). The model's next turn is the retry,
so each message names what would work: the columns that do exist, the quarters that do
exist, or the path that could not be read. The only thing that leaves `run` is
`CancelledError`.
"""

import asyncio
import csv
import io
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.tools.base import ToolCall, ToolResponse, ToolSpec

TOOL_NAME = "query_csv"

# The column the `quarters` filter reads. The tool advertises quarter filtering and its
# error messages quote the quarters on file, so a dataset without this column is a
# wiring mistake the model is told about rather than an IndexError mid-filter.
QUARTER_COLUMN = "quarter"

# The description is a prompt (§6), not a docstring: when to reach for the tool, when
# not, and what it will refuse. It lives beside the params model rather than in
# `prompts/` (§11 — one module per *agent*) because it documents these fields; splitting
# them is how a schema and a description come to say different things.
#
# The quarter range is stated as a literal because `info()` must stay pure and cheap —
# reading the file on every turn to interpolate two strings is the wrong trade. If the
# injected dataset ever disagrees, `run` corrects the model in one turn: its empty-result
# and unknown-quarter messages list the quarters actually on file.
DESCRIPTION = (
    "Read this company's own quarterly financials from a bundled CSV. This is the only "
    "source for our revenue, costs and profit — use it for any question about them and "
    "quote the figures rather than estimating. Columns: quarter, revenue, costs, profit, "
    "in whole currency units; the bundled dataset covers 2024Q1 through 2025Q4, oldest "
    "row first. Narrow the result with `columns`, `quarters` and `last_n`; omit them all "
    "to get the whole table. Returns CSV text. Do not use it for industry benchmarks, "
    "definitions or macro background — that is the `search` tool."
)


class QueryCsvParams(BaseModel):
    """The arguments the model may send, and the check applied to what it sent.

    Published verbatim as the tool's `input_schema`, so every `description` here is
    prompt text the model reads while choosing arguments — hence the examples. `extra`
    is forbidden so a hallucinated `filter` or `where` argument comes back as a readable
    validation error instead of being silently dropped (§7).
    """

    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(
        default_factory=list,
        description="Columns to return, e.g. ['quarter', 'profit']. Empty means every column.",
    )
    quarters: list[str] = Field(
        default_factory=list,
        description="Quarters to return, e.g. ['2025Q1', '2025Q2']. Empty means every row.",
    )
    last_n: int | None = Field(
        default=None,
        gt=0,
        description="After the other filters, keep only the most recent N rows. Omit to keep all.",
    )


class QueryCsvTool:
    """Filtered reads of one CSV of quarterly financials. Implements `BaseTool` (§6).

    Stateless past its path: the file is re-read per call rather than cached, because the
    dataset is eight rows and a cache would be a second source of truth for a file the
    user can edit between runs.
    """

    def __init__(self, dataset: Path) -> None:
        """Store the injected dataset path. Nothing is read yet (§3.3).

        Args:
            dataset: the CSV to query, from `app.py` via `agents/toolsets.py`. Not
                validated here — an unreadable path is a `ToolResponse`, not a startup
                crash, because the model can be told about it and route around it.
        """
        self._dataset = dataset

    def info(self) -> ToolSpec:
        """See `BaseTool.info`. Pure: no disk is touched, it runs on every turn."""
        return ToolSpec(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=QueryCsvParams.model_json_schema(),
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        """Filter the dataset and return the matching rows as CSV text. See `BaseTool.run`."""
        try:
            params = QueryCsvParams.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {_problems(exc)}", is_error=True
            )

        try:
            # `to_thread` because the read blocks, and the engine dispatches subtasks
            # concurrently — a synchronous read here stalls every other agent (§10).
            rows = await asyncio.to_thread(_read_rows, self._dataset)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            return ToolResponse(
                content=f"Could not read the dataset at {self._dataset}: {exc}", is_error=True
            )

        return _select(rows, params, self._dataset)


def _select(rows: list[list[str]], params: QueryCsvParams, dataset: Path) -> ToolResponse:
    """Apply the filters to a parsed CSV. Pure, so every branch is testable as data.

    Args:
        rows: the file, header first, as returned by `_read_rows`.
        params: the validated arguments.
        dataset: named in shape errors, so the operator knows which file to fix.

    Returns:
        The projected rows as CSV text, or the failure as content (§6).
    """
    if not rows:
        return ToolResponse(content=f"The dataset at {dataset} is empty.", is_error=True)

    header, data = rows[0], rows[1:]
    columns = ", ".join(header)

    if QUARTER_COLUMN not in header:
        return ToolResponse(
            content=f"The dataset at {dataset} has no {QUARTER_COLUMN!r} column, so it cannot be "
            f"queried by quarter. Its columns are {columns}.",
            is_error=True,
        )

    ragged = next((line for line, row in enumerate(data, start=2) if len(row) != len(header)), None)
    if ragged is not None:
        # Checked before indexing: a short row would otherwise be an IndexError, which
        # §6 forbids leaving a tool, and the operator needs the line number anyway.
        return ToolResponse(
            content=f"The dataset at {dataset} is malformed: line {ragged} does not have "
            f"{len(header)} fields.",
            is_error=True,
        )

    unknown_columns = [name for name in params.columns if name not in header]
    if unknown_columns:
        return ToolResponse(
            content=f"No column named {', '.join(unknown_columns)}. The columns are {columns}.",
            is_error=True,
        )

    quarter_at = header.index(QUARTER_COLUMN)
    available = [row[quarter_at] for row in data]
    # Case-folded: a model writing `2025q2` meant `2025Q2`, and spending a turn on that
    # teaches it nothing. The error messages still quote the file's own spelling.
    on_file = {quarter.upper() for quarter in available}
    wanted = {quarter.strip().upper() for quarter in params.quarters}

    unknown_quarters = sorted(wanted - on_file)
    if unknown_quarters:
        return ToolResponse(
            content=f"No rows for {', '.join(unknown_quarters)}. The dataset covers "
            f"{_quarters(available)}.",
            is_error=True,
        )

    selected = [row for row in data if not wanted or row[quarter_at].upper() in wanted]
    if params.last_n is not None:
        # The file is chronological, oldest first, so the most recent rows are the tail.
        # Not sorted here: re-ordering the operator's file would hide a corrupt one.
        selected = selected[-params.last_n :]

    if not selected:
        return ToolResponse(
            content=f"No rows matched. The dataset covers {_quarters(available)}.", is_error=True
        )

    return ToolResponse(content=_to_csv(params.columns or header, header, selected))


def _read_rows(path: Path) -> list[list[str]]:
    """Read the whole CSV, skipping blank lines. Blocking — call it in a thread.

    `csv.reader` rather than `DictReader`: rows stay `list[str]`, where `DictReader`
    yields `str | Any` values and `Any` is banned in `tools/` (§7).
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle) if row]


def _to_csv(columns: list[str], header: list[str], rows: list[list[str]]) -> str:
    """Project `rows` onto `columns` and render header-first CSV text.

    Column order follows the request, not the file: the model asked for
    `['profit', 'quarter']` and reads the answer more easily in the shape it asked for.
    Written through `csv.writer` so a value containing a comma is quoted, rather than
    through `join`, which would silently produce a file the model cannot parse back.
    """
    indexes = [header.index(name) for name in columns]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([row[index] for index in indexes] for row in rows)
    return buffer.getvalue()


def _quarters(available: list[str]) -> str:
    """Name the quarters on file for an error message, or say there are none."""
    return ", ".join(available) if available else "no quarters (it has no data rows)"


def _problems(exc: ValidationError) -> str:
    """Flatten a pydantic error into one line the model can act on.

    Deliberately duplicated in `search.py` (§2.3): two tools, five lines, and no third
    caller yet to show whether the axis of variation is the wording or the shape. It
    moves into `tools/base.py` for every implementer at the third (§2.2).
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors()
    )
