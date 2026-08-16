"""The Data Retrieval agent's primary tool: filtered reads of one financials CSV.

- **stdlib `csv`, not pandas** — select-and-filter over eight rows. pandas is the Analytics
  agent's dependency (#6), costs a second of import on first use, and importing it here
  would make this the second place deciding what a DataFrame is. Rows stay `list[str]`.
- **The dataset is injected** — `agents/toolsets.py` passes the path (§3.3); nothing here
  reads `os.environ` or `Config` (§9), which also keeps the tool pointable at a fixture.
- **Every failure is content, not an exception** (§6) — each message names what would work
  instead. Only `CancelledError` leaves `run`.
"""

import asyncio
import csv
import io
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.tools.base import (
    ToolCall,
    ToolResponse,
    ToolSpec,
    format_validation_error,
)

TOOL_NAME = "query_csv"

# The column the `quarters` filter reads. A dataset without it is a wiring mistake the
# model is told about, rather than an IndexError mid-filter.
QUARTER_COLUMN = "quarter"

# A prompt (§6), beside the params model rather than in `prompts/` (§11 — one module per
# *agent*) because it documents these fields; split them and the schema and the
# description drift apart.
#
# The quarter range is a literal because `info()` must stay pure and cheap. If the
# injected dataset disagrees, `run`'s empty-result and unknown-quarter messages list the
# quarters actually on file.
DESCRIPTION = (
    "Read this company's own quarterly financials from a bundled CSV. This is the only "
    "source for our revenue, costs and profit — use it for any question about them and "
    "quote the figures rather than estimating. Columns: quarter, revenue, costs, profit, "
    "in whole currency units; the bundled dataset covers 2024Q1 through 2025Q4, oldest "
    "row first. Narrow the result with `columns`, `quarters` and `last_n`; omit them all "
    "to get the whole table. Returns CSV text. Do not use it for industry benchmarks, "
    "definitions or macro background — that is the `search` tool."
)


# What the planner is told this puts within reach (`ToolSpec.provides`). Coarser than
# `DESCRIPTION` on purpose: a planner chooses between subjects, and columns and filters are
# the worker's problem. The quarter range is a literal here for the same reason it is there.
PROVIDES = (
    "this company's own quarterly revenue, costs and profit, 2024Q1 through 2025Q4 — "
    "no share price, headcount, or any other measure"
)


class QueryCsvParams(BaseModel):
    """The arguments the model may send, and the check applied to what it sent.

    Published verbatim as `input_schema`, so every `description` here is prompt text —
    hence the examples. `extra` is forbidden so a hallucinated `filter` or `where` comes
    back as a readable validation error instead of being silently dropped (§7).
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

    Stateless past its path: re-read per call, because the dataset is eight rows and a
    cache would be a second source of truth for a file the user can edit between runs.
    """

    def __init__(self, dataset: Path) -> None:
        """Store the injected dataset path; nothing is read yet (§3.3).

        The path is not validated here — an unreadable one is a `ToolResponse` the model
        can route around, not a startup crash.
        """
        self._dataset = dataset

    def info(self) -> ToolSpec:
        """See `BaseTool.info`. Pure: no disk is touched, it runs on every turn."""
        return ToolSpec(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=QueryCsvParams.model_json_schema(),
            provides=PROVIDES,
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        """Filter the dataset and return the matching rows as CSV text. See `BaseTool.run`."""
        try:
            params = QueryCsvParams.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {format_validation_error(exc)}",
                is_error=True,
            )

        try:
            # The engine dispatches subtasks concurrently, so a blocking read here would
            # stall every other agent (§10).
            rows = await asyncio.to_thread(_read_rows, self._dataset)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            return ToolResponse(
                content=f"Could not read the dataset at {self._dataset}: {exc}", is_error=True
            )

        return _select(rows, params, self._dataset)


def _select(rows: list[list[str]], params: QueryCsvParams, dataset: Path) -> ToolResponse:
    """Apply the filters to a parsed CSV. Pure, so every branch is testable as data.

    `rows` is the file, header first. Returns the projected rows as CSV text, or the
    failure as content (§6) — `dataset` is named in shape errors so the operator knows
    which file to fix.
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
        # Checked before indexing: a short row would otherwise be an IndexError, which §6
        # forbids leaving a tool.
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
    # teaches it nothing. Error messages still quote the file's own spelling.
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
        # The file is chronological, oldest first, so recent rows are the tail. Not sorted
        # here: re-ordering the operator's file would hide a corrupt one.
        selected = selected[-params.last_n :]

    if not selected:
        # An error, unlike `search`'s `is_empty`: this dataset is fixed and named, so no
        # match means the model asked for something it does not hold.
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

    Column order follows the request, not the file — the model reads the answer in the
    shape it asked for. `csv.writer`, not `join`, so a value containing a comma is quoted
    rather than producing a file the model cannot parse back.
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
