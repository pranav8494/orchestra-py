"""Tests for the Analytics worker.

Asserts the contract the aggregator and #7 depend on: a pointer out, an `AnalysisResult`
behind it, a step with no computation failing rather than reporting, cancellation
propagated.

Two tests run the real `RunPythonTool` because the acceptance criterion is arithmetic, not
plumbing. The rest use `conftest.FakeTool`. Bounds, the turn cap, warnings and unknown tool
names belong to the shared loop — see `test_tool_loop.py`.
"""

import asyncio
import csv
import json
from collections.abc import Sequence

import pytest

from conftest import FINANCIALS_CSV, FakeProvider, FakeTool, tool_call
from orchestra.agents.toolsets import analytics_tools
from orchestra.agents.workers.analytics import AnalysisResult, AnalyticsWorker
from orchestra.agents.workers.data_retrieval import RetrievedDataset, RetrievedTable
from orchestra.artifacts import DEFAULT_PREVIEW_LIMIT, ArtifactStore
from orchestra.config import default_data_dir
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import (
    AgentRole,
    KeyFigure,
    Subtask,
    SubtaskContext,
    TaskEvent,
    TaskState,
)
from orchestra.prompts import ANALYTICS_SYSTEM_PROMPT
from orchestra.providers.base import AssistantTurn
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse
from orchestra.tools.python_exec import TOOL_NAME as RUN_PYTHON_TOOL

REQUEST = "Summarize the last 3 quarters' financial trends"
UPSTREAM = "fetch_financials"
UPSTREAM_POINTER = f"artifact:{UPSTREAM}.json"

# The shape the prompt teaches: read every entry of `datasets`, load each `csv` with
# pandas, print one labelled number. `shift(1)` rather than `pct_change()`, which warns
# about its fill method.
QOQ_SCRIPT = f"""\
import io
import json

import pandas as pd

data = json.load(open("./{UPSTREAM}.json"))
frames = [pd.read_csv(io.StringIO(table["csv"])) for table in data["datasets"]]
rows = pd.concat(frames, ignore_index=True).sort_values("quarter")
rows["growth"] = (rows["revenue"] / rows["revenue"].shift(1) - 1) * 100
latest = rows.iloc[-1]
print(f"{{latest['quarter']}} revenue growth: {{latest['growth']:.2f}}% QoQ")
"""

# The second script of a two-script step. Its length is not padding: JSON-escaped, one
# script of this shape outweighs everything the aggregator is shown — the point of the
# preview-ordering test below.
LEVELS_SCRIPT = f"""\
import io
import json

import pandas as pd

data = json.load(open("./{UPSTREAM}.json"))
frames = [pd.read_csv(io.StringIO(table["csv"])) for table in data["datasets"]]
rows = pd.concat(frames, ignore_index=True)
rows["quarter"] = rows["quarter"].astype(str).str.strip()
rows = rows.drop_duplicates(subset="quarter").sort_values("quarter")
revenue = pd.to_numeric(rows["revenue"], errors="coerce").dropna()
costs = pd.to_numeric(rows["costs"], errors="coerce").dropna()
margin = (revenue - costs) / revenue * 100
growth = (revenue / revenue.shift(1) - 1) * 100
by_year = revenue.groupby(rows["quarter"].str[:4]).sum()
print(f"mean quarterly revenue: {{revenue.mean():.0f}}")
print(f"mean quarterly margin: {{margin.mean():.1f}}%")
print(f"mean quarterly growth: {{growth.mean():.2f}}%")
print(f"strongest year: {{by_year.idxmax()}} at {{by_year.max():.0f}}")
"""


def _context(*, inputs: dict[str, str] | None = None, **overrides: object) -> SubtaskContext:
    """A worker's slice, built through the ledger like the engine does."""
    fields: dict[str, object] = {
        "id": "analyse_trends",
        "role": AgentRole.ANALYTICS,
        "instruction": "Compute quarter-over-quarter revenue growth",
        "inputs": sorted(inputs or {}),
    }
    state = TaskState(user_request=REQUEST, artifacts=dict(inputs or {}))
    return state.state_slice(Subtask.model_validate(fields | overrides))


def _worker(
    provider: FakeProvider,
    store: ArtifactStore,
    tools: Sequence[BaseTool],
    broker: Broker[TaskEvent] | None = None,
    **bounds: int,
) -> AnalyticsWorker:
    """The worker under test; an unsubscribed `Broker` stands in when a test ignores
    warnings, as an unobserved run would."""
    return AnalyticsWorker(
        provider=provider,
        store=store,
        tools=tuple(tools),
        broker=broker if broker is not None else Broker(),
        **bounds,
    )


def _call(code: str, call_id: str = "call-1", **arguments: object) -> ToolCall:
    """`conftest.tool_call` with this agent's one tool, and `code` — always present — first."""
    return tool_call(RUN_PYTHON_TOOL, call_id, code=code, **arguments)


def _stored(store: ArtifactStore, pointer: str) -> AnalysisResult:
    """Read the artifact back through its own schema — the aggregator's view of it."""
    return AnalysisResult.model_validate(json.loads(store.get_text(pointer)))


def _seed_upstream(store: ArtifactStore, csv_text: str) -> str:
    """Write the artifact #5 would have written, so the executor resolves a real pointer."""
    dataset = RetrievedDataset(
        instruction="Load the quarterly financials",
        summary="Eight quarters of revenue, costs and profit.",
        datasets=[RetrievedTable(query='{"name": "quarterly_financials"}', csv=csv_text)],
    )
    return store.put_text(f"{UPSTREAM}.json", dataset.model_dump_json(indent=2))


@pytest.mark.asyncio
async def test_worker_stores_the_analysis_and_returns_its_pointer(store: ArtifactStore) -> None:
    """One script, one summary, one artifact (#6)."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call("print(1 + 1)", inputs=[UPSTREAM_POINTER]),),
                usage_tokens=120,
            ),
            AssistantTurn(text="Revenue grew 10.6% quarter over quarter.", usage_tokens=80),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="2025Q4 growth: 10.65%")])

    pointer = await _worker(provider, store, [tool]).run(
        _context(inputs={UPSTREAM: UPSTREAM_POINTER})
    )

    assert pointer == "artifact:analyse_trends.json"
    analysis = _stored(store, pointer)
    assert [figure.value for figure in analysis.figures] == ["2025Q4 growth: 10.65%"]
    assert [item.stdout for item in analysis.computations] == ["2025Q4 growth: 10.65%"]
    assert analysis.computations[0].code == "print(1 + 1)"
    assert analysis.summary == "Revenue grew 10.6% quarter over quarter."
    assert analysis.instruction == "Compute quarter-over-quarter revenue growth"


@pytest.mark.asyncio
async def test_worker_sources_each_figure_to_the_upstream_artifact_it_was_given(
    store: ArtifactStore,
) -> None:
    """#9's pairing: the figure cites the upstream pointer the call named, and carries no
    label — the report's wording is the aggregator's to write."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call("print('growth')", inputs=[UPSTREAM_POINTER]),),
                usage_tokens=20,
            ),
            AssistantTurn(text="Revenue grew.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="2025Q4 growth: 10.65%")])

    pointer = await _worker(provider, store, [tool]).run(
        _context(inputs={UPSTREAM: UPSTREAM_POINTER})
    )

    assert _stored(store, pointer).figures == [
        KeyFigure(value="2025Q4 growth: 10.65%", source=UPSTREAM_POINTER)
    ]


@pytest.mark.asyncio
async def test_worker_sources_a_figure_to_the_upstream_artifact_not_a_staged_data_file(
    store: ArtifactStore,
) -> None:
    """Regression: past `INLINE_MAX_BYTES` retrieval hands the file on by pointer, so the
    script stages two inputs — the step's artifact and the raw data file `fetch_data`
    registered. Only the first is a subtask output, so sourcing the number to the data
    file has `TaskState.backed_figures` drop it and the report loses the figure."""
    raw = "artifact:big_dataset.csv"  # a `put_file` registration, no step's output
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call("print('total: 42')", inputs=[raw, UPSTREAM_POINTER]),),
                usage_tokens=20,
            ),
            AssistantTurn(text="Totals computed.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="total: 42")])
    state = TaskState(user_request=REQUEST)
    state.artifacts[UPSTREAM] = UPSTREAM_POINTER

    pointer = await _worker(provider, store, [tool]).run(
        _context(inputs={UPSTREAM: UPSTREAM_POINTER})
    )

    figures = _stored(store, pointer).figures
    assert [figure.source for figure in figures] == [UPSTREAM_POINTER]
    # The rule the report applies, asserted here rather than trusted: a figure sourced to
    # `raw` survives this worker and is silently dropped two steps later.
    assert state.backed_figures(figures) == figures


@pytest.mark.asyncio
async def test_worker_sources_a_figure_to_the_first_upstream_pointer_of_several(
    store: ArtifactStore,
) -> None:
    """A fan-in names two upstream artifacts. The first of them is the citation — chosen
    over the second deliberately, so which one a number cites does not vary with how the
    model happened to order a list."""
    second = "artifact:fetch_benchmarks.json"
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call("print('compared')", inputs=[UPSTREAM_POINTER, second]),),
                usage_tokens=20,
            ),
            AssistantTurn(text="Compared both.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="ours is 4pp ahead")])

    pointer = await _worker(provider, store, [tool]).run(
        _context(inputs={UPSTREAM: UPSTREAM_POINTER, "fetch_benchmarks": second})
    )

    assert [figure.source for figure in _stored(store, pointer).figures] == [UPSTREAM_POINTER]


@pytest.mark.asyncio
async def test_worker_drops_every_figure_when_its_step_declares_no_inputs(
    store: ArtifactStore,
) -> None:
    """A step given nothing can cite nothing, whatever its script staged: the number is
    dropped and the computation kept, so the step completes without inventing provenance."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call("print('42')", inputs=["artifact:big_dataset.csv"]),),
                usage_tokens=20,
            ),
            AssistantTurn(text="Counted.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="42")])

    pointer = await _worker(provider, store, [tool]).run(_context())

    analysis = _stored(store, pointer)
    assert analysis.figures == []
    assert [item.stdout for item in analysis.computations] == ["42"]


def test_analytics_prompt_asks_for_the_given_pointer_among_the_inputs() -> None:
    """The rule the worker implements, stated to the model: *naming* the step's own
    artifact is what a figure can cite. Order is deliberately not claimed to matter —
    `_figure` picks the first upstream pointer wherever it appears, so a prompt promising
    otherwise would be a rule the code does not keep."""
    assert "name the pointer you were given in inputs" in ANALYTICS_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "arguments",
    [{}, {"inputs": []}, {"inputs": ["/tmp/sales.csv"]}, {"inputs": UPSTREAM_POINTER}],
    ids=["absent", "empty", "not-a-pointer", "not-a-list"],
)
@pytest.mark.asyncio
async def test_worker_records_a_computation_without_a_figure_when_no_input_is_named(
    store: ArtifactStore, arguments: dict[str, object]
) -> None:
    """`call.arguments` is model output. A number nobody can trace back is dropped rather
    than sourced to a guess — but the script ran, so the step still completes."""
    # `ToolCall` directly, not `_call`: these argument maps are the malformed ones a helper
    # taking typed keywords cannot express.
    call = ToolCall(
        id="call-1", name=RUN_PYTHON_TOOL, arguments={"code": "print('total: 42')"} | arguments
    )
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(call,), usage_tokens=20),
            AssistantTurn(text="Totals computed.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="total: 42")])

    pointer = await _worker(provider, store, [tool]).run(_context())

    analysis = _stored(store, pointer)
    assert analysis.figures == []
    assert [item.stdout for item in analysis.computations] == ["total: 42"]


@pytest.mark.asyncio
async def test_worker_offers_only_the_executor_and_keeps_the_request_out_of_the_system_prompt(
    store: ArtifactStore,
) -> None:
    """One tool by design: a retrieval tool here would let the agent bypass the plan's
    dependency and analyse data no step was ordered to fetch (§11)."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("print(1)"),), usage_tokens=10),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    _seed_upstream(store, "quarter,revenue\n2025Q4,7015000\n")
    tools = [FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="1")])]

    await _worker(provider, store, tools).run(_context(inputs={UPSTREAM: UPSTREAM_POINTER}))

    assert [tool.info().name for tool in analytics_tools(store)] == [RUN_PYTHON_TOOL]
    assert {spec.name for spec in provider.send_calls[0].tools} == {RUN_PYTHON_TOOL}
    briefing = provider.send_calls[0].messages[0].content
    assert REQUEST not in provider.send_calls[0].system
    assert REQUEST in briefing
    assert UPSTREAM_POINTER in briefing


@pytest.mark.asyncio
async def test_worker_computes_the_correct_quarter_over_quarter_growth(
    store: ArtifactStore,
) -> None:
    """The ticket's spot-check: nothing faked below the provider, and the printed figure is
    checked against the same rows read here with the csv module. 2025Q4 revenue 7015000
    over 2025Q3 6340000 is +10.65%, verifiable by hand."""
    financials = (default_data_dir() / FINANCIALS_CSV).read_text(encoding="utf-8")
    rows = list(csv.DictReader(financials.splitlines()))
    latest, previous = float(rows[-1]["revenue"]), float(rows[-2]["revenue"])
    expected = f"{rows[-1]['quarter']} revenue growth: {(latest / previous - 1) * 100:.2f}% QoQ"
    assert expected == "2025Q4 revenue growth: 10.65% QoQ"  # by hand, not by pandas

    pointer = _seed_upstream(store, financials)
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call(QOQ_SCRIPT, inputs=[pointer]),),
                usage_tokens=300,
            ),
            AssistantTurn(text="Revenue grew 10.65% in the final quarter.", usage_tokens=60),
        ]
    )
    worker = _worker(provider, store, analytics_tools(store))

    stored = await worker.run(_context(inputs={UPSTREAM: pointer}))

    assert _stored(store, stored).computations[0].stdout.strip() == expected


@pytest.mark.asyncio
async def test_worker_finishes_after_the_model_corrects_a_failing_script(
    store: ArtifactStore,
) -> None:
    """§6: a traceback is data the model reads and fixes, not an unwound loop. The real
    executor, because a fake error string would not show that a broken script names its
    own fault."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("print(revenue)"),), usage_tokens=30),
            AssistantTurn(
                text="", tool_calls=(_call("print('total: 42')", "call-2"),), usage_tokens=30
            ),
            AssistantTurn(text="Recovered after a NameError.", usage_tokens=30),
        ]
    )

    pointer = await _worker(provider, store, analytics_tools(store)).run(_context())

    analysis = _stored(store, pointer)
    assert [item.stdout.strip() for item in analysis.computations] == ["total: 42"]
    # The traceback reached the model as a flagged result on the following turn.
    failure = provider.send_calls[1].messages[-1].tool_results[0]
    assert (failure.call_id, failure.is_error) == ("call-1", True)
    assert "NameError" in failure.content


@pytest.mark.asyncio
async def test_worker_that_computed_nothing_fails_the_subtask(store: ArtifactStore) -> None:
    """A summary with no computation behind it is an invented answer."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("print(oops)"),), usage_tokens=20),
            AssistantTurn(text="Revenue grew by about 10%.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="NameError: oops", is_error=True)])

    with pytest.raises(TaskFailure, match="without computing anything"):
        await _worker(provider, store, [tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_puts_every_figure_ahead_of_every_script_in_the_preview(
    store: ArtifactStore,
) -> None:
    """Field order is the aggregator's budget: it sees a preview, not the payload. Any code
    ahead of a figure elides that figure, and the report is then written from a prompt that
    never saw the number it names. Two computations, because one passes by accident."""
    figures = ["mean quarterly revenue: 6112500", "2025Q4 revenue growth: 10.65% QoQ"]
    assert len(json.dumps(LEVELS_SCRIPT)) > DEFAULT_PREVIEW_LIMIT
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call(LEVELS_SCRIPT, inputs=[UPSTREAM_POINTER]),),
                usage_tokens=20,
            ),
            AssistantTurn(
                text="",
                tool_calls=(_call(QOQ_SCRIPT, "call-2", inputs=[UPSTREAM_POINTER]),),
                usage_tokens=20,
            ),
            AssistantTurn(text="Growth accelerated into Q4.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content=figure) for figure in figures])

    pointer = await _worker(provider, store, [tool]).run(
        _context(inputs={UPSTREAM: UPSTREAM_POINTER})
    )

    preview = store.preview(pointer)
    assert "Growth accelerated into Q4." in preview
    for figure in figures:
        assert figure in preview
    assert "elided" in preview  # the scripts fell off the end, not the numbers


@pytest.mark.asyncio
async def test_worker_propagates_cancellation(store: ArtifactStore) -> None:
    """§10: a cancelled run unwinds through the worker, never swallowed."""
    provider = FakeProvider(turns=[AssistantTurn(text="unreached")], blocker=asyncio.Event())
    worker = _worker(provider, store, [FakeTool(RUN_PYTHON_TOOL, [])])

    task = asyncio.create_task(worker.run(_context()))
    await asyncio.sleep(0)  # let it reach the blocked provider call
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_worker_without_tools_is_a_wiring_bug(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        AnalyticsWorker(
            provider=FakeProvider(),
            store=store,
            tools=(),
            broker=Broker(),
        )
