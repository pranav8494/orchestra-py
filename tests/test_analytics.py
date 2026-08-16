"""Tests for the Analytics worker (CONVENTIONS.md §12).

What is asserted is the contract the aggregator and #7 depend on: a pointer out, an
`AnalysisResult` behind it, a step with no computation behind it failing rather than
reporting, and cancellation propagated.

Two tests run the real `RunPythonTool` against a real store, because the acceptance
criterion is arithmetic and not plumbing: one seeds the committed dataset as a
`RetrievedDataset` and checks the quarter-over-quarter figure the script prints against
the same rows, the other lets a script raise and watches the agent correct itself. The
rest use `conftest.FakeTool` — a test that also started an interpreter could not say
which half broke. Bounds, the turn cap, warnings and unknown tool names belong to the
shared loop and are covered in `test_tool_loop.py`.
"""

import asyncio
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from conftest import FakeProvider, FakeTool
from orchestra.agents.toolsets import FINANCIALS_CSV, analytics_tools
from orchestra.agents.workers.analytics import AnalysisResult, AnalyticsWorker
from orchestra.agents.workers.data_retrieval import RetrievedDataset, RetrievedTable
from orchestra.artifacts import DEFAULT_PREVIEW_LIMIT, ArtifactStore
from orchestra.config import default_data_dir
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, Subtask, SubtaskContext, TaskEvent, TaskState
from orchestra.providers.base import AssistantTurn
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse
from orchestra.tools.python_exec import TOOL_NAME as RUN_PYTHON_TOOL

REQUEST = "Summarize the last 3 quarters' financial trends"
UPSTREAM = "fetch_financials"
UPSTREAM_POINTER = f"artifact:{UPSTREAM}.json"

# What the model is expected to write, and the shape the prompt teaches: read every
# entry of `datasets`, load each `csv` with pandas, print one labelled number. Growth
# from `shift(1)` rather than `pct_change()`, which warns about its fill method.
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

# The other half of a two-script step: the prompt invites the agent to split the work,
# and this is what the second script looks like — load, clean, derive, print. Its length
# is not padding, which is the whole point of the preview test below: JSON-escaped, one
# script of this shape is longer than everything the aggregator is ever shown.
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
    """A worker's slice, built the way the engine builds it — through the ledger, so an
    input it names is one some step really produced."""
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
    """The worker under test. A broker is built when a test does not care about one —
    warnings are published to nobody, which is what an unobserved run does anyway."""
    return AnalyticsWorker(
        provider=provider,
        store=store,
        tools=tuple(tools),
        broker=broker if broker is not None else Broker(),
        **bounds,
    )


def _call(code: str, call_id: str = "call-1", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=RUN_PYTHON_TOOL, arguments={"code": code} | arguments)


def _stored(store: ArtifactStore, pointer: str) -> AnalysisResult:
    """Read the artifact back through its own schema — the aggregator's view of it."""
    return AnalysisResult.model_validate(json.loads(store.get_text(pointer)))


def _seed_upstream(store: ArtifactStore, csv_text: str) -> str:
    """Write the artifact #5 would have written, so the executor resolves a real pointer."""
    dataset = RetrievedDataset(
        instruction="Load the quarterly financials",
        summary="Eight quarters of revenue, costs and profit.",
        datasets=[RetrievedTable(query='{"last_n": 8}', csv=csv_text)],
    )
    return store.put_text(f"{UPSTREAM}.json", dataset.model_dump_json(indent=2))


@pytest.mark.asyncio
async def test_worker_stores_the_analysis_and_returns_its_pointer(tmp_path: Path) -> None:
    """The sample subtask, end to end: one script, one summary, one artifact (#6)."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="", tool_calls=(_call("print(1 + 1)", inputs=[]),), usage_tokens=120
            ),
            AssistantTurn(text="Revenue grew 10.6% quarter over quarter.", usage_tokens=80),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="2025Q4 growth: 10.65%")])
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, [tool]).run(_context())

    assert pointer == "artifact:analyse_trends.json"
    analysis = _stored(store, pointer)
    assert analysis.figures == ["2025Q4 growth: 10.65%"]
    assert [item.stdout for item in analysis.computations] == analysis.figures
    assert analysis.computations[0].code == "print(1 + 1)"
    assert analysis.summary == "Revenue grew 10.6% quarter over quarter."
    assert analysis.instruction == "Compute quarter-over-quarter revenue growth"


@pytest.mark.asyncio
async def test_worker_offers_only_the_executor_and_keeps_the_request_out_of_the_system_prompt(
    tmp_path: Path,
) -> None:
    """One tool, by design: a retrieval tool here would let the agent bypass the plan's
    dependency and analyse data no step was ordered to fetch. Untrusted text stays a user
    turn (§11), and the upstream pointer is named there so the model can ask for it."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("print(1)"),), usage_tokens=10),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    store = ArtifactStore(tmp_path)
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
async def test_worker_computes_the_correct_quarter_over_quarter_growth(tmp_path: Path) -> None:
    """The ticket's spot-check, through the real executor and the committed dataset.

    Nothing is faked below the provider: the upstream artifact is a real
    `RetrievedDataset`, the script is real pandas in a real subprocess, and the figure it
    prints is checked against the same rows read here with the csv module. 2025Q4 revenue
    7015000 over 2025Q3 6340000 is +10.65%, which is what a reader can verify by hand.
    """
    financials = (default_data_dir() / FINANCIALS_CSV).read_text(encoding="utf-8")
    rows = list(csv.DictReader(financials.splitlines()))
    latest, previous = float(rows[-1]["revenue"]), float(rows[-2]["revenue"])
    expected = f"{rows[-1]['quarter']} revenue growth: {(latest / previous - 1) * 100:.2f}% QoQ"
    assert expected == "2025Q4 revenue growth: 10.65% QoQ"  # checked by hand, not by pandas

    store = ArtifactStore(tmp_path)
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
async def test_worker_finishes_after_the_model_corrects_a_failing_script(tmp_path: Path) -> None:
    """§6: a traceback is data the model reads and fixes, not an unwound loop.

    The real executor, because the traceback is the thing being relied on — a fake error
    string would assert the loop's plumbing and not that a broken script names its own
    fault. The failed call is not kept, so only the corrected one reaches the artifact.
    """
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("print(revenue)"),), usage_tokens=30),
            AssistantTurn(
                text="", tool_calls=(_call("print('total: 42')", "call-2"),), usage_tokens=30
            ),
            AssistantTurn(text="Recovered after a NameError.", usage_tokens=30),
        ]
    )
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, analytics_tools(store)).run(_context())

    analysis = _stored(store, pointer)
    assert [item.stdout.strip() for item in analysis.computations] == ["total: 42"]
    # The traceback reached the model as a flagged result on the following turn.
    failure = provider.send_calls[1].messages[-1].tool_results[0]
    assert (failure.call_id, failure.is_error) == ("call-1", True)
    assert "NameError" in failure.content


@pytest.mark.asyncio
async def test_worker_that_computed_nothing_fails_the_subtask(tmp_path: Path) -> None:
    """A summary with no computation behind it is the invented answer the design forbids."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("print(oops)"),), usage_tokens=20),
            AssistantTurn(text="Revenue grew by about 10%.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content="NameError: oops", is_error=True)])

    with pytest.raises(TaskFailure, match="without computing anything"):
        await _worker(provider, ArtifactStore(tmp_path), [tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_puts_every_figure_ahead_of_every_script_in_the_preview(
    tmp_path: Path,
) -> None:
    """Field order is the aggregator's budget: it is shown a preview, not the payload.

    Two computations, because one passes by accident: interleaving the pairs protects the
    first script's output and nothing else, and the prompt invites two scripts. One real
    pandas script escapes to more than the whole preview, so any code ahead of a figure
    is that figure elided — and the report is then written from a prompt that never saw
    the number it is supposed to name.
    """
    figures = ["mean quarterly revenue: 6112500", "2025Q4 revenue growth: 10.65% QoQ"]
    # One script, escaped, outweighs everything the aggregator is shown: put any code
    # ahead of a figure and that figure is past the cut.
    assert len(json.dumps(LEVELS_SCRIPT)) > DEFAULT_PREVIEW_LIMIT
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(LEVELS_SCRIPT),), usage_tokens=20),
            AssistantTurn(text="", tool_calls=(_call(QOQ_SCRIPT, "call-2"),), usage_tokens=20),
            AssistantTurn(text="Growth accelerated into Q4.", usage_tokens=20),
        ]
    )
    tool = FakeTool(RUN_PYTHON_TOOL, [ToolResponse(content=figure) for figure in figures])
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, [tool]).run(_context())

    preview = store.preview(pointer)
    assert "Growth accelerated into Q4." in preview
    for figure in figures:
        assert figure in preview
    assert "elided" in preview  # the scripts are what fell off the end, not the numbers


@pytest.mark.asyncio
async def test_worker_propagates_cancellation(tmp_path: Path) -> None:
    """§10: a cancelled run unwinds through the worker, never swallowed."""
    provider = FakeProvider(turns=[AssistantTurn(text="unreached")], blocker=asyncio.Event())
    worker = _worker(provider, ArtifactStore(tmp_path), [FakeTool(RUN_PYTHON_TOOL, [])])

    task = asyncio.create_task(worker.run(_context()))
    await asyncio.sleep(0)  # let it reach the blocked provider call
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_worker_without_tools_is_a_wiring_bug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        AnalyticsWorker(
            provider=FakeProvider(),
            store=ArtifactStore(tmp_path),
            tools=(),
            broker=Broker(),
        )
