"""Tests for the Data Retrieval worker's tool-use loop.

Asserts the contract the engine and the next agent depend on: a pointer out, a
`RetrievedDataset` behind it, both bounds enforced, tool failures fed back to the model
rather than raised, and cancellation propagated.

The tools here are `conftest.FakeTool`; the real ones are exercised in `test_tools.py`.
"""

import asyncio
import json
from collections.abc import Sequence

import pytest

from conftest import FakeProvider, FakeTool, tool_call
from orchestra.agents.toolsets import FETCH_DATA_TOOL, SEARCH_TOOL
from orchestra.agents.workers.data_retrieval import (
    DataRetrievalWorker,
    RetrievedDataset,
)
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, Subtask, SubtaskContext, TaskEvent, TaskState
from orchestra.providers.base import AssistantTurn
from orchestra.tools.base import BaseTool, ToolResponse
from orchestra.tools.fetch_data import INLINED_KEY, POINTER_KEY

REQUEST = "Summarize the last 3 quarters' financial trends"
CSV = "quarter,revenue,costs,profit\n2025Q2,1200,700,500\n"
POINTER = "artifact:quarterly_financials.csv"


def _fetched(csv: str = CSV) -> ToolResponse:
    """What `fetch_data` returns for a file small enough to inline: the text, plus the
    pointer and the inline flag the worker reads out of `metadata` rather than the prose.
    """
    return ToolResponse(content=csv, metadata={POINTER_KEY: POINTER, INLINED_KEY: "true"})


def _fetched_by_pointer() -> ToolResponse:
    """What it returns for a file too large to inline: a summary and the pointer."""
    return ToolResponse(
        content=f"quarterly_financials: CSV with columns quarter, revenue. Stored as {POINTER}.",
        metadata={POINTER_KEY: POINTER, INLINED_KEY: "false"},
    )


def _context(**overrides: object) -> SubtaskContext:
    """A worker's slice, built the way the engine builds it."""
    fields: dict[str, object] = {
        "id": "fetch_financials",
        "role": AgentRole.DATA_RETRIEVAL,
        "instruction": "Fetch the last 3 quarters of financials",
    }
    subtask = Subtask.model_validate(fields | overrides)
    return TaskState(user_request=REQUEST).state_slice(subtask)


def _worker(
    provider: FakeProvider,
    store: ArtifactStore,
    tools: Sequence[BaseTool],
    broker: Broker[TaskEvent] | None = None,
    **bounds: int,
) -> DataRetrievalWorker:
    """The worker under test; an unsubscribed `Broker` stands in when a test ignores
    warnings, as an unobserved run would."""
    return DataRetrievalWorker(
        provider=provider,
        store=store,
        tools=tuple(tools),
        broker=broker if broker is not None else Broker(),
        **bounds,
    )


def _stored(store: ArtifactStore, pointer: str) -> RetrievedDataset:
    """Read the artifact back through its own schema — the next agent's view of it."""
    return RetrievedDataset.model_validate(json.loads(store.get_text(pointer)))


@pytest.mark.asyncio
async def test_worker_stores_the_fetched_dataset_and_returns_its_pointer(
    store: ArtifactStore,
) -> None:
    """One fetch, one summary, one artifact (#5)."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, name="quarterly_financials"),),
                usage_tokens=120,
            ),
            AssistantTurn(text="Retrieved three quarters of revenue and costs.", usage_tokens=80),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched()])

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert pointer == "artifact:fetch_financials.json"
    dataset = _stored(store, pointer)
    assert [table.csv for table in dataset.datasets] == [CSV]
    assert dataset.summary == "Retrieved three quarters of revenue and costs."
    assert dataset.instruction == "Fetch the last 3 quarters of financials"
    assert csv_tool.calls[0].arguments == {"name": "quarterly_financials"}


@pytest.mark.asyncio
async def test_worker_offers_every_tool_and_keeps_the_request_out_of_the_system_prompt(
    store: ArtifactStore,
) -> None:
    """Both tools are on the table each turn, and untrusted text stays a user turn (§11)."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call(FETCH_DATA_TOOL),), usage_tokens=10),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    tools = [
        FakeTool(FETCH_DATA_TOOL, [_fetched()]),
        FakeTool(SEARCH_TOOL, []),
    ]

    await _worker(provider, store, tools).run(_context())

    offered = {spec.name for spec in provider.send_calls[0].tools}
    assert offered == {FETCH_DATA_TOOL, SEARCH_TOOL}
    assert REQUEST not in provider.send_calls[0].system
    assert REQUEST in provider.send_calls[0].messages[0].content


@pytest.mark.asyncio
async def test_worker_records_search_results_as_sources(store: ArtifactStore) -> None:
    """The second tool's answer is kept as provenance, not discarded."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(
                    tool_call(FETCH_DATA_TOOL, "c1", name="quarterly_financials"),
                    tool_call(SEARCH_TOOL, "c2", query="saas margin benchmark"),
                ),
                usage_tokens=200,
            ),
            AssistantTurn(text="Figures plus sector context.", usage_tokens=50),
        ]
    )
    tools = [
        FakeTool(FETCH_DATA_TOOL, [_fetched()]),
        FakeTool(SEARCH_TOOL, [ToolResponse(content="Sector margins run 70-80%.")]),
    ]

    pointer = await _worker(provider, store, tools).run(_context())

    dataset = _stored(store, pointer)
    assert [table.csv for table in dataset.datasets] == [CSV]
    assert [source.query for source in dataset.sources] == ["saas margin benchmark"]
    assert dataset.sources[0].result == "Sector margins run 70-80%."


@pytest.mark.asyncio
async def test_worker_feeds_a_tool_failure_back_to_the_model_instead_of_raising(
    store: ArtifactStore,
) -> None:
    """§6: a tool error is data the model reads and corrects, not an unwound loop."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, name="headcount"),),
                usage_tokens=30,
            ),
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, "c2", name="quarterly_financials"),),
                usage_tokens=30,
            ),
            AssistantTurn(text="Recovered after a bad dataset name.", usage_tokens=30),
        ]
    )
    csv_tool = FakeTool(
        FETCH_DATA_TOOL,
        [
            ToolResponse(content="There is no dataset named 'headcount'.", is_error=True),
            _fetched(),
        ],
    )

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert [table.csv for table in _stored(store, pointer).datasets] == [CSV]
    # The failure reached the model as a flagged result on the following turn.
    replayed = provider.send_calls[1].messages[-1].tool_results
    assert [(result.call_id, result.is_error) for result in replayed] == [("call-1", True)]


@pytest.mark.asyncio
async def test_worker_answers_an_unknown_tool_name_without_ending_the_subtask(
    store: ArtifactStore,
) -> None:
    """A hallucinated tool is the model's to correct — it gets told the real ones."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call("fetch_from_sql"),), usage_tokens=20),
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, "c2", name="quarterly_financials"),),
                usage_tokens=20,
            ),
            AssistantTurn(text="Used the right tool.", usage_tokens=20),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched()])

    await _worker(provider, store, [csv_tool]).run(_context())

    answer = provider.send_calls[1].messages[-1].tool_results[0]
    assert answer.is_error is True
    assert FETCH_DATA_TOOL in answer.content


@pytest.mark.asyncio
async def test_worker_that_retrieved_nothing_fails_the_subtask(store: ArtifactStore) -> None:
    """A summary with no data behind it is an invented answer."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call(FETCH_DATA_TOOL),), usage_tokens=20),
            AssistantTurn(text="I could not find the data.", usage_tokens=20),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [ToolResponse(content="no such file", is_error=True)])

    with pytest.raises(TaskFailure, match="without retrieving anything"):
        await _worker(provider, store, [csv_tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_still_calling_tools_at_the_turn_cap_fails_the_subtask(
    store: ArtifactStore,
) -> None:
    """§10: the loop is bounded, and exceeding the bound is a failure, not a retry."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call(FETCH_DATA_TOOL),), usage_tokens=10)
            for _ in range(2)
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched() for _ in range(2)])

    with pytest.raises(TaskFailure, match="after 2 turns"):
        await _worker(provider, store, [csv_tool], max_turns=2).run(_context())


@pytest.mark.asyncio
async def test_worker_over_its_token_budget_fails_the_subtask(store: ArtifactStore) -> None:
    """The second bound: turns alone will not catch a model making expensive calls."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call(FETCH_DATA_TOOL),), usage_tokens=5_000),
            AssistantTurn(text="never reached", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched()])

    with pytest.raises(TaskFailure, match="budget before finishing"):
        await _worker(provider, store, [csv_tool], token_budget=100).run(_context())


@pytest.mark.asyncio
async def test_worker_propagates_cancellation(store: ArtifactStore) -> None:
    """§10: a cancelled run unwinds through the worker, never swallowed."""
    provider = FakeProvider(turns=[AssistantTurn(text="unreached")], blocker=asyncio.Event())
    csv_tool = FakeTool(FETCH_DATA_TOOL, [])
    worker = _worker(provider, store, [csv_tool])

    task = asyncio.create_task(worker.run(_context()))
    await asyncio.sleep(0)  # let it reach the blocked provider call
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_worker_without_tools_is_a_wiring_bug(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        DataRetrievalWorker(
            provider=FakeProvider(),
            store=store,
            tools=(),
            broker=Broker(),
        )


# --------------------------------------------------------------------------
# Regressions from the review of PR #27.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_replays_the_providers_own_turn_verbatim(store: ArtifactStore) -> None:
    """Rebuilding the assistant turn from `text` and `tool_calls` drops the blocks this
    codebase never decodes, and the next request is rejected; `raw_content` carries them."""
    blocks = object()  # opaque, exactly as the loop must treat it
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, name="quarterly_financials"),),
                usage_tokens=10,
                raw_content=blocks,
            ),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched()])

    await _worker(provider, store, [csv_tool]).run(_context())

    replayed = provider.send_calls[1].messages[1]
    assert replayed.raw_content is blocks


@pytest.mark.asyncio
async def test_worker_keeps_every_successful_fetch_not_just_the_last(store: ArtifactStore) -> None:
    """Regression: a step may need two of the bundled files. Keeping only the last stored
    half the data under a summary describing all of it."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, "c1", name="quarterly_financials"),),
                usage_tokens=10,
            ),
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, "c2", name="expense_breakdown"),),
                usage_tokens=10,
            ),
            AssistantTurn(text="Revenue and costs.", usage_tokens=10),
        ]
    )
    revenue, costs = "quarter,revenue\n2025Q2,1200\n", "quarter,costs\n2025Q2,700\n"
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched(revenue), _fetched(costs)])

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    dataset = _stored(store, pointer)
    assert [table.csv for table in dataset.datasets] == [revenue, costs]
    # The arguments ride along, so a reader can tell which file answered what.
    assert "quarterly_financials" in dataset.datasets[0].query


@pytest.mark.asyncio
async def test_worker_records_a_pointer_when_the_file_was_too_large_to_inline(
    store: ArtifactStore,
) -> None:
    """#40: the rows stay out of the transcript, so the artifact carries the pointer and
    an empty `csv` — the analysis step opens the file itself."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, name="quarterly_financials"),),
                usage_tokens=10,
            ),
            AssistantTurn(text="The file is large; passing it on by pointer.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched_by_pointer()])

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    (table,) = _stored(store, pointer).datasets
    assert table.pointer == POINTER
    # Not the summary: storing prose as `csv` is what would reach pd.read_csv.
    assert table.csv == ""


@pytest.mark.asyncio
async def test_worker_keeps_an_earlier_success_when_a_later_call_fails(
    store: ArtifactStore,
) -> None:
    """The failure path must not discard what already succeeded this turn."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, "c1", name="quarterly_financials"),),
                usage_tokens=10,
            ),
            AssistantTurn(
                text="",
                tool_calls=(tool_call(FETCH_DATA_TOOL, "c2", name="headcount"),),
                usage_tokens=10,
            ),
            AssistantTurn(text="One worked, one did not.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(
        FETCH_DATA_TOOL,
        [_fetched(), ToolResponse(content="There is no dataset named 'headcount'.", is_error=True)],
    )

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert [table.csv for table in _stored(store, pointer).datasets] == [CSV]


@pytest.mark.asyncio
async def test_worker_does_not_record_a_search_that_matched_nothing(store: ArtifactStore) -> None:
    """Regression: `search` reports a miss as a success so the model does not retry it. If
    the worker recorded that as a source, a run that retrieved nothing would have written
    an artifact and been marked DONE."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="", tool_calls=(tool_call(SEARCH_TOOL, query="unrelated"),), usage_tokens=10
            ),
            AssistantTurn(text="I found nothing.", usage_tokens=10),
        ]
    )
    search_tool = FakeTool(
        SEARCH_TOOL, [ToolResponse(content="Nothing matched 'unrelated'.", is_empty=True)]
    )

    with pytest.raises(TaskFailure, match="without retrieving anything"):
        await _worker(provider, store, [search_tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_treats_a_truncated_reply_as_a_failure_not_a_finished_turn(
    store: ArtifactStore,
) -> None:
    """A reply cut off by the output limit has no tool call — the same shape as "done".
    Without the check, half a sentence is stored as the summary and the run reports
    success."""
    provider = FakeProvider(
        turns=[AssistantTurn(text="I was about to sa", usage_tokens=10, stop_reason="max_tokens")]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [])

    with pytest.raises(TaskFailure, match="cut off by the model's output limit"):
        await _worker(provider, store, [csv_tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_bound_failure_names_what_was_lost(store: ArtifactStore) -> None:
    """§8: the message has to distinguish "raise the cap" from "debug the agent"."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call(FETCH_DATA_TOOL),), usage_tokens=10)
            for _ in range(2)
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched() for _ in range(2)])

    with pytest.raises(TaskFailure, match=r"kept results from fetch_data x2, which are lost"):
        await _worker(provider, store, [csv_tool], max_turns=2).run(_context())


@pytest.mark.asyncio
async def test_worker_replays_narration_that_accompanied_a_tool_call(store: ArtifactStore) -> None:
    """A turn can carry text *and* tool calls; the text is narration, not the summary."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="Let me check the last three quarters.",
                tool_calls=(tool_call(FETCH_DATA_TOOL, name="quarterly_financials"),),
                usage_tokens=10,
            ),
            AssistantTurn(text="The real summary.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched()])

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert _stored(store, pointer).summary == "The real summary."
    assert provider.send_calls[1].messages[1].content == "Let me check the last three quarters."


@pytest.mark.asyncio
async def test_worker_with_no_usage_reported_is_still_bounded_by_its_turn_cap(
    store: ArtifactStore,
) -> None:
    """A provider reporting zero tokens leaves the budget inert — turns are the backstop."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call(FETCH_DATA_TOOL),), usage_tokens=0)
            for _ in range(3)
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched() for _ in range(3)])

    with pytest.raises(TaskFailure, match="after 3 turns"):
        await _worker(provider, store, [csv_tool], max_turns=3).run(_context())


@pytest.mark.asyncio
async def test_worker_publishes_a_tool_warning_without_failing_the_step(
    store: ArtifactStore,
) -> None:
    """A degraded tool is news for the operator, not a reason to fail a step that worked,
    and only the worker can see it — the engine publishes transitions, not mid-step
    events."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="", tool_calls=(tool_call(SEARCH_TOOL, query="margins"),), usage_tokens=10
            ),
            AssistantTurn(text="Answered from the corpus.", usage_tokens=10),
        ]
    )
    search_tool = FakeTool(
        SEARCH_TOOL,
        [ToolResponse(content="a note", warning="Live search was unavailable: HTTP 401.")],
    )
    broker: Broker[TaskEvent] = Broker()

    async with broker.subscribe() as queue:
        pointer = await _worker(provider, store, [search_tool], broker).run(_context())
        published = [queue.get_nowait() for _ in range(queue.qsize())]

    assert pointer == "artifact:fetch_financials.json"
    warnings = [event for event in published if event.kind is EventKind.SUBTASK_WARNING]
    assert [(event.subtask_id, event.message) for event in warnings] == [
        ("fetch_financials", "Live search was unavailable: HTTP 401.")
    ]


@pytest.mark.asyncio
async def test_worker_publishes_nothing_when_no_tool_degraded(store: ArtifactStore) -> None:
    """The quiet path stays quiet — a warning per call would train the eye to ignore it."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(tool_call(FETCH_DATA_TOOL),), usage_tokens=10),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(FETCH_DATA_TOOL, [_fetched()])
    broker: Broker[TaskEvent] = Broker()

    async with broker.subscribe() as queue:
        await _worker(provider, store, [csv_tool], broker).run(_context())
        published = [queue.get_nowait() for _ in range(queue.qsize())]

    assert [event for event in published if event.kind is EventKind.SUBTASK_WARNING] == []
