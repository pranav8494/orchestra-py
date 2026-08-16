"""Tests for the Data Retrieval worker's tool-use loop (CONVENTIONS.md §12).

What is asserted is the contract the engine and the next agent depend on: a pointer out,
a `RetrievedDataset` behind it, both bounds enforced, tool failures fed back to the
model rather than raised, and cancellation propagated.

The tools here are fakes. The real ones are exercised in `test_tools.py` — this file is
about the loop, and a test that also parsed a CSV could not say which half broke.
"""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from conftest import FakeProvider
from orchestra.agents.toolsets import QUERY_CSV_TOOL, SEARCH_TOOL
from orchestra.agents.workers.data_retrieval import (
    DataRetrievalWorker,
    RetrievedDataset,
)
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, Subtask, SubtaskContext, TaskEvent, TaskState
from orchestra.providers.base import AssistantTurn
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse, ToolSpec

REQUEST = "Summarize the last 3 quarters' financial trends"
CSV = "quarter,revenue,costs,profit\n2025Q2,1200,700,500\n"


class FakeTool:
    """A `BaseTool` that answers from a queue and records what it was called with."""

    def __init__(self, name: str, responses: list[ToolResponse]) -> None:
        self._name = name
        self._responses = responses
        self.calls: list[ToolCall] = []

    def info(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"fake {self._name}",
            input_schema={"type": "object", "properties": {}},
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        self.calls.append(call)
        if not self._responses:
            raise AssertionError(f"FakeTool {self._name!r} has no queued response")
        return self._responses.pop(0)


_PROTOCOL_CHECK: BaseTool = FakeTool("x", [])


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
    """The worker under test. A broker is built when a test does not care about one —
    warnings are published to nobody, which is what an unobserved run does anyway."""
    return DataRetrievalWorker(
        provider=provider,
        store=store,
        tools=tuple(tools),
        broker=broker if broker is not None else Broker(),
        **bounds,
    )


def _call(name: str, call_id: str = "call-1", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _stored(store: ArtifactStore, pointer: str) -> RetrievedDataset:
    """Read the artifact back through its own schema — the next agent's view of it."""
    return RetrievedDataset.model_validate(json.loads(store.get_text(pointer)))


@pytest.mark.asyncio
async def test_worker_stores_the_filtered_dataset_and_returns_its_pointer(
    tmp_path: Path,
) -> None:
    """The sample subtask, end to end: one query, one summary, one artifact (#5)."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL, last_n=3),), usage_tokens=120),
            AssistantTurn(text="Retrieved three quarters of revenue and costs.", usage_tokens=80),
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)])
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert pointer == "artifact:fetch_financials.json"
    dataset = _stored(store, pointer)
    assert [table.csv for table in dataset.datasets] == [CSV]
    assert dataset.summary == "Retrieved three quarters of revenue and costs."
    assert dataset.instruction == "Fetch the last 3 quarters of financials"
    assert csv_tool.calls[0].arguments == {"last_n": 3}


@pytest.mark.asyncio
async def test_worker_offers_every_tool_and_keeps_the_request_out_of_the_system_prompt(
    tmp_path: Path,
) -> None:
    """Both tools are on the table each turn, and untrusted text stays a user turn (§11)."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL),), usage_tokens=10),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    tools = [
        FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)]),
        FakeTool(SEARCH_TOOL, []),
    ]

    await _worker(provider, ArtifactStore(tmp_path), tools).run(_context())

    offered = {spec.name for spec in provider.send_calls[0].tools}
    assert offered == {QUERY_CSV_TOOL, SEARCH_TOOL}
    assert REQUEST not in provider.send_calls[0].system
    assert REQUEST in provider.send_calls[0].messages[0].content


@pytest.mark.asyncio
async def test_worker_records_search_results_as_sources(tmp_path: Path) -> None:
    """Multi-tool usage: the second tool's answer is kept as provenance, not discarded."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(
                    _call(QUERY_CSV_TOOL, "c1", last_n=3),
                    _call(SEARCH_TOOL, "c2", query="saas margin benchmark"),
                ),
                usage_tokens=200,
            ),
            AssistantTurn(text="Figures plus sector context.", usage_tokens=50),
        ]
    )
    tools = [
        FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)]),
        FakeTool(SEARCH_TOOL, [ToolResponse(content="Sector margins run 70-80%.")]),
    ]
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, tools).run(_context())

    dataset = _stored(store, pointer)
    assert [table.csv for table in dataset.datasets] == [CSV]
    assert [source.query for source in dataset.sources] == ["saas margin benchmark"]
    assert dataset.sources[0].result == "Sector margins run 70-80%."


@pytest.mark.asyncio
async def test_worker_feeds_a_tool_failure_back_to_the_model_instead_of_raising(
    tmp_path: Path,
) -> None:
    """§6: a tool error is data the model reads and corrects, not an unwound loop."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="", tool_calls=(_call(QUERY_CSV_TOOL, columns=["margin"]),), usage_tokens=30
            ),
            AssistantTurn(
                text="", tool_calls=(_call(QUERY_CSV_TOOL, "c2", last_n=3),), usage_tokens=30
            ),
            AssistantTurn(text="Recovered after a bad column.", usage_tokens=30),
        ]
    )
    csv_tool = FakeTool(
        QUERY_CSV_TOOL,
        [
            ToolResponse(content="No column 'margin'. Columns: quarter, revenue.", is_error=True),
            ToolResponse(content=CSV),
        ],
    )
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert [table.csv for table in _stored(store, pointer).datasets] == [CSV]
    # The failure reached the model as a flagged result on the following turn.
    replayed = provider.send_calls[1].messages[-1].tool_results
    assert [(result.call_id, result.is_error) for result in replayed] == [("call-1", True)]


@pytest.mark.asyncio
async def test_worker_answers_an_unknown_tool_name_without_ending_the_subtask(
    tmp_path: Path,
) -> None:
    """A hallucinated tool is the model's mistake to correct — it gets told the real ones."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("fetch_from_sql"),), usage_tokens=20),
            AssistantTurn(
                text="", tool_calls=(_call(QUERY_CSV_TOOL, "c2", last_n=3),), usage_tokens=20
            ),
            AssistantTurn(text="Used the right tool.", usage_tokens=20),
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)])

    await _worker(provider, ArtifactStore(tmp_path), [csv_tool]).run(_context())

    answer = provider.send_calls[1].messages[-1].tool_results[0]
    assert answer.is_error is True
    assert QUERY_CSV_TOOL in answer.content


@pytest.mark.asyncio
async def test_worker_that_retrieved_nothing_fails_the_subtask(tmp_path: Path) -> None:
    """A summary with no data behind it is the invented answer the design forbids."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL),), usage_tokens=20),
            AssistantTurn(text="I could not find the data.", usage_tokens=20),
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content="no such file", is_error=True)])

    with pytest.raises(TaskFailure, match="without retrieving anything"):
        await _worker(provider, ArtifactStore(tmp_path), [csv_tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_still_calling_tools_at_the_turn_cap_fails_the_subtask(
    tmp_path: Path,
) -> None:
    """§10: the loop is bounded, and exceeding the bound is a failure, not a retry."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL),), usage_tokens=10)
            for _ in range(2)
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV) for _ in range(2)])

    with pytest.raises(TaskFailure, match="after 2 turns"):
        await _worker(provider, ArtifactStore(tmp_path), [csv_tool], max_turns=2).run(_context())


@pytest.mark.asyncio
async def test_worker_over_its_token_budget_fails_the_subtask(tmp_path: Path) -> None:
    """The second bound: turns alone will not catch a model making expensive calls."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL),), usage_tokens=5_000),
            AssistantTurn(text="never reached", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)])

    with pytest.raises(TaskFailure, match="budget before finishing"):
        await _worker(provider, ArtifactStore(tmp_path), [csv_tool], token_budget=100).run(
            _context()
        )


@pytest.mark.asyncio
async def test_worker_propagates_cancellation(tmp_path: Path) -> None:
    """§10: a cancelled run unwinds through the worker, never swallowed."""
    provider = FakeProvider(turns=[AssistantTurn(text="unreached")], blocker=asyncio.Event())
    csv_tool = FakeTool(QUERY_CSV_TOOL, [])
    worker = _worker(provider, ArtifactStore(tmp_path), [csv_tool])

    task = asyncio.create_task(worker.run(_context()))
    await asyncio.sleep(0)  # let it reach the blocked provider call
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_worker_without_tools_is_a_wiring_bug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        DataRetrievalWorker(
            provider=FakeProvider(),
            store=ArtifactStore(tmp_path),
            tools=(),
            broker=Broker(),
        )


# --------------------------------------------------------------------------
# Regressions from the review of PR #27.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_replays_the_providers_own_turn_verbatim(tmp_path: Path) -> None:
    """The model reasons before calling a tool, and the API wants that reasoning back.

    Rebuilding the assistant turn from `text` and `tool_calls` drops the blocks this
    codebase never decodes, and the next request is rejected. `raw_content` is the
    handle that carries them, so what is asserted is that the loop hands it back.
    """
    blocks = object()  # opaque, exactly as the loop must treat it
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call(QUERY_CSV_TOOL, last_n=3),),
                usage_tokens=10,
                raw_content=blocks,
            ),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)])

    await _worker(provider, ArtifactStore(tmp_path), [csv_tool]).run(_context())

    replayed = provider.send_calls[1].messages[1]
    assert replayed.raw_content is blocks


@pytest.mark.asyncio
async def test_worker_keeps_every_successful_query_not_just_the_last(tmp_path: Path) -> None:
    """Regression: two complementary queries must not overwrite each other.

    The tool advertises a `columns` filter, so splitting a request across two calls is
    behaviour it invites. Keeping only the last silently stored half the data under a
    summary describing all of it.
    """
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call(QUERY_CSV_TOOL, "c1", columns=["revenue"]),),
                usage_tokens=10,
            ),
            AssistantTurn(
                text="",
                tool_calls=(_call(QUERY_CSV_TOOL, "c2", columns=["costs"]),),
                usage_tokens=10,
            ),
            AssistantTurn(text="Revenue and costs.", usage_tokens=10),
        ]
    )
    revenue, costs = "quarter,revenue\n2025Q2,1200\n", "quarter,costs\n2025Q2,700\n"
    csv_tool = FakeTool(
        QUERY_CSV_TOOL, [ToolResponse(content=revenue), ToolResponse(content=costs)]
    )
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    dataset = _stored(store, pointer)
    assert [table.csv for table in dataset.datasets] == [revenue, costs]
    # The arguments ride along, so a reader can tell which table answered which question.
    assert "revenue" in dataset.datasets[0].query


@pytest.mark.asyncio
async def test_worker_keeps_an_earlier_success_when_a_later_call_fails(tmp_path: Path) -> None:
    """The failure path must not discard what already succeeded this turn."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="", tool_calls=(_call(QUERY_CSV_TOOL, "c1", last_n=3),), usage_tokens=10
            ),
            AssistantTurn(
                text="",
                tool_calls=(_call(QUERY_CSV_TOOL, "c2", columns=["margin"]),),
                usage_tokens=10,
            ),
            AssistantTurn(text="One worked, one did not.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(
        QUERY_CSV_TOOL,
        [ToolResponse(content=CSV), ToolResponse(content="No column named margin.", is_error=True)],
    )
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert [table.csv for table in _stored(store, pointer).datasets] == [CSV]


@pytest.mark.asyncio
async def test_worker_does_not_record_a_search_that_matched_nothing(tmp_path: Path) -> None:
    """Regression: "nothing matched" is not provenance, and must not pass the guard.

    `search` reports a miss as a success so the model does not retry it pointlessly. If
    the worker recorded that as a source, a run that retrieved nothing at all would have
    written an artifact and been marked DONE.
    """
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="", tool_calls=(_call(SEARCH_TOOL, query="unrelated"),), usage_tokens=10
            ),
            AssistantTurn(text="I found nothing.", usage_tokens=10),
        ]
    )
    search_tool = FakeTool(
        SEARCH_TOOL, [ToolResponse(content="Nothing matched 'unrelated'.", is_empty=True)]
    )

    with pytest.raises(TaskFailure, match="without retrieving anything"):
        await _worker(provider, ArtifactStore(tmp_path), [search_tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_treats_a_truncated_reply_as_a_failure_not_a_finished_turn(
    tmp_path: Path,
) -> None:
    """A reply cut off by the output limit has no tool call — the same shape as "done".

    Thinking shares the output budget on this model, so truncation is reachable. Without
    the check, half a sentence is stored as the summary and the run reports success.
    """
    provider = FakeProvider(
        turns=[AssistantTurn(text="I was about to sa", usage_tokens=10, stop_reason="max_tokens")]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [])

    with pytest.raises(TaskFailure, match="cut off by the model's output limit"):
        await _worker(provider, ArtifactStore(tmp_path), [csv_tool]).run(_context())


@pytest.mark.asyncio
async def test_worker_bound_failure_names_what_was_lost(tmp_path: Path) -> None:
    """§8: the message has to distinguish "raise the cap" from "debug the agent"."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL),), usage_tokens=10)
            for _ in range(2)
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV) for _ in range(2)])

    with pytest.raises(TaskFailure, match="2 table\\(s\\) and 0 search result\\(s\\)"):
        await _worker(provider, ArtifactStore(tmp_path), [csv_tool], max_turns=2).run(_context())


@pytest.mark.asyncio
async def test_worker_replays_narration_that_accompanied_a_tool_call(tmp_path: Path) -> None:
    """A turn can carry text *and* tool calls; the text is narration, not the summary."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="Let me check the last three quarters.",
                tool_calls=(_call(QUERY_CSV_TOOL, last_n=3),),
                usage_tokens=10,
            ),
            AssistantTurn(text="The real summary.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)])
    store = ArtifactStore(tmp_path)

    pointer = await _worker(provider, store, [csv_tool]).run(_context())

    assert _stored(store, pointer).summary == "The real summary."
    assert provider.send_calls[1].messages[1].content == "Let me check the last three quarters."


@pytest.mark.asyncio
async def test_worker_with_no_usage_reported_is_still_bounded_by_its_turn_cap(
    tmp_path: Path,
) -> None:
    """A provider reporting zero tokens leaves the budget inert — turns are the backstop."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL),), usage_tokens=0)
            for _ in range(3)
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV) for _ in range(3)])

    with pytest.raises(TaskFailure, match="after 3 turns"):
        await _worker(provider, ArtifactStore(tmp_path), [csv_tool], max_turns=3).run(_context())


@pytest.mark.asyncio
async def test_worker_publishes_a_tool_warning_without_failing_the_step(tmp_path: Path) -> None:
    """A degraded tool is news for the operator, not a reason to fail a step that worked.

    The worker is the only thing that can see it: the engine publishes the step's
    transitions, and this happens partway through one.
    """
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="", tool_calls=(_call(SEARCH_TOOL, query="margins"),), usage_tokens=10
            ),
            AssistantTurn(text="Answered from the corpus.", usage_tokens=10),
        ]
    )
    search_tool = FakeTool(
        SEARCH_TOOL,
        [ToolResponse(content="a note", warning="Live search was unavailable: HTTP 401.")],
    )
    broker: Broker[TaskEvent] = Broker()
    store = ArtifactStore(tmp_path)

    async with broker.subscribe() as queue:
        pointer = await _worker(provider, store, [search_tool], broker).run(_context())
        published = [queue.get_nowait() for _ in range(queue.qsize())]

    # The step still produced its artifact.
    assert pointer == "artifact:fetch_financials.json"
    warnings = [event for event in published if event.kind is EventKind.SUBTASK_WARNING]
    assert [(event.subtask_id, event.message) for event in warnings] == [
        ("fetch_financials", "Live search was unavailable: HTTP 401.")
    ]


@pytest.mark.asyncio
async def test_worker_publishes_nothing_when_no_tool_degraded(tmp_path: Path) -> None:
    """The quiet path stays quiet — a warning per call would train the eye to ignore it."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(QUERY_CSV_TOOL),), usage_tokens=10),
            AssistantTurn(text="Done.", usage_tokens=10),
        ]
    )
    csv_tool = FakeTool(QUERY_CSV_TOOL, [ToolResponse(content=CSV)])
    broker: Broker[TaskEvent] = Broker()

    async with broker.subscribe() as queue:
        await _worker(provider, ArtifactStore(tmp_path), [csv_tool], broker).run(_context())
        published = [queue.get_nowait() for _ in range(queue.qsize())]

    assert [event for event in published if event.kind is EventKind.SUBTASK_WARNING] == []
