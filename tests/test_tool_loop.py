"""Tests for the shared tool-use loop the worker agents run (CONVENTIONS.md §12).

What is asserted is what a worker built on `ToolLoop` may assume: a summary and the
calls worth keeping, tool failures fed back to the model rather than raised, both bounds
enforced with the caller's label in the message, and cancellation propagated.

The retrieval-shaped assertions stay in `test_data_retrieval.py` — this file never names
a real tool, because a loop that only worked for `query_csv` would still pass there.
"""

import asyncio
from collections.abc import Sequence

import pytest

from conftest import FakeProvider
from orchestra.agents.workers.tool_loop import LoopResult, ToolLoop
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, EventKind, Subtask, SubtaskContext, TaskEvent, TaskState
from orchestra.providers.base import AssistantTurn
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse

# Imported rather than copied: §2 applies to test helpers too, and this one already
# exists in the suite that first needed it.
from test_data_retrieval import FakeTool

REQUEST = "Summarize the last 3 quarters' financial trends"
SYSTEM = "You are a test agent."
LOOKUP = "lookup"
OTHER = "other"


def _context() -> SubtaskContext:
    """A worker's slice, built the way the engine builds it."""
    subtask = Subtask(id="analyse_trend", role=AgentRole.ANALYTICS, instruction="Analyse the trend")
    return TaskState(user_request=REQUEST).state_slice(subtask)


def _loop(
    provider: FakeProvider,
    tools: Sequence[BaseTool],
    broker: Broker[TaskEvent] | None = None,
    **bounds: int,
) -> ToolLoop:
    """The loop under test, labelled the way an Analytics worker would label it."""
    return ToolLoop(
        provider=provider,
        broker=broker if broker is not None else Broker(),
        tools=tuple(tools),
        system_prompt=SYSTEM,
        label="Analysis",
        **bounds,
    )


def _call(name: str, call_id: str = "call-1", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_loop_returns_the_models_summary_and_every_kept_call() -> None:
    """The happy path: the closing text, and the calls in the order they were made."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call(LOOKUP, "c1", q="a"), _call(OTHER, "c2", q="b")),
                usage_tokens=100,
            ),
            AssistantTurn(text="Both answered.", usage_tokens=20),
        ]
    )
    tools = [
        FakeTool(LOOKUP, [ToolResponse(content="first")]),
        FakeTool(OTHER, [ToolResponse(content="second")]),
    ]

    result = await _loop(provider, tools).run(_context())

    assert isinstance(result, LoopResult)
    assert result.summary == "Both answered."
    assert [(outcome.call.name, outcome.response.content) for outcome in result.kept] == [
        (LOOKUP, "first"),
        (OTHER, "second"),
    ]
    # The system prompt it was constructed with, and every tool, on every turn (§11).
    assert provider.send_calls[0].system == SYSTEM
    assert {spec.name for spec in provider.send_calls[0].tools} == {LOOKUP, OTHER}
    assert REQUEST in provider.send_calls[0].messages[0].content


@pytest.mark.asyncio
async def test_loop_feeds_a_tool_error_back_to_the_model_and_does_not_keep_it() -> None:
    """§6: an error is data the model reads and corrects, and never a kept result."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c1"),), usage_tokens=10),
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c2"),), usage_tokens=10),
            AssistantTurn(text="Recovered.", usage_tokens=10),
        ]
    )
    tool = FakeTool(
        LOOKUP,
        [ToolResponse(content="bad argument; try 'q'", is_error=True), ToolResponse(content="ok")],
    )

    result = await _loop(provider, [tool]).run(_context())

    assert [outcome.response.content for outcome in result.kept] == ["ok"]
    replayed = provider.send_calls[1].messages[-1].tool_results
    assert [(item.call_id, item.is_error, item.content) for item in replayed] == [
        ("c1", True, "bad argument; try 'q'")
    ]


@pytest.mark.asyncio
async def test_loop_does_not_keep_a_result_that_matched_nothing() -> None:
    """`is_empty` ran correctly and found nothing — the model sees it, the caller does not.

    Kept apart from `is_error` on purpose: a worker deciding whether its step produced
    anything must not be able to count "nothing matched" as a result (`ToolResponse`).
    """
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c1"),), usage_tokens=10),
            AssistantTurn(text="Nothing there.", usage_tokens=10),
        ]
    )
    tool = FakeTool(LOOKUP, [ToolResponse(content="Nothing matched.", is_empty=True)])

    result = await _loop(provider, [tool]).run(_context())

    assert result.kept == ()
    assert result.summary == "Nothing there."
    # It still reached the model, unflagged: a retry would find nothing either.
    answered = provider.send_calls[1].messages[-1].tool_results[0]
    assert (answered.is_error, answered.content) == (False, "Nothing matched.")


@pytest.mark.asyncio
async def test_loop_answers_an_unknown_tool_name_without_ending_the_step() -> None:
    """A hallucinated tool is the model's mistake to correct — it gets told the real ones."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call("run_sql", "c1"),), usage_tokens=10),
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c2"),), usage_tokens=10),
            AssistantTurn(text="Used the right tool.", usage_tokens=10),
        ]
    )
    tool = FakeTool(LOOKUP, [ToolResponse(content="ok")])

    result = await _loop(provider, [tool]).run(_context())

    assert result.summary == "Used the right tool."
    answer = provider.send_calls[1].messages[-1].tool_results[0]
    assert answer.is_error is True
    assert LOOKUP in answer.content


@pytest.mark.asyncio
async def test_loop_still_calling_tools_at_the_turn_cap_fails_with_its_label() -> None:
    """§10: the loop is bounded, and the label says which agent hit the bound."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c1"),), usage_tokens=10)
            for _ in range(2)
        ]
    )
    tool = FakeTool(LOOKUP, [ToolResponse(content="ok") for _ in range(2)])

    with pytest.raises(TaskFailure, match=r"Analysis for 'analyse_trend' .* after 2 turns"):
        await _loop(provider, [tool], max_turns=2).run(_context())


@pytest.mark.asyncio
async def test_loop_over_its_token_budget_fails_with_its_label() -> None:
    """The second bound: turns alone will not catch a model making expensive calls."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c1"),), usage_tokens=5_000),
            AssistantTurn(text="never reached", usage_tokens=10),
        ]
    )
    tool = FakeTool(LOOKUP, [ToolResponse(content="ok")])

    with pytest.raises(TaskFailure, match="Analysis for 'analyse_trend' spent its 100-token"):
        await _loop(provider, [tool], token_budget=100).run(_context())


@pytest.mark.asyncio
async def test_loop_with_a_truncated_reply_fails_with_its_label() -> None:
    """A reply cut off by the output limit has no tool call — the same shape as "done"."""
    provider = FakeProvider(
        turns=[AssistantTurn(text="I was about to sa", usage_tokens=10, stop_reason="max_tokens")]
    )

    with pytest.raises(TaskFailure, match=r"Analysis for .* cut off by the model's output limit"):
        await _loop(provider, [FakeTool(LOOKUP, [])]).run(_context())


@pytest.mark.asyncio
async def test_loop_bound_failure_counts_what_it_had_kept_per_tool() -> None:
    """§8: the message has to distinguish "raise the cap" from "debug the agent"."""
    provider = FakeProvider(
        turns=[
            AssistantTurn(
                text="",
                tool_calls=(_call(LOOKUP, "c1"), _call(LOOKUP, "c2"), _call(OTHER, "c3")),
                usage_tokens=10,
            ),
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c4"),), usage_tokens=10),
        ]
    )
    tools = [
        FakeTool(LOOKUP, [ToolResponse(content="ok") for _ in range(3)]),
        FakeTool(OTHER, [ToolResponse(content="ok")]),
    ]

    with pytest.raises(TaskFailure, match=r"kept results from lookup x3, other x1, which are lost"):
        await _loop(provider, tools, max_turns=2).run(_context())


@pytest.mark.asyncio
async def test_loop_bound_failure_with_nothing_kept_says_so() -> None:
    """The other half of the same message: an empty hand is itself the diagnosis."""
    provider = FakeProvider(
        turns=[AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c1"),), usage_tokens=10)]
    )
    tool = FakeTool(LOOKUP, [ToolResponse(content="broken", is_error=True)])

    with pytest.raises(TaskFailure, match=r"It had kept nothing at that point\."):
        await _loop(provider, [tool], max_turns=1).run(_context())


@pytest.mark.asyncio
async def test_loop_publishes_a_degraded_tool_as_a_lifecycle_event() -> None:
    """A tool that fell back is news for the operator, not a reason to fail the step.

    Only the loop can see it: the engine publishes the step's transitions, and this
    happens partway through one.
    """
    provider = FakeProvider(
        turns=[
            AssistantTurn(text="", tool_calls=(_call(LOOKUP, "c1"),), usage_tokens=10),
            AssistantTurn(text="Answered anyway.", usage_tokens=10),
        ]
    )
    tool = FakeTool(LOOKUP, [ToolResponse(content="a note", warning="Live search was down.")])
    broker: Broker[TaskEvent] = Broker()

    async with broker.subscribe() as queue:
        result = await _loop(provider, [tool], broker).run(_context())
        published = [queue.get_nowait() for _ in range(queue.qsize())]

    assert result.summary == "Answered anyway."
    warnings = [event for event in published if event.kind is EventKind.SUBTASK_WARNING]
    assert [(event.subtask_id, event.message) for event in warnings] == [
        ("analyse_trend", "Live search was down.")
    ]


@pytest.mark.asyncio
async def test_loop_propagates_cancellation() -> None:
    """§10: a cancelled run unwinds through the loop, never swallowed."""
    provider = FakeProvider(turns=[AssistantTurn(text="unreached")], blocker=asyncio.Event())
    task = asyncio.create_task(_loop(provider, [FakeTool(LOOKUP, [])]).run(_context()))
    await asyncio.sleep(0)  # let it reach the blocked provider call
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_loop_without_tools_is_a_wiring_bug() -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        _loop(FakeProvider(), [])


@pytest.mark.parametrize("bound", ["max_turns", "token_budget"])
def test_loop_with_a_non_positive_bound_is_a_wiring_bug(bound: str) -> None:
    """Both bounds are checked at construction, like the engine's — a wiring bug (§10)."""
    with pytest.raises(ValueError, match=f"{bound} must be at least 1, got 0"):
        _loop(FakeProvider(), [FakeTool(LOOKUP, [])], None, **{bound: 0})
