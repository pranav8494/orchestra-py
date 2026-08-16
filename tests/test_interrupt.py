"""The mid-run pause: what a message does to the plan, and what it must never touch (#12).

The engine-level tests here are the ticket's own claims — a pause waits for the step in
flight, a replan leaves finished work alone — so they run the real engine and the real
handler against scripted doubles rather than asserting on the handler in isolation.
"""

import asyncio

import pytest

from conftest import FakeProvider, ScriptedChat, ScriptedWorker, dispatches, wait_until
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.interrupt import (
    UNUSABLE_REPLY,
    InterruptAction,
    InterruptDraft,
    InterruptHandler,
)
from orchestra.agents.planner import SubtaskDraft
from orchestra.core.errors import ProviderError, TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import (
    AgentRole,
    EventKind,
    Plan,
    Subtask,
    SubtaskStatus,
    TaskEvent,
    TaskState,
)

CHART_STEP = "chart_revenue"


def _plan() -> Plan:
    """The demo's three-step plan: fetch, analyse, chart."""
    return Plan(
        subtasks=[
            Subtask(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Load the quarters."),
            Subtask(
                id="analyse",
                role=AgentRole.ANALYTICS,
                instruction="Compute the revenue trend.",
                inputs=["fetch"],
                depends_on=["fetch"],
            ),
            Subtask(
                id=CHART_STEP,
                role=AgentRole.VISUALIZATION,
                instruction="Draw a line chart of the trend.",
                inputs=["analyse"],
                depends_on=["analyse"],
            ),
        ]
    )


def _midway_state() -> TaskState:
    """The ledger as it stands when the chart step is the only one left."""
    state = TaskState(user_request="Chart the last three quarters.", plan=_plan())
    assert state.plan is not None
    for subtask in state.plan.subtasks[:2]:
        subtask.status = SubtaskStatus.DONE
        subtask.output_pointer = f"artifact:{subtask.id}.json"
        state.artifacts[subtask.id] = f"artifact:{subtask.id}.json"
    return state


def _bar_chart_draft() -> InterruptDraft:
    """A replan that swaps the chart step for a bar chart and touches nothing else."""
    return InterruptDraft(
        action=InterruptAction.REPLAN,
        reply="I'll draw it as a bar chart instead.",
        subtasks=[
            SubtaskDraft(
                id="chart_revenue_bar",
                role=AgentRole.VISUALIZATION,
                instruction="Draw a bar chart of the trend.",
                inputs=["analyse"],
                depends_on=["analyse"],
            )
        ],
    )


def _handler(provider: FakeProvider, chat: ScriptedChat) -> InterruptHandler:
    return InterruptHandler(provider, chat=chat, broker=Broker(), retrievable_data="- quarters")


# --------------------------------------------------------------------------------------
# The four actions
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replan_replaces_only_the_unfinished_steps() -> None:
    """The ticket's fourth criterion: completed steps and their artifacts survive intact."""
    state = _midway_state()
    before = dict(state.artifacts)
    chat = ScriptedChat(messages=["make it a bar chart instead"])

    await _handler(FakeProvider(responses=[_bar_chart_draft()]), chat).handle(state)

    assert state.plan is not None
    assert [subtask.id for subtask in state.plan.subtasks] == [
        "fetch",
        "analyse",
        "chart_revenue_bar",
    ]
    assert [subtask.status for subtask in state.plan.subtasks[:2]] == [
        SubtaskStatus.DONE,
        SubtaskStatus.DONE,
    ]
    assert state.artifacts == before  # nothing already produced was dropped
    assert state.plan.subtasks[-1].status is SubtaskStatus.PENDING


@pytest.mark.asyncio
async def test_continue_leaves_the_plan_exactly_as_it_was() -> None:
    """Nothing to change: the run resumes on the plan it was already executing."""
    state = _midway_state()
    plan = state.plan
    chat = ScriptedChat(messages=["just checking in"])
    provider = FakeProvider(
        responses=[InterruptDraft(action=InterruptAction.CONTINUE, reply="All on track.")]
    )

    await _handler(provider, chat).handle(state)

    assert state.plan is plan
    assert state.events == []  # no new plan to announce
    assert chat.said == ["All on track."]


@pytest.mark.asyncio
async def test_restart_step_resets_the_step_and_everything_downstream() -> None:
    """Redoing a step invalidates whatever was computed from its old output."""
    state = _midway_state()
    provider = FakeProvider(
        responses=[
            InterruptDraft(
                action=InterruptAction.RESTART_STEP, reply="Reloading the data.", restart="fetch"
            )
        ]
    )

    await _handler(provider, ScriptedChat(messages=["the data was stale"])).handle(state)

    assert state.plan is not None
    assert all(subtask.status is SubtaskStatus.PENDING for subtask in state.plan.subtasks)
    assert all(subtask.output_pointer is None for subtask in state.plan.subtasks)
    assert state.artifacts == {}


@pytest.mark.asyncio
async def test_clarify_keeps_the_chat_open_and_the_plan_untouched() -> None:
    """The ticket's third criterion: `clarify` answers and stays; the next message decides."""
    state = _midway_state()
    chat = ScriptedChat(messages=["what will the chart look like?", "make it a bar chart instead"])
    provider = FakeProvider(
        responses=[
            InterruptDraft(
                action=InterruptAction.CLARIFY, reply="A line chart of revenue by quarter."
            ),
            _bar_chart_draft(),
        ]
    )

    await _handler(provider, chat).handle(state)

    assert chat.sessions == 1  # both turns happened inside one pause
    assert chat.said == [
        "A line chart of revenue by quarter.",
        "I'll draw it as a bar chart instead.",
    ]
    assert state.plan is not None
    assert state.plan.subtasks[-1].id == "chart_revenue_bar"


@pytest.mark.asyncio
async def test_a_reply_carries_the_whole_conversation_forward() -> None:
    """Context is kept for the pause: the second call sees the situation, both messages and
    the first reply."""
    chat = ScriptedChat(messages=["what will the chart look like?", "fine, carry on"])
    provider = FakeProvider(
        responses=[
            InterruptDraft(action=InterruptAction.CLARIFY, reply="A line chart."),
            InterruptDraft(action=InterruptAction.CONTINUE, reply="Carrying on."),
        ]
    )

    await _handler(provider, chat).handle(_midway_state())

    situation, first, reply, second = provider.calls[1].messages
    assert CHART_STEP in situation.content  # the plan as it stood when they interrupted
    assert first.content == "what will the chart look like?"
    assert reply.content == "A line chart."
    assert second.content == "fine, carry on"


@pytest.mark.asyncio
async def test_resuming_without_a_message_asks_the_model_nothing() -> None:
    """Enter at the prompt is a resume, not an empty turn."""
    state = _midway_state()
    provider = FakeProvider()

    await _handler(provider, ScriptedChat(messages=[])).handle(state)

    assert provider.calls == []
    assert state.plan is not None
    assert state.plan.subtasks[-1].id == CHART_STEP


# --------------------------------------------------------------------------------------
# Rejections and failures
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_replan_reusing_a_completed_id_is_rejected_and_costs_only_the_turn() -> None:
    """A finished step's id may not be reassigned — and a reply nothing can be made of
    leaves the plan alone and the user still talking."""
    state = _midway_state()
    plan = state.plan
    collision = InterruptDraft(
        action=InterruptAction.REPLAN,
        reply="Redoing the fetch.",
        subtasks=[
            SubtaskDraft(id="fetch", role=AgentRole.DATA_RETRIEVAL, instruction="Load it again.")
        ],
    )
    chat = ScriptedChat(messages=["redo everything"])
    provider = FakeProvider(responses=[collision, collision, collision])

    await _handler(provider, chat).handle(state)

    assert len(provider.calls) == 3  # the rejection was fed back twice before giving up
    assert chat.said == [UNUSABLE_REPLY]
    assert state.plan is plan
    assert state.artifacts  # the completed work is still registered


@pytest.mark.asyncio
async def test_restart_step_naming_an_unknown_step_is_rejected() -> None:
    """`restart` has to name a step in the plan, or the pause would silently do nothing."""
    state = _midway_state()
    unknown = InterruptDraft(
        action=InterruptAction.RESTART_STEP, reply="Redoing it.", restart="nonexistent"
    )
    chat = ScriptedChat(messages=["redo the last one"])

    await _handler(FakeProvider(responses=[unknown] * 3), chat).handle(state)

    assert chat.said == [UNUSABLE_REPLY]
    assert state.artifacts  # nothing was reset


@pytest.mark.asyncio
async def test_a_provider_outage_during_a_pause_propagates() -> None:
    """A transport failure is not a rejection to feed back — it leaves `handle` (§8)."""
    provider = FakeProvider(responses=[ProviderError("rate limited")])

    with pytest.raises(ProviderError):
        await _handler(provider, ScriptedChat(messages=["change the chart"])).handle(
            _midway_state()
        )


@pytest.mark.asyncio
async def test_a_pause_cancelled_mid_conversation_propagates_and_leaves_the_plan_alone() -> None:
    """§10: cancellation is re-raised, and a half-finished pause changes nothing."""
    state = _midway_state()
    plan = state.plan
    chat = ScriptedChat(messages=["change the chart"], blocker=asyncio.Event())
    pause = asyncio.create_task(_handler(FakeProvider(), chat).handle(state))
    await wait_until(lambda: chat.sessions == 1, what="the pause to open")

    pause.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pause

    assert state.plan is plan
    assert state.events == []


# --------------------------------------------------------------------------------------
# The event stream
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_replan_publishes_the_plan_that_is_now_running() -> None:
    """The dashboard draws pending rows from `plan_created` and nothing else, so a reshaped
    plan has to arrive as one (§6, #11)."""
    broker: Broker[TaskEvent] = Broker()
    chat = ScriptedChat(messages=["make it a bar chart instead"])
    handler = InterruptHandler(
        FakeProvider(responses=[_bar_chart_draft()]), chat=chat, broker=broker
    )

    async with broker.subscribe() as queue:
        await handler.handle(_midway_state())
        event = queue.get_nowait()

    assert event.kind is EventKind.PLAN_CREATED
    assert event.plan is not None
    assert [subtask.id for subtask in event.plan.subtasks] == [
        "fetch",
        "analyse",
        "chart_revenue_bar",
    ]
    # Statuses ride along, so the redrawn table still shows the finished steps as done.
    assert event.plan.subtasks[0].status is SubtaskStatus.DONE


# --------------------------------------------------------------------------------------
# Through the engine
# --------------------------------------------------------------------------------------


def _engine(chat: ScriptedChat, provider: FakeProvider, worker: ScriptedWorker) -> ExecutionEngine:
    broker: Broker[TaskEvent] = Broker()
    return ExecutionEngine(
        workers=dict.fromkeys(AgentRole, worker),
        broker=broker,
        max_concurrency=2,
        interrupts=InterruptHandler(provider, chat=chat, broker=broker),
    )


@pytest.mark.asyncio
async def test_the_engine_pauses_between_steps_and_never_during_one() -> None:
    """The ticket's first criterion. The key is pressed while `fetch` is running; the chat
    opens once it has finished and before the next step is dispatched, so the orchestrator
    reshapes a settled ledger rather than one being written to."""
    state = TaskState(user_request="Chart the quarters.", plan=_plan())
    gate = asyncio.Event()
    worker = ScriptedWorker(gate=gate, gate_ids=frozenset({"fetch"}))
    at_pause: list[tuple[int, int]] = []
    chat = ScriptedChat(
        messages=[],
        armed=lambda: worker.contexts != [],  # the key lands once a step is under way
        on_session=lambda: at_pause.append((worker.running, len(worker.contexts))),
    )

    run = asyncio.create_task(_engine(chat, FakeProvider(), worker).run(state))
    await wait_until(lambda: worker.running == 1, what="the first step to start")

    assert chat.sessions == 0  # asked for, and not honoured while `fetch` holds the run
    gate.set()
    await run

    assert at_pause == [(0, 1)]  # nothing running, and only `fetch` had been dispatched
    assert [context.subtask.id for context in worker.contexts] == ["fetch", "analyse", CHART_STEP]


@pytest.mark.asyncio
async def test_the_engine_runs_the_replanned_step_and_not_the_replaced_one() -> None:
    """The demo scenario end to end: "what will the chart look like?" -> answer -> "make it
    a bar chart instead" changes the visualization step and nothing else."""
    state = TaskState(user_request="Chart the last three quarters.", plan=_plan())
    worker = ScriptedWorker()
    chat = ScriptedChat(
        messages=["what will the chart look like?", "make it a bar chart instead"],
        # Interrupted with the chart step still to come, as the demo does.
        armed=lambda: set(state.artifacts) == {"fetch", "analyse"},
    )
    provider = FakeProvider(
        responses=[
            InterruptDraft(
                action=InterruptAction.CLARIFY, reply="A line chart of revenue by quarter."
            ),
            _bar_chart_draft(),
        ]
    )

    await _engine(chat, provider, worker).run(state)

    ran = [context.subtask.id for context in worker.contexts]
    assert CHART_STEP not in ran  # the replaced step never ran
    assert "chart_revenue_bar" in ran
    assert ran.count("fetch") == 1 and ran.count("analyse") == 1  # neither was redone
    # The new step consumes the artifact the analytics step had already produced, rather
    # than a rerun of it.
    replanned = next(
        context for context in worker.contexts if context.subtask.id == "chart_revenue_bar"
    )
    assert replanned.inputs == {"analyse": "artifact:analyse.txt"}
    assert state.artifacts["analyse"] == "artifact:analyse.txt"


@pytest.mark.asyncio
async def test_a_restarted_step_gets_its_attempts_afresh() -> None:
    """A step the user sent back is not spending the budget its earlier failure used.

    `fetch` always fails, so with two attempts each it is dispatched twice, restarted, and
    dispatched twice more. Inheriting the spent budget would give it one dispatch, not two.
    The run's step cap still bounds the total (§10).
    """
    state = TaskState(user_request="Chart the quarters.", plan=_plan())
    worker = ScriptedWorker(fail_ids=frozenset({"fetch"}))
    chat = ScriptedChat(
        messages=["reload the data"],
        armed=lambda: dispatches(worker, "fetch") == 2,  # after its attempts ran out
    )
    provider = FakeProvider(
        responses=[
            InterruptDraft(action=InterruptAction.RESTART_STEP, reply="Reloading.", restart="fetch")
        ]
    )
    engine = ExecutionEngine(
        workers=dict.fromkeys(AgentRole, worker),
        broker=Broker(),
        subtask_attempts=2,
        interrupts=InterruptHandler(provider, chat=chat, broker=Broker()),
    )

    await engine.run(state)

    assert state.plan is not None
    assert dispatches(worker, "fetch") == 4
    assert state.plan.subtasks[0].status is SubtaskStatus.FAILED


@pytest.mark.asyncio
async def test_a_pause_that_changed_nothing_leaves_the_attempt_counters_alone() -> None:
    """The other half of the rule above: only a step the user sent back is given a fresh
    budget. `fetch` fails once, the user pauses and continues, and it still has exactly the
    one attempt it had left."""
    state = TaskState(user_request="Chart the quarters.", plan=_plan())
    worker = ScriptedWorker(fail_ids=frozenset({"fetch"}))
    chat = ScriptedChat(
        messages=["how is it going?"], armed=lambda: dispatches(worker, "fetch") == 1
    )
    provider = FakeProvider(
        responses=[InterruptDraft(action=InterruptAction.CONTINUE, reply="First step is retrying.")]
    )
    engine = ExecutionEngine(
        workers=dict.fromkeys(AgentRole, worker),
        broker=Broker(),
        subtask_attempts=2,
        interrupts=InterruptHandler(provider, chat=chat, broker=Broker()),
    )

    await engine.run(state)

    assert chat.sessions == 1
    assert dispatches(worker, "fetch") == 2


@pytest.mark.asyncio
async def test_no_pause_is_opened_once_the_step_cap_has_ended_the_run() -> None:
    """A conversation cannot lift the run's work budget, so offering one would be a chat
    whose replan nothing would dispatch (§10)."""
    state = TaskState(
        user_request="Fetch both.",
        plan=Plan(
            subtasks=[
                Subtask(id="a", role=AgentRole.DATA_RETRIEVAL, instruction="Load one."),
                Subtask(id="b", role=AgentRole.DATA_RETRIEVAL, instruction="Load the other."),
            ]
        ),
    )
    worker = ScriptedWorker()
    chat = ScriptedChat(messages=["replan it"], armed=lambda: worker.contexts != [])
    engine = ExecutionEngine(
        workers=dict.fromkeys(AgentRole, worker),
        broker=Broker(),
        max_concurrency=1,
        step_cap=1,
        interrupts=InterruptHandler(FakeProvider(), chat=chat, broker=Broker()),
    )

    with pytest.raises(TaskFailure):
        await engine.run(state)

    assert chat.sessions == 0
