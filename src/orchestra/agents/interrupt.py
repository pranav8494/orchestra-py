"""The mid-run pause: the user talks to the orchestrator, and what is left changes (#12).

**Finished work is never redone.** A replan replaces only the subtasks that have not
completed; the completed ones stay in the plan with their status and pointers, so the
artifacts they produced survive the pause untouched.

**The decision is model output, so it is validated like any other.** The same
`agents/structured.parse_validated` path the planner uses, the same `SubtaskDraft`
becoming the same `Plan` through `subtasks_to_plan` — a replanned step is indistinguishable
from a planned one. A reply that survives no attempt costs the turn, not the run: the user
is told and asked again.

**One conversation per pause.** The transcript opens with the state of the run at the
moment of the interrupt and is dropped when the user resumes, because a later pause is a
different moment.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from orchestra.agents.planner import SubtaskDraft, data_roster, subtasks_to_plan
from orchestra.agents.structured import Rejected, parse_validated
from orchestra.core.events import Broker
from orchestra.core.interrupt import Chat
from orchestra.core.state import EventKind, Plan, Subtask, SubtaskStatus, TaskEvent, TaskState
from orchestra.prompts import (
    INTERRUPT_REFORMAT_INSTRUCTION,
    INTERRUPT_SITUATION,
    INTERRUPT_SYSTEM_PROMPT,
)
from orchestra.providers.base import MessageRole, Provider, ProviderMessage

# Said to the user when no attempt produced a usable decision. Here rather than in
# `prompts/`: it is read by a person, not by a model (§11).
UNUSABLE_REPLY = "Sorry - I could not act on that. Try again, or press Enter to resume."

_NO_STEPS = "- none"


class InterruptAction(StrEnum):
    """What the orchestrator decided to do about a message. A closed set, so an enum (§7)."""

    REPLAN = "replan"
    RESTART_STEP = "restart_step"
    CLARIFY = "clarify"
    CONTINUE = "continue"


class InterruptDraft(BaseModel):
    """The orchestrator's reply to one message, as the model writes it.

    All four actions in one schema because one call must be able to return any of them;
    which fields each needs is `_decide`'s check, not the schema's — as with `PlannerDraft`.
    """

    # extra="forbid" so a field the model invents is a visible failure, not a silent drop.
    model_config = ConfigDict(extra="forbid")

    action: InterruptAction = Field(description="What to do about this message.")
    reply: str = Field(
        min_length=1, description="One or two sentences addressed to the user. Always required."
    )
    subtasks: list[SubtaskDraft] = Field(
        default_factory=list,
        description="For `replan`: the full replacement for every step that has not finished.",
    )
    restart: str = Field(
        default="", description="For `restart_step`: the id of the one step to run again."
    )


@dataclass(frozen=True, slots=True)
class Decision:
    """A validated reply — an internal value object, not a trust boundary (§7).

    `plan` is already merged with the completed steps, so applying it is one assignment.
    """

    action: InterruptAction
    reply: str
    plan: Plan | None = None  # replan only
    restart: str = ""  # restart_step only


class InterruptHandler:
    """Hosts the pause. One per run, built in `app.py` with the chat that reads the terminal.

    Implements `core.interrupt.Interrupter`, which is all the engine sees.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        chat: Chat,
        broker: Broker[TaskEvent],
        retrievable_data: str = "",
    ) -> None:
        """Wire the handler.

        Args:
            chat: the user's end of the conversation, from `cli/chat.py`.
            broker: where the reshaped plan is published, so the dashboard redraws its
                rows from the plan that is now running.
            retrievable_data: what the team can obtain, as the planner is told it — a
                replan that invented a source would plan work with nowhere to get its data.
        """
        self._provider = provider
        self._chat = chat
        self._broker = broker
        self._system = f"{INTERRUPT_SYSTEM_PROMPT}\n\n{data_roster(retrievable_data)}"

    def pending(self) -> bool:
        """Has the user asked to interrupt? Consuming — see `core.interrupt.Chat`."""
        return self._chat.requested()

    async def handle(self, state: TaskState) -> frozenset[str]:
        """Run one pause to its end and apply what the user settled on.

        Returns the ids sent back to be run again, for the engine's attempt counters.

        The conversation is bounded by the person at the prompt, not by a cap: every lap
        needs a line from them, so this is not the model-driven loop §10 requires a
        ceiling on.

        Raises:
            ProviderError: the provider failed; passed through from the adapter.
            asyncio.CancelledError: propagated, never swallowed (§10).
        """
        decision = await self._converse(state)
        if decision is None:
            return frozenset()
        return await self._apply(state, decision)

    async def _converse(self, state: TaskState) -> Decision | None:
        """Talk until the user resumes or the orchestrator settles on something to do.

        `clarify` keeps the conversation open; every other action ends it and takes effect
        on resume. Returns `None` when the user resumed without settling on anything.
        """
        turns = [ProviderMessage(role=MessageRole.USER, content=_situation(state))]
        with self._chat.session():
            while message := await self._chat.next_message():
                turns.append(ProviderMessage(role=MessageRole.USER, content=message))
                decision, _rejection = await parse_validated(
                    provider=self._provider,
                    system=self._system,
                    messages=turns,
                    output_format=InterruptDraft,
                    validate=_decide(state),
                    instruction=INTERRUPT_REFORMAT_INSTRUCTION,
                )
                if decision is None:
                    # The rejection names the schema and quotes the draft; it is feedback
                    # for the model, not something to put in front of the user.
                    self._chat.say(UNUSABLE_REPLY)
                    continue
                self._chat.say(decision.reply)
                turns.append(
                    ProviderMessage(role=MessageRole.ASSISTANT, content=decision.reply),
                )
                if decision.action is not InterruptAction.CLARIFY:
                    return decision
        return None

    async def _apply(self, state: TaskState, decision: Decision) -> frozenset[str]:
        """Commit `decision` to the ledger, publish the plan now running, and name what it
        sent back to be rerun.

        Called after the conversation has closed, never during it: publishing while the
        terminal belongs to a prompt would redraw the live region over what the user is
        typing.
        """
        plan = state.plan
        if plan is None or decision.action is InterruptAction.CONTINUE:
            return frozenset()

        stale: frozenset[str] = frozenset()
        if decision.plan is not None:
            kept = sum(1 for subtask in plan.subtasks if subtask.status is SubtaskStatus.DONE)
            plan = decision.plan
            state.plan = plan
            # Everything not carried over is a step about to run for the first time — and a
            # replacement may legally reuse the id of the unfinished step it replaces, so
            # the engine has to be told to drop that id's attempt count and its last error.
            # Without this the reconciliation reports a failure the new step never had.
            stale = frozenset(
                subtask.id for subtask in plan.subtasks if subtask.status is not SubtaskStatus.DONE
            )
            message = f"Replanned: {len(plan.subtasks)} subtasks, {kept} already done"
        else:
            stale = _reset(state, plan, decision.restart)
            message = f"Rerunning {len(stale)} subtasks: {', '.join(sorted(stale))}"
        # Named, not inferred: only the pause knows which ids are starting over, and the
        # engine's counters are keyed by id, not by object.

        # `plan_created`, not a kind of its own: a subscriber cannot draw rows from
        # transitions it has not seen, and this is a new set of rows. Deep-copied for the
        # reason the engine gives — the loop mutates `Subtask.status` in place afterwards.
        event = TaskEvent(
            kind=EventKind.PLAN_CREATED, message=message, plan=plan.model_copy(deep=True)
        )
        state.events.append(event)
        await self._broker.publish_lifecycle(event)
        return stale


def _reset(state: TaskState, plan: Plan, subtask_id: str) -> frozenset[str]:
    """Send `subtask_id` and everything downstream of it back to pending, and say which.

    Downstream too: a step's output is its dependents' input, so redoing it leaves
    anything computed from the old output describing data that no longer exists. The
    artifact registration goes with it, or `state_slice` would hand a worker the pointer
    the rerun is replacing.
    """
    stale = {subtask_id}
    # Plan order is not topological, so close over the edges rather than assuming one pass
    # reaches every dependent. Plans are a handful of steps; this is not a hot loop.
    grew = True
    while grew:
        grew = False
        for subtask in plan.subtasks:
            if subtask.id not in stale and stale & set(subtask.depends_on):
                stale.add(subtask.id)
                grew = True

    for subtask in plan.subtasks:
        if subtask.id in stale:
            subtask.status = SubtaskStatus.PENDING
            subtask.output_pointer = None
            state.artifacts.pop(subtask.id, None)
    return frozenset(stale)


def _decide(state: TaskState) -> Callable[[InterruptDraft], Decision]:
    """A validator that turns a draft into a `Decision` against the plan as it stands.

    A closure rather than a free function because every rule here is about *this* ledger:
    which ids exist, and which steps are finished.
    """
    subtasks = [] if state.plan is None else state.plan.subtasks
    known = {subtask.id for subtask in subtasks}
    kept = [subtask for subtask in subtasks if subtask.status is SubtaskStatus.DONE]

    def validate(draft: InterruptDraft) -> Decision:
        if draft.action is not InterruptAction.REPLAN and draft.subtasks:
            raise Rejected(f"action {draft.action.value!r} sends no subtasks")
        if draft.action is not InterruptAction.RESTART_STEP and draft.restart:
            raise Rejected(f"action {draft.action.value!r} names no step to restart")

        if draft.action is InterruptAction.REPLAN:
            if not draft.subtasks:
                raise Rejected(
                    "action 'replan' needs the subtasks that replace the unfinished steps"
                )
            plan = subtasks_to_plan(draft.subtasks, kept=kept)
            return Decision(action=draft.action, reply=draft.reply, plan=plan)

        if draft.action is InterruptAction.RESTART_STEP:
            if draft.restart not in known:
                raise Rejected(
                    f"`restart` must name a step in the plan; {sorted(known)} are the ids"
                )
            return Decision(action=draft.action, reply=draft.reply, restart=draft.restart)

        return Decision(action=draft.action, reply=draft.reply)

    return validate


def _situation(state: TaskState) -> str:
    """The run as it stands, for the orchestrator's first turn of the pause.

    Built here, not in `prompts/`, which holds no runtime formatting (§11). Pointers,
    never payloads — the ledger's own rule, and an artifact can be a whole dataset (§6).
    """
    subtasks = [] if state.plan is None else state.plan.subtasks
    done = [subtask for subtask in subtasks if subtask.status is SubtaskStatus.DONE]
    remaining = [subtask for subtask in subtasks if subtask.status is not SubtaskStatus.DONE]
    return "\n".join(
        [
            INTERRUPT_SITUATION,
            "",
            f"Original request: {state.user_request}",
            "",
            "Completed steps - their artifacts are kept, and they are never replanned:",
            *(_step_line(subtask) for subtask in done),
            *([] if done else [_NO_STEPS]),
            "",
            "Steps that have not finished - these are what `replan` replaces:",
            *(_step_line(subtask) for subtask in remaining),
            *([] if remaining else [_NO_STEPS]),
        ]
    )


def _step_line(subtask: Subtask) -> str:
    """One step for the situation block: who does what, and what it produced."""
    produced = f" -> {subtask.output_pointer}" if subtask.output_pointer else ""
    depends = f" after {subtask.depends_on}" if subtask.depends_on else ""
    return f"- {subtask.id} ({subtask.role.value}){depends}: {subtask.instruction}{produced}"
