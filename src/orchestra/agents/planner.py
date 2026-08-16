"""The orchestrator: one request in, one validated `Plan` in `TaskState` out.

**Why a draft model.** The draft is the trust boundary, `Plan` the ledger entry, and the
conversion between them is where model output is validated (§7). Engine-owned fields
(`status`, `output_pointer`) stay out of the schema so the model cannot declare a step
already done.

**Why retries.** Structured output guarantees JSON *shape*; the retry is for what a
schema cannot say — unique ids, known `depends_on`, acyclic graph. `agents/structured.py`
runs up to three attempts with each rejection fed back, then the run fails.

**Why the planner owns clarification (#10).** A request missing a parameter is a planning
problem, so the ambiguity check and its guardrail are the planner's. One round per *run*,
not per call: a ledger that already carries clarifications never opens another, so a later
replan (#12) re-entering `create_plan` cannot start a second round. What follows validates
with `_plan_only`, which rejects `clarify` outright.
"""

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from orchestra.agents.structured import Rejected, parse_validated
from orchestra.core.errors import TaskFailure
from orchestra.core.question import MAX_QUESTIONS, ClarificationRequest, Question
from orchestra.core.state import (
    AgentRole,
    Clarification,
    EventKind,
    Plan,
    Subtask,
    TaskEvent,
    TaskState,
)
from orchestra.prompts import (
    PLANNER_AVAILABLE_DATA,
    PLANNER_CLARIFICATION_PREAMBLE,
    PLANNER_CLARIFY_SPENT,
    PLANNER_CLARIFY_UNANSWERED,
    PLANNER_NO_DATA_LISTED,
    PLANNER_REFORMAT_INSTRUCTION,
    PLANNER_SYSTEM_PROMPT,
)
from orchestra.providers.base import MessageRole, Provider, ProviderMessage
from orchestra.tools.base import BaseTool, ToolCall
from orchestra.tools.question import TOOL_NAME as ASK_USER_TOOL


class PlannerAction(StrEnum):
    """What the model decided to send back. A closed set, so an enum (§7)."""

    PLAN = "plan"
    CLARIFY = "clarify"


class SubtaskDraft(BaseModel):
    """One subtask as the model writes it — engine-owned fields absent by design."""

    # extra="forbid" so a field the model invents is a visible failure, not a silent drop.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Short unique lowercase slug for this step.")
    role: AgentRole = Field(description="The specialist that performs this step.")
    instruction: str = Field(
        min_length=1, description="One self-contained sentence stating what to produce."
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Ids of the subtasks whose output this step consumes.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of the subtasks that must finish before this one starts.",
    )


class PlannerDraft(BaseModel):
    """The planner's reply as the model writes it. Passed to the provider as `output_format`.

    Both outcomes in one schema because one call must be able to return either; which
    fields are required of which `action` is `_to_outcome`'s check, not the schema's.
    """

    model_config = ConfigDict(extra="forbid")

    action: PlannerAction = Field(
        description="`plan` to send subtasks, `clarify` to send questions instead."
    )
    subtasks: list[SubtaskDraft] = Field(
        default_factory=list, description="The steps to run, in no particular order. For `plan`."
    )
    questions: list[Question] = Field(
        default_factory=list,
        description=f"1 to {MAX_QUESTIONS} things to ask the user. For `clarify`.",
    )


class Planner:
    """Turns a request into a plan. One per run, built in `app.py` with its provider."""

    def __init__(
        self,
        provider: Provider,
        *,
        ask_tool: BaseTool | None = None,
        retrievable_data: str = "",
    ) -> None:
        """Store the provider and, when someone can answer, the tool that asks.

        `ask_tool` is `None` for a non-interactive run: the check still runs, but a
        `clarify` reply is rejected back to the model instead of blocking on a prompt
        nobody can answer. Nothing is read from config or the environment here (§6).

        `retrievable_data` is what the team can actually obtain, from
        `agents/toolsets.retrievable_data`. Without it the planner plans in a vacuum: it
        reads `data_retrieval` as able to fetch anything, and will both plan for and ask
        about data no tool holds (#10).
        """
        self._provider = provider
        self._ask_tool = ask_tool
        self._system = _system_prompt(retrievable_data)

    async def create_plan(self, state: TaskState) -> Plan:
        """Plan `state.user_request`, set `state.plan`, append `plan_created`, return it.

        At most one round of clarifying questions on the way; their answers are appended
        to `state.clarifications` and go back with the request.

        Raises:
            TaskFailure: no usable plan across every attempt. Exit 5, not 4 — the provider
                answered each time, so "retry the provider" is the wrong advice (§8, §10).
            ProviderError: the provider failed; passed through from the adapter.
            asyncio.CancelledError: propagated, never swallowed (§10).
        """
        ask_tool = self._ask_tool
        # The ledger is the run's record of its one round, so asking is unreachable once
        # it holds anything — whoever called, and however often.
        if ask_tool is not None and not state.clarifications:
            outcome = await self._request(state, _to_outcome)
            if isinstance(outcome, Plan):
                return self._commit(state, outcome)
            await self._ask(state, outcome, ask_tool)

        # Nobody could be asked, or the round is over. Which reason the model gets turns
        # on whether an answer was recorded, not on how we got here: a round everyone
        # declined leaves nothing to plan with, exactly as no round does.
        refusal = PLANNER_CLARIFY_SPENT if state.clarifications else PLANNER_CLARIFY_UNANSWERED
        return self._commit(state, await self._request(state, _plan_only(refusal)))

    async def _request[ResultT](
        self, state: TaskState, validate: Callable[[PlannerDraft], ResultT]
    ) -> ResultT:
        """One structured call, retried until `validate` accepts it (#9).

        Raises:
            TaskFailure: every attempt was rejected.
        """
        result, rejection = await parse_validated(
            provider=self._provider,
            system=self._system,
            messages=_messages(state),
            output_format=PlannerDraft,
            validate=validate,
            instruction=PLANNER_REFORMAT_INSTRUCTION,
        )
        if result is None:
            raise TaskFailure(f"The planner returned no usable plan. Last rejection: {rejection}")
        return result

    async def _ask(
        self, state: TaskState, request: ClarificationRequest, ask_tool: BaseTool
    ) -> None:
        """Put each question to the user and record the answers on the ledger.

        Through the tool rather than the `Asker` behind it, so a question the model asks
        mid-run and one the planner asks up front are answered by the same code (§1.5).
        """
        for index, question in enumerate(request.questions, start=1):
            response = await ask_tool.run(
                ToolCall(id=f"clarify-{index}", name=ASK_USER_TOOL, arguments=question.model_dump())
            )
            # A blank answer is a decline, an error is this call being malformed; neither
            # may be quoted back to the model as an answer.
            if response.is_error or response.is_empty:
                continue
            state.clarifications.append(
                Clarification(question=question.text, answer=response.content)
            )

    def _commit(self, state: TaskState, plan: Plan) -> Plan:
        """Write the plan to the ledger and record that it exists."""
        state.plan = plan
        state.events.append(
            TaskEvent(
                kind=EventKind.PLAN_CREATED,
                message=f"Planned {len(plan.subtasks)} subtasks",
            )
        )
        return plan


def _system_prompt(retrievable_data: str) -> str:
    """The planner's instructions with the run's data roster appended.

    Composed once, in `__init__`: the roster is fixed for the run, and §11 keeps runtime
    formatting out of `prompts/`. The system turn, not a user one — this is what the team
    can do, not something the user said.
    """
    listing = (
        f"{PLANNER_AVAILABLE_DATA}\n\n{retrievable_data}"
        if retrievable_data
        else (PLANNER_NO_DATA_LISTED)
    )
    return f"{PLANNER_SYSTEM_PROMPT}\n\n{listing}"


def _messages(state: TaskState) -> list[ProviderMessage]:
    """The conversation the planner sends: the request, then any answers it earned.

    The request stays its own user turn — untrusted text spliced into the instructions
    can rewrite them (§11) — and the answers are a second turn rather than an edit of it.
    """
    messages = [ProviderMessage(role=MessageRole.USER, content=state.user_request)]
    if state.clarifications:
        answers = "\n".join(
            f"Q: {answered.question}\nA: {answered.answer}" for answered in state.clarifications
        )
        messages.append(
            ProviderMessage(
                role=MessageRole.USER, content=f"{PLANNER_CLARIFICATION_PREAMBLE}\n\n{answers}"
            )
        )
    return messages


def _plan_only(refusal: str) -> Callable[[PlannerDraft], Plan]:
    """A validator that accepts a plan and rejects questions, saying why in `refusal`.

    The reason differs by round — nobody to ask, or asked already — and the model only
    corrects itself if it is told which (§9).
    """

    def validate(draft: PlannerDraft) -> Plan:
        if draft.action is PlannerAction.CLARIFY:
            raise Rejected(refusal)
        return _to_plan(draft)

    return validate


def _to_outcome(draft: PlannerDraft) -> Plan | ClarificationRequest:
    """Convert model output into whichever outcome its `action` declares.

    Raises:
        ValidationError: `clarify` with no questions, or more than `MAX_QUESTIONS`.
        Rejected: the draft does not carry what its own action needs.
    """
    if draft.action is PlannerAction.CLARIFY:
        if draft.subtasks:
            raise Rejected("action 'clarify' asks questions instead of planning; send no subtasks")
        return ClarificationRequest(questions=draft.questions)
    return _to_plan(draft)


def _check_inputs(draft: PlannerDraft) -> None:
    """Reject data edges that are missing or unordered.

    `Plan` validates `depends_on` but not `inputs`, whose semantics are unsettled in
    `core/` (#4) — so the rule is enforced here: an input names a subtask in this plan,
    and consuming a step's output means depending on it. Without the second half the
    engine can start a consumer before its producer has written anything.

    Raises:
        Rejected: an input names an unknown subtask, or one not depended on.
    """
    known = {subtask.id for subtask in draft.subtasks}
    for subtask in draft.subtasks:
        unknown = sorted(set(subtask.inputs) - known)
        if unknown:
            raise Rejected(f"subtask {subtask.id!r} takes inputs from unknown steps: {unknown}")
        unordered = sorted(set(subtask.inputs) - set(subtask.depends_on))
        if unordered:
            raise Rejected(
                f"subtask {subtask.id!r} consumes {unordered} but does not depend on them; "
                "every id in `inputs` must also appear in `depends_on`"
            )


def _to_plan(draft: PlannerDraft) -> Plan:
    """Convert model output into a ledger `Plan`, leaving engine-owned fields default.

    Raises:
        ValidationError: the draft is not a runnable DAG.
        Rejected: the draft has no subtasks, or its data edges are unknown or unordered.
    """
    if not draft.subtasks:
        raise Rejected("action 'plan' needs at least one subtask")
    if draft.questions:
        raise Rejected("questions belong to action 'clarify'; a plan carries none")
    _check_inputs(draft)
    return Plan(
        subtasks=[
            Subtask(
                id=subtask.id,
                role=subtask.role,
                instruction=subtask.instruction,
                inputs=list(subtask.inputs),
                depends_on=list(subtask.depends_on),
            )
            for subtask in draft.subtasks
        ]
    )
