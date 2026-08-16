"""The orchestrator: one request in, one validated `Plan` in `TaskState` out.

**Why a draft model.** The draft is the trust boundary, `Plan` the ledger entry, and the
conversion between them is where model output is validated (§7). Engine-owned fields
(`status`, `output_pointer`) stay out of the schema so the model cannot declare a step
already done.

**Why one retry.** Structured output guarantees JSON *shape*; the retry is for what a
schema cannot say — unique ids, known `depends_on`, acyclic graph. One reformat attempt
with the rejection fed back, then the run fails. General retry policy is #9's.
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.core.errors import TaskFailure
from orchestra.core.state import (
    AgentRole,
    EventKind,
    Plan,
    Subtask,
    TaskEvent,
    TaskState,
)
from orchestra.prompts import PLANNER_REFORMAT_INSTRUCTION, PLANNER_SYSTEM_PROMPT
from orchestra.providers.base import MessageRole, Provider, ProviderMessage


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


class PlanDraft(BaseModel):
    """The plan as the model writes it. Passed to the provider as `output_format`."""

    model_config = ConfigDict(extra="forbid")

    subtasks: list[SubtaskDraft] = Field(
        min_length=1, description="The steps to run, in no particular order."
    )


class Planner:
    """Turns a request into a plan. One per run, built in `app.py` with its provider."""

    def __init__(self, provider: Provider) -> None:
        """Store the provider. Nothing is read from config or the environment here (§6)."""
        self._provider = provider

    async def create_plan(self, state: TaskState) -> Plan:
        """Plan `state.user_request`, set `state.plan`, append `plan_created`, return it.

        Raises:
            TaskFailure: an unusable plan twice. Exit 5, not 4 — the provider answered
                both times, so "retry the provider" is the wrong advice (§8, §10).
            ProviderError: the provider failed; passed through from the adapter.
            asyncio.CancelledError: propagated, never swallowed (§10).
        """
        messages = [ProviderMessage(role=MessageRole.USER, content=state.user_request)]

        plan, rejection = await self._draft_plan(messages)
        if plan is None:
            messages.append(
                ProviderMessage(
                    role=MessageRole.USER,
                    content=f"{PLANNER_REFORMAT_INSTRUCTION}\n\n{rejection}",
                )
            )
            plan, rejection = await self._draft_plan(messages)
        if plan is None:
            # Not a loop: a model that failed the same schema twice with the error in
            # front of it will not succeed on the third try, and the user is waiting.
            raise TaskFailure(f"The planner returned an unusable plan twice. Last: {rejection}")

        state.plan = plan
        state.events.append(
            TaskEvent(
                kind=EventKind.PLAN_CREATED,
                message=f"Planned {len(plan.subtasks)} subtasks",
            )
        )
        return plan

    async def _draft_plan(self, messages: list[ProviderMessage]) -> tuple[Plan | None, str]:
        """Request one plan and validate it.

        Returns:
            The plan and an empty string, or `None` and the reason to feed back.
        """
        draft = await self._provider.parse_structured(
            system=PLANNER_SYSTEM_PROMPT,
            messages=messages,
            output_format=PlanDraft,
        )
        if draft is None:
            # A refusal and off-schema JSON are indistinguishable at the adapter, so the
            # feedback has to fit either.
            return None, (
                "No usable plan was returned. Reply with a plan matching the schema "
                "exactly, and nothing else."
            )
        try:
            return _to_plan(draft), ""
        except (ValidationError, _RejectedDraftError) as exc:
            # The rejected draft goes back with the reason: the model cannot see its own
            # previous message as data, and "fix this" needs a "this".
            return None, f"{exc}\n\nThe rejected plan was:\n{draft.model_dump_json(indent=2)}"


class _RejectedDraftError(ValueError):
    """The draft is well-formed JSON but not a runnable plan.

    Handled identically to the `ValidationError` `Plan` raises; separate only because
    `Plan` does not check `inputs` — see `_check_inputs`.
    """


def _check_inputs(draft: PlanDraft) -> None:
    """Reject data edges that are missing or unordered.

    `Plan` validates `depends_on` but not `inputs`, whose semantics are unsettled in
    `core/` (#4) — so the rule is enforced here: an input names a subtask in this plan,
    and consuming a step's output means depending on it. Without the second half the
    engine can start a consumer before its producer has written anything.

    Raises:
        _RejectedDraftError: an input names an unknown subtask, or one not depended on.
    """
    known = {subtask.id for subtask in draft.subtasks}
    for subtask in draft.subtasks:
        unknown = sorted(set(subtask.inputs) - known)
        if unknown:
            raise _RejectedDraftError(
                f"subtask {subtask.id!r} takes inputs from unknown steps: {unknown}"
            )
        unordered = sorted(set(subtask.inputs) - set(subtask.depends_on))
        if unordered:
            raise _RejectedDraftError(
                f"subtask {subtask.id!r} consumes {unordered} but does not depend on them; "
                "every id in `inputs` must also appear in `depends_on`"
            )


def _to_plan(draft: PlanDraft) -> Plan:
    """Convert model output into a ledger `Plan`, leaving engine-owned fields default.

    Raises:
        ValidationError: the draft is not a runnable DAG.
        _RejectedDraftError: the draft's data edges are unknown or unordered.
    """
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
