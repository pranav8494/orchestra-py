"""The orchestrator: one request in, one validated `Plan` in `TaskState` out.

**Why a draft model.** The draft is the trust boundary, `Plan` the ledger entry, and the
conversion between them is where model output is validated (§7). Engine-owned fields
(`status`, `output_pointer`) stay out of the schema so the model cannot declare a step
already done.

**Why retries.** Structured output guarantees JSON *shape*; the retry is for what a
schema cannot say — unique ids, known `depends_on`, acyclic graph. `agents/structured.py`
runs up to three attempts with each rejection fed back, then the run fails.
"""

from pydantic import BaseModel, ConfigDict, Field

from orchestra.agents.structured import Rejected, parse_validated
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
            TaskFailure: no usable plan across every attempt. Exit 5, not 4 — the provider
                answered each time, so "retry the provider" is the wrong advice (§8, §10).
            ProviderError: the provider failed; passed through from the adapter.
            asyncio.CancelledError: propagated, never swallowed (§10).
        """
        plan, rejection = await parse_validated(
            provider=self._provider,
            system=PLANNER_SYSTEM_PROMPT,
            messages=[ProviderMessage(role=MessageRole.USER, content=state.user_request)],
            output_format=PlanDraft,
            validate=_to_plan,
            instruction=PLANNER_REFORMAT_INSTRUCTION,
        )
        if plan is None:
            raise TaskFailure(f"The planner returned no usable plan. Last rejection: {rejection}")

        state.plan = plan
        state.events.append(
            TaskEvent(
                kind=EventKind.PLAN_CREATED,
                message=f"Planned {len(plan.subtasks)} subtasks",
            )
        )
        return plan


def _check_inputs(draft: PlanDraft) -> None:
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


def _to_plan(draft: PlanDraft) -> Plan:
    """Convert model output into a ledger `Plan`, leaving engine-owned fields default.

    Raises:
        ValidationError: the draft is not a runnable DAG.
        Rejected: the draft's data edges are unknown or unordered — `Plan` checks
            `depends_on` but not `inputs`, so `_check_inputs` owns that half.
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
