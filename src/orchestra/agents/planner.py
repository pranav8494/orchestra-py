"""The orchestrator: one request in, one validated `Plan` in `TaskState` out.

**Why a draft model.** `PlanDraft`/`SubtaskDraft` are the schema the model fills in;
`core.state.Plan` is what the engine runs. They differ deliberately: `Subtask.status`
and `Subtask.output_pointer` are the engine's to write, so putting them in the schema
would invite the model to declare a step already done. The draft is the trust boundary,
`Plan` is the ledger entry, and the conversion between them is where the model's output
is validated (§7).

**Why one retry.** Structured output guarantees the *shape* of the JSON, so the retry
here is for what a JSON schema cannot say: that ids are unique, that `depends_on` names
a step in this plan, and that the graph is acyclic. One reformat attempt with the
rejection fed back, then the run fails. The general retry policy is #9's; this is not
its first draft.
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.core.errors import ProviderError
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
        """Store the provider. Nothing is read from config or the environment here (§6).

        Args:
            provider: the model provider to plan with.
        """
        self._provider = provider

    async def create_plan(self, state: TaskState) -> Plan:
        """Plan `state.user_request`, record it in `state`, and return it.

        Args:
            state: the run's ledger. On success `state.plan` is set and a
                `plan_created` event is appended to `state.events`.

        Returns:
            The validated plan, the same object as `state.plan`.

        Raises:
            ProviderError: the provider failed, or returned an unusable plan twice.
            asyncio.CancelledError: the caller cancelled the run; propagated, never
                swallowed (§10).
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
            # Deliberately not a loop: a model that has failed the same schema twice
            # with the error in front of it is not going to succeed on the third try,
            # and the user is waiting.
            raise ProviderError(f"The planner returned an unusable plan twice. Last: {rejection}")

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
            The plan and an empty string, or `None` and the reason to feed back. Both
            "no structured output" and "failed our validation" are the same thing to
            the caller — something to say to the model and try once more.
        """
        draft = await self._provider.parse_structured(
            system=PLANNER_SYSTEM_PROMPT,
            messages=messages,
            output_format=PlanDraft,
        )
        if draft is None:
            return None, "No plan was returned. Reply with the plan and nothing else."
        try:
            return _to_plan(draft), ""
        except ValidationError as exc:
            # `Plan` enforces unique ids, known dependencies and acyclicity — none of
            # which a JSON schema can express, so this is the retry's whole reason to
            # exist. The rejected draft goes back too: the model cannot see its own
            # previous message as data, and "fix this" needs a "this".
            return None, f"{exc}\n\nThe rejected plan was:\n{draft.model_dump_json(indent=2)}"


def _to_plan(draft: PlanDraft) -> Plan:
    """Convert model output into a ledger `Plan`, leaving engine-owned fields default.

    Raises:
        ValidationError: the draft is not a runnable DAG.
    """
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
