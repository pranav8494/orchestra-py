"""Runs a bounded tool-use conversation for one subtask and hands back what the tools returned.

**Both bounds, always (§10).** `max_turns` catches a model that keeps calling tools;
`token_budget` catches one calling *expensive* ones, which a turn count will not see.
Exceeding either is a `TaskFailure` — one failed subtask, not a retry.

**Failures come back as data.** A tool error, and a tool name that does not exist, are
answered to the model so the next turn is its retry; only a bound unwinds the loop (§6).

**The transcript is resent whole every turn**, so the budget counts it once per lap and
bites sooner than a distinct-token count would. Deliberate: resent tokens are billed
tokens, and it is spend being bounded.

**A truncated reply is not a finished one.** A reply cut off mid-generation carries no
tool call, which is exactly what "the agent is done" looks like; unchecked, half a
sentence becomes the summary and the run reports success (§8).

Shared by every worker that runs tools (#5-#7), which differ in their prompt, their
toolset and what they do with the results — not in any of the above. What counts as
"produced nothing", and what artifact to write, stay with the worker: only it knows.
"""

from collections import Counter
from dataclasses import dataclass

from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import EventKind, SubtaskContext, TaskEvent
from orchestra.providers.base import (
    AssistantTurn,
    MessageRole,
    Provider,
    ProviderMessage,
    ToolResult,
)
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse

# How many model turns one subtask may take. Six is room for a query, a correction after
# a tool error, a second query, and a closing summary — more than the sample subtask
# needs and less than a loop.
DEFAULT_MAX_TURNS = 6

# Input plus output tokens across those turns. Sized so a subtask that somehow retries
# every turn still costs less than the run's other two agents together.
DEFAULT_TOKEN_BUDGET = 60_000

# `AssistantTurn.stop_reason` when the reply was cut off by the output limit.
TRUNCATED = "max_tokens"


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """One call the loop kept, paired with what it returned.

    Both halves, because the arguments are the provenance for the content: a worker
    recording a result without what was asked for stores an answer to a lost question.
    """

    call: ToolCall
    response: ToolResponse


@dataclass(frozen=True, slots=True)
class LoopResult:
    """What one bounded conversation produced: the closing text and the usable calls."""

    summary: str  # the model's last turn, the one that asked for no tools
    # Every call whose response was neither an error nor empty, in the order made. The
    # loop does not group them by tool — which names matter is the worker's business.
    kept: tuple[ToolOutcome, ...]


class ToolLoop:
    """The tool-use conversation a worker runs. Composed by workers, not subclassed (§7)."""

    def __init__(
        self,
        *,
        provider: Provider,
        broker: Broker[TaskEvent],
        tools: tuple[BaseTool, ...],
        system_prompt: str,
        label: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        """Take the wired services, the agent's prompt and toolset, and the loop's bounds.

        Args:
            provider: the model provider to run the tool-use conversation with.
            broker: the run's event stream, for warnings a tool raises mid-step. The
                engine publishes the step's *transitions*; only this loop can see a tool
                degrade partway through one, so it reports that itself.
            tools: the calling agent's toolset, from `agents/toolsets.py`.
            system_prompt: the calling agent's prompt, from `orchestra.prompts` (§11).
            label: the agent's name in a bound failure ("Retrieval", "Analysis"), so the
                operator can tell which agent hit the cap without reading a traceback.
            max_turns: how many model turns one subtask may take.
            token_budget: input plus output tokens one subtask may spend.

        Raises:
            ValueError: an empty toolset or a non-positive bound — a wiring bug, not a
                user-facing error, so it fails at construction like the engine's.
        """
        if not tools:
            raise ValueError("ToolLoop needs at least one tool")
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns}")
        if token_budget < 1:
            raise ValueError(f"token_budget must be at least 1, got {token_budget}")
        self._provider = provider
        self._broker = broker
        self._tools = {tool.info().name: tool for tool in tools}
        self._specs = [tool.info() for tool in tools]
        self._system_prompt = system_prompt
        self._label = label
        self._max_turns = max_turns
        self._token_budget = token_budget

    async def run(self, context: SubtaskContext) -> LoopResult:
        """Talk to the model until it stops calling tools, running what it asks for.

        Args:
            context: the worker's slice of the ledger, briefed to the model as a user
                turn — the untrusted text stays out of the system prompt (§11).

        Returns:
            The closing summary and every call worth keeping. Both may be empty: whether
            that means the step failed is the worker's judgement, not the loop's.

        Raises:
            TaskFailure: a bound was hit, or the reply was truncated.
            asyncio.CancelledError: propagated from the provider or a tool (§10).
        """
        messages = [ProviderMessage(role=MessageRole.USER, content=_briefing(context))]
        kept: list[ToolOutcome] = []
        summary = ""
        spent = 0

        for _ in range(self._max_turns):
            # Checked before the call, not after: the cost of a turn is only known once
            # it has been paid, so this bounds what the loop *starts*, and one turn may
            # carry the total past the budget. A ceiling on spend, not on the last bill.
            if spent >= self._token_budget:
                raise TaskFailure(
                    f"{self._label} for {context.subtask.id!r} spent its {self._token_budget}-token "
                    f"budget before finishing. {_gathered(kept)}"
                )
            turn = await self._provider.send(
                system=self._system_prompt, messages=messages, tools=self._specs
            )
            spent += turn.usage_tokens

            if turn.stop_reason == TRUNCATED:
                raise TaskFailure(
                    f"{self._label} for {context.subtask.id!r} was cut off by the model's output "
                    f"limit. {_gathered(kept)}"
                )

            if not turn.tool_calls:
                summary = turn.text
                break

            results = []
            for call in turn.tool_calls:
                response = await self._invoke(call)
                results.append(
                    ToolResult(
                        call_id=call.id, content=response.content, is_error=response.is_error
                    )
                )
                if response.warning:
                    await self._warn(context, response.warning)
                # `is_empty` and not just `is_error`: a lookup that matched nothing ran
                # correctly, but handing it back would let "nothing was found" pass a
                # worker's did-we-produce-anything check (§6, and `ToolResponse`).
                if response.is_error or response.is_empty:
                    continue
                kept.append(ToolOutcome(call=call, response=response))
            messages.extend(_exchange(turn, tuple(results)))
        else:
            raise TaskFailure(
                f"{self._label} for {context.subtask.id!r} was still calling tools after "
                f"{self._max_turns} turns. {_gathered(kept)}"
            )

        return LoopResult(summary=summary, kept=tuple(kept))

    async def _warn(self, context: SubtaskContext, warning: str) -> None:
        """Tell the run that this step degraded, without failing it.

        Must-deliver rather than lossy progress: a dropped frame of a progress stream
        costs a subscriber nothing, but this one changes what the operator believes the
        answer is made of, which is the same class of fact as a state transition (§6).
        Bounded inside the broker, so a wedged dashboard still cannot hang the run.

        Not recorded on the ledger, unlike the engine's own events: a worker sees only
        its slice and has no `TaskState` to append to (§6). The notice is durable
        anyway — it is inside the artifact the step writes.
        """
        await self._broker.publish_lifecycle(
            TaskEvent(
                kind=EventKind.SUBTASK_WARNING,
                subtask_id=context.subtask.id,
                message=warning,
            )
        )

    async def _invoke(self, call: ToolCall) -> ToolResponse:
        """Run one tool call, answering for a tool that does not exist.

        A hallucinated tool name is the model's mistake to correct, so it comes back as
        a readable error naming the real tools rather than ending the subtask (§6).
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResponse(
                content=f"There is no tool named {call.name!r}. "
                f"The tools available are: {', '.join(sorted(self._tools))}.",
                is_error=True,
            )
        return await tool.run(call)


def _gathered(kept: list[ToolOutcome]) -> str:
    """Say what the step had in hand when it stopped short.

    A bound was hit, so nothing is written and the work is lost. Naming it in the failure
    is the difference between "raise the cap" and "debug the agent" (§8). Counted by tool
    name because that is all the loop knows — a worker's own vocabulary for its results
    is not visible here, and the count is what says whether the agent was making progress.
    """
    if not kept:
        return "It had kept nothing at that point."
    counts = Counter(outcome.call.name for outcome in kept)  # insertion-ordered
    breakdown = ", ".join(f"{name} x{count}" for name, count in counts.items())
    return f"It had kept results from {breakdown}, which are lost with the step."


def _exchange(turn: AssistantTurn, results: tuple[ToolResult, ...]) -> list[ProviderMessage]:
    """The two messages one round of tool use adds to the conversation.

    The assistant's own turn has to be replayed verbatim: the API is stateless, and a
    `tool_result` whose `tool_use` is missing from the history is rejected outright.
    Verbatim means `raw_content` — the turn as the provider returned it, blocks this
    codebase never decodes included, because the model reasons before calling a tool and
    the API wants that reasoning back with the call (see `AssistantTurn.raw_content`).
    `content` and `tool_calls` ride along for the fakes, which have no raw turn to keep.
    """
    return [
        ProviderMessage(
            role=MessageRole.ASSISTANT,
            content=turn.text,
            tool_calls=turn.tool_calls,
            raw_content=turn.raw_content,
        ),
        ProviderMessage(role=MessageRole.USER, tool_results=results),
    ]


def _briefing(context: SubtaskContext) -> str:
    """Build the user turn: the step, the request behind it, and what already exists.

    Formatting lives here, not in `prompts/` (§11), and the untrusted text — the user's
    request and the planner's instruction — stays out of the system prompt.

    Earlier artifacts are named, not resolved: a worker that needs a payload reads it
    through the store, and the pointers are here so the model does not go and fetch
    something the team already has.
    """
    lines = [
        f"Subtask: {context.subtask.instruction}",
        f"The request this serves: {context.user_request}",
    ]
    if context.inputs:
        lines.append(
            "Earlier steps already produced: "
            + ", ".join(f"{name} ({pointer})" for name, pointer in sorted(context.inputs.items()))
        )
    lines += [
        f"Clarification asked: {item.question}\nThe user answered: {item.answer}"
        for item in context.clarifications
    ]
    return "\n".join(lines)
