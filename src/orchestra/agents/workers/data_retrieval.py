"""The first real worker: a bounded tool-use loop over the bundled offline data (#5).

**The model chooses the tool, not the code.** Inspecting the instruction and picking
`query_csv` or `search` here is a keyword matcher wearing an agent's name, and it cannot
serve a step that needs both. Both schemas go to the model; the loop runs what it asks
for and stops.

**Both bounds, always (§10).** `max_turns` catches a model that keeps calling tools;
`token_budget` catches one calling *expensive* ones, which a turn count will not see.
Exceeding either is a `TaskFailure` — one failed subtask, not a retry.

**One artifact.** `RetrievedDataset` holds the rows, the search provenance and the
agent's summary under the one pointer `Worker.run` returns, as JSON so #6 reads it with
`json.loads`. Rows stay CSV text — what the tool returned and what pandas wants.

**Every successful query is kept.** `datasets` is a list: the tool advertises a `columns`
filter, so asking for revenue and then costs is invited behaviour, and last-one-wins
would store half the data under a summary describing all of it.
"""

import asyncio
import json

from pydantic import BaseModel, ConfigDict, Field

from orchestra.agents.toolsets import QUERY_CSV_TOOL, SEARCH_TOOL
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import ArtifactPointer, EventKind, SubtaskContext, TaskEvent
from orchestra.prompts import DATA_RETRIEVAL_SYSTEM_PROMPT
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


class RetrievalSource(BaseModel):
    """One search the agent ran and what came back — provenance for the soft claims.

    Kept because a summary sentence like "margins are typical for the sector" is
    otherwise unattributable, and the report's rule is that a claim traces to something
    (§5 of the research doc).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    result: str


class RetrievedTable(BaseModel):
    """One successful `query_csv` call: what was asked for, and the rows it returned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str  # the call's arguments, rendered — provenance, symmetric with a source
    csv: str


class RetrievedDataset(BaseModel):
    """The artifact this worker writes. The payload behind the pointer it returns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str
    summary: str
    # Empty when the subtask was answered from background alone — a legitimate outcome
    # for "find out what the sector's typical margin is", so not an error here.
    datasets: list[RetrievedTable] = Field(default_factory=list)
    sources: list[RetrievalSource] = Field(default_factory=list)


class DataRetrievalWorker:
    """Retrieves data with tools and stores what it found. Built in `app.py` (§3.3)."""

    def __init__(
        self,
        *,
        provider: Provider,
        store: ArtifactStore,
        tools: tuple[BaseTool, ...],
        broker: Broker[TaskEvent],
        max_turns: int = DEFAULT_MAX_TURNS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        """Take the wired services and the loop's bounds.

        Args:
            provider: the model provider to run the tool-use conversation with.
            store: the run's artifact store, where the retrieved dataset is written.
            tools: this agent's toolset, from `agents/toolsets.py`.
            broker: the run's event stream, for warnings this worker raises mid-step.
                The engine publishes the step's *transitions*; only the worker can see a
                tool degrade partway through one, so it reports that itself.
            max_turns: how many model turns one subtask may take.
            token_budget: input plus output tokens one subtask may spend.

        Raises:
            ValueError: an empty toolset or a non-positive bound — a wiring bug, not a
                user-facing error, so it fails at construction like the engine's.
        """
        if not tools:
            raise ValueError("DataRetrievalWorker needs at least one tool")
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns}")
        if token_budget < 1:
            raise ValueError(f"token_budget must be at least 1, got {token_budget}")
        self._provider = provider
        self._store = store
        self._broker = broker
        self._tools = {tool.info().name: tool for tool in tools}
        self._specs = [tool.info() for tool in tools]
        self._max_turns = max_turns
        self._token_budget = token_budget

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Retrieve what the subtask asks for and store it. See `Worker.run`.

        Raises:
            TaskFailure: the loop hit a bound, or it ended having retrieved nothing.
            asyncio.CancelledError: propagated from the provider or the store (§10).
        """
        messages = [ProviderMessage(role=MessageRole.USER, content=_briefing(context))]
        datasets: list[RetrievedTable] = []
        sources: list[RetrievalSource] = []
        summary = ""
        spent = 0

        for _ in range(self._max_turns):
            # Checked before the call, not after: the cost of a turn is only known once
            # it has been paid, so this bounds what the loop *starts*, and one turn may
            # carry the total past the budget. A ceiling on spend, not on the last bill.
            if spent >= self._token_budget:
                raise TaskFailure(
                    f"Retrieval for {context.subtask.id!r} spent its {self._token_budget}-token "
                    f"budget before finishing. {_gathered(datasets, sources)}"
                )
            turn = await self._provider.send(
                system=DATA_RETRIEVAL_SYSTEM_PROMPT, messages=messages, tools=self._specs
            )
            # The whole prompt is resent every turn, so this counts the transcript once
            # per lap and the budget bites sooner than a distinct-token count would.
            # Deliberate: resent tokens are billed tokens, and it is spend being bounded.
            spent += turn.usage_tokens

            if turn.stop_reason == TRUNCATED:
                # A reply cut off mid-generation carries no tool call, which is exactly
                # what "the agent is finished" looks like. Left unchecked, half a sentence
                # is stored as the summary and the run reports success (§8).
                raise TaskFailure(
                    f"Retrieval for {context.subtask.id!r} was cut off by the model's output "
                    f"limit. {_gathered(datasets, sources)}"
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
                # `is_empty` and not just `is_error`: a search that matched nothing ran
                # correctly, but recording it would let "nothing was found" satisfy the
                # did-we-retrieve-anything check below (§6, and `ToolResponse`).
                if response.is_error or response.is_empty:
                    continue
                if call.name == QUERY_CSV_TOOL:
                    datasets.append(_table(call, response))
                elif call.name == SEARCH_TOOL:
                    sources.append(_source(call, response))
            messages.extend(_exchange(turn, tuple(results)))
        else:
            raise TaskFailure(
                f"Retrieval for {context.subtask.id!r} was still calling tools after "
                f"{self._max_turns} turns. {_gathered(datasets, sources)}"
            )

        if not datasets and not sources:
            # Every tool call failed, or none was made. Either way the step produced no
            # data, and a summary with nothing behind it is exactly the invented answer
            # the design forbids — better a failed subtask the report can name (§8).
            raise TaskFailure(
                f"Retrieval for {context.subtask.id!r} finished without retrieving anything."
            )

        dataset = RetrievedDataset(
            instruction=context.subtask.instruction,
            summary=summary,
            datasets=datasets,
            sources=sources,
        )
        # `to_thread` because the store is synchronous filesystem I/O, and blocking the
        # event loop would serialise the engine's concurrent dispatch (§10).
        return await asyncio.to_thread(
            self._store.put_text, f"{context.subtask.id}.json", dataset.model_dump_json(indent=2)
        )

    async def _warn(self, context: SubtaskContext, warning: str) -> None:
        """Tell the run that this step degraded, without failing it.

        Must-deliver rather than lossy progress: a dropped frame of a progress stream
        costs a subscriber nothing, but this one changes what the operator believes the
        answer is made of, which is the same class of fact as a state transition (§6).
        Bounded inside the broker, so a wedged dashboard still cannot hang the run.

        Not recorded on the ledger, unlike the engine's own events: a worker sees only
        its slice and has no `TaskState` to append to (§6). The notice is durable
        anyway — it is inside the artifact this step writes.
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


def _source(call: ToolCall, response: ToolResponse) -> RetrievalSource:
    """Pair a search call with its result, defensively about the argument's shape.

    `call.arguments` is model output, so `query` may be absent or not a string even
    though the tool accepted the call — recording it is not worth a second validation
    pass, and `str()` cannot fail here.
    """
    return RetrievalSource(query=str(call.arguments.get("query", "")), result=response.content)


def _table(call: ToolCall, response: ToolResponse) -> RetrievedTable:
    """Pair a `query_csv` call with the rows it returned.

    The arguments are rendered as sorted JSON rather than kept as a mapping: the artifact
    is read by an agent and a person, and both want to see what was asked for without
    the field becoming a second schema for #6 to know about.
    """
    return RetrievedTable(query=json.dumps(call.arguments, sort_keys=True), csv=response.content)


def _gathered(datasets: list[RetrievedTable], sources: list[RetrievalSource]) -> str:
    """Say what the step had in hand when it stopped short.

    A bound was hit, so the artifact is never written and the work is lost. Naming it in
    the failure is the difference between "raise the cap" and "debug the agent" (§8).
    """
    if not datasets and not sources:
        return "It had retrieved nothing at that point."
    return (
        f"It had retrieved {len(datasets)} table(s) and {len(sources)} search result(s), "
        f"which are lost with the step."
    )


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

    Earlier artifacts are named, not resolved: this agent retrieves rather than reads,
    and the pointers are there so it does not go and fetch something the team has.
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
