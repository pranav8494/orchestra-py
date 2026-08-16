"""The first real worker: a bounded tool-use loop over the bundled offline data (#5).

**The model chooses the tool, not the code.** The alternative — inspect the instruction,
decide `query_csv` or `search` here, call it — is a keyword matcher wearing an agent's
name, and it cannot handle a step that needs both. So the two tools go to the model with
their schemas and it picks. The loop's job is only to run what it asks for and to stop.

**Both bounds, always (§10).** `max_turns` catches a model that keeps calling tools;
`token_budget` catches one that keeps calling *expensive* tools, which the turn count
alone will not see. Exceeding either is a `TaskFailure` — one failed subtask, not a
retry and not a hung run.

**One artifact, three parts.** `RetrievedDataset` carries the filtered rows, the search
snippets behind any claim about the wider market, and the agent's own summary. One
pointer because `Worker.run` returns one, and JSON because the next agent (#6) reads it
with `json.loads` rather than a parser written for this file. The rows stay CSV text
inside it: that is what the tool returned and what pandas will want, and re-encoding it
into JSON objects here would be a transformation nobody asked for and everybody would
have to undo.
"""

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from orchestra.agents.toolsets import QUERY_CSV_TOOL, SEARCH_TOOL
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.state import ArtifactPointer, SubtaskContext
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


class RetrievalSource(BaseModel):
    """One search the agent ran and what came back — provenance for the soft claims.

    Kept because a summary sentence like "margins are typical for the sector" is
    otherwise unattributable, and the report's rule is that a claim traces to something
    (§5 of the research doc).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    result: str


class RetrievedDataset(BaseModel):
    """The artifact this worker writes. The payload behind the pointer it returns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str
    summary: str
    # Empty when the subtask was answered from background alone — a legitimate outcome
    # for "find out what the sector's typical margin is", so not an error here.
    dataset_csv: str = ""
    sources: list[RetrievalSource] = Field(default_factory=list)


class DataRetrievalWorker:
    """Retrieves data with tools and stores what it found. Built in `app.py` (§3.3)."""

    def __init__(
        self,
        *,
        provider: Provider,
        store: ArtifactStore,
        tools: tuple[BaseTool, ...],
        max_turns: int = DEFAULT_MAX_TURNS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        """Take the wired services and the loop's bounds.

        Args:
            provider: the model provider to run the tool-use conversation with.
            store: the run's artifact store, where the retrieved dataset is written.
            tools: this agent's toolset, from `agents/toolsets.py`.
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
        dataset_csv = ""
        sources: list[RetrievalSource] = []
        summary = ""
        spent = 0

        for _ in range(self._max_turns):
            if spent >= self._token_budget:
                raise TaskFailure(
                    f"Retrieval for {context.subtask.id!r} spent its {self._token_budget}-token "
                    f"budget before finishing."
                )
            turn = await self._provider.send(
                system=DATA_RETRIEVAL_SYSTEM_PROMPT, messages=messages, tools=self._specs
            )
            spent += turn.usage_tokens

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
                if response.is_error:
                    continue
                # Last success wins: a second query is the agent correcting or narrowing
                # the first, so the later answer is the one it meant to keep.
                if call.name == QUERY_CSV_TOOL:
                    dataset_csv = response.content
                elif call.name == SEARCH_TOOL:
                    sources.append(_source(call, response))
            messages.extend(_exchange(turn, tuple(results)))
        else:
            raise TaskFailure(
                f"Retrieval for {context.subtask.id!r} was still calling tools after "
                f"{self._max_turns} turns."
            )

        if not dataset_csv and not sources:
            # Every tool call failed, or none was made. Either way the step produced no
            # data, and a summary with nothing behind it is exactly the invented answer
            # the design forbids — better a failed subtask the report can name (§8).
            raise TaskFailure(
                f"Retrieval for {context.subtask.id!r} finished without retrieving anything."
            )

        dataset = RetrievedDataset(
            instruction=context.subtask.instruction,
            summary=summary,
            dataset_csv=dataset_csv,
            sources=sources,
        )
        # `to_thread` because the store is synchronous filesystem I/O, and blocking the
        # event loop would serialise the engine's concurrent dispatch (§10).
        return await asyncio.to_thread(
            self._store.put_text, f"{context.subtask.id}.json", dataset.model_dump_json(indent=2)
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


def _exchange(turn: AssistantTurn, results: tuple[ToolResult, ...]) -> list[ProviderMessage]:
    """The two messages one round of tool use adds to the conversation.

    The assistant's own turn has to be replayed verbatim, tool calls included: the API
    is stateless, and a `tool_result` whose `tool_use` is missing from the history is
    rejected outright.
    """
    return [
        ProviderMessage(role=MessageRole.ASSISTANT, content=turn.text, tool_calls=turn.tool_calls),
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
    lines += [f"Clarification - {item.question} {item.answer}" for item in context.clarifications]
    return "\n".join(lines)
