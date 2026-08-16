"""The first real worker: retrieval over the bundled offline data (#5).

**The model chooses the tool, not the code.** Inspecting the instruction and picking
`query_csv` or `search` here is a keyword matcher wearing an agent's name, and it cannot
serve a step that needs both. Both schemas go to the model; the loop runs what it asks
for and stops.

**The loop itself is `tool_loop.ToolLoop`** — the bounds, the retries and the transcript
are the same for every worker that runs tools, so this module is only what makes the
retrieval agent that agent: its two tool names, its artifact, and what "retrieved
nothing" means.

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
from orchestra.agents.workers.tool_loop import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TOKEN_BUDGET,
    ToolLoop,
)
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import ArtifactPointer, SubtaskContext, TaskEvent
from orchestra.prompts import DATA_RETRIEVAL_SYSTEM_PROMPT
from orchestra.providers.base import Provider
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse


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
            broker: the run's event stream, for warnings raised mid-step. The engine
                publishes the step's *transitions*; only the loop can see a tool degrade
                partway through one, so it reports that itself.
            max_turns: how many model turns one subtask may take.
            token_budget: input plus output tokens one subtask may spend.

        Raises:
            ValueError: an empty toolset or a non-positive bound — a wiring bug, not a
                user-facing error, so it fails at construction like the engine's. Checked
                by the loop, which owns both bounds.
        """
        self._store = store
        self._loop = ToolLoop(
            provider=provider,
            broker=broker,
            tools=tools,
            system_prompt=DATA_RETRIEVAL_SYSTEM_PROMPT,
            label="Retrieval",
            max_turns=max_turns,
            token_budget=token_budget,
        )

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Retrieve what the subtask asks for and store it. See `Worker.run`.

        Raises:
            TaskFailure: the loop hit a bound, or it ended having retrieved nothing.
            asyncio.CancelledError: propagated from the provider or the store (§10).
        """
        result = await self._loop.run(context)

        # Split by tool name rather than by response shape: the two kinds are the same
        # thing to the loop and different artifacts here — rows to read, and provenance.
        datasets = [
            _table(outcome.call, outcome.response)
            for outcome in result.kept
            if outcome.call.name == QUERY_CSV_TOOL
        ]
        sources = [
            _source(outcome.call, outcome.response)
            for outcome in result.kept
            if outcome.call.name == SEARCH_TOOL
        ]

        if not datasets and not sources:
            # Every tool call failed, or none was made. Either way the step produced no
            # data, and a summary with nothing behind it is exactly the invented answer
            # the design forbids — better a failed subtask the report can name (§8).
            raise TaskFailure(
                f"Retrieval for {context.subtask.id!r} finished without retrieving anything."
            )

        dataset = RetrievedDataset(
            instruction=context.subtask.instruction,
            summary=result.summary,
            datasets=datasets,
            sources=sources,
        )
        # `to_thread` because the store is synchronous filesystem I/O, and blocking the
        # event loop would serialise the engine's concurrent dispatch (§10).
        return await asyncio.to_thread(
            self._store.put_text, f"{context.subtask.id}.json", dataset.model_dump_json(indent=2)
        )


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
