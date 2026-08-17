"""The first real worker: retrieval over the bundled offline data (#5).

**The model chooses the tool, not the code.** Picking `fetch_data` or `search` from the
instruction here is a keyword matcher wearing an agent's name, and it cannot serve a step
needing both. Both schemas go to the model; the loop runs what it asks for.

**The loop is `tool_loop.ToolLoop`**, so this module is only what makes the retrieval
agent that agent: its two tool names, its artifact, and what "retrieved nothing" means.

**One artifact.** `RetrievedDataset` holds rows, search provenance and summary under the
one pointer `Worker.run` returns, as JSON so #6 reads it with `json.loads`. Rows stay the
text the tool returned — what pandas wants; a file too large to inline is carried as its
artifact pointer instead, for the analysis step to open.

**Every successful fetch is kept.** `datasets` is a list because a step may need two
files: last-one-wins would store half the data under a summary describing all of it.
"""

import asyncio
import json

from pydantic import BaseModel, ConfigDict, Field

from orchestra.agents.toolsets import FETCH_DATA_TOOL, SEARCH_TOOL
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
from orchestra.tools.fetch_data import INLINED_KEY, POINTER_KEY


class RetrievalSource(BaseModel):
    """One search the agent ran and what came back — provenance for the soft claims.

    Without it a summary sentence like "margins are typical for the sector" traces to
    nothing, which the report's rules forbid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    result: str


class RetrievedTable(BaseModel):
    """One successful `fetch_data` call: what was asked for, and what came back.

    Either `csv` holds the file's own text, or `pointer` alone names it: a file too large
    to inline is passed on for the analysis step to open (#40).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str  # the call's arguments, rendered — provenance, like a source's
    csv: str = ""
    pointer: str = ""


class RetrievedDataset(BaseModel):
    """The artifact this worker writes. The payload behind the pointer it returns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str
    summary: str
    # Empty when the subtask was answered from background alone — legitimate for "what is
    # the sector's typical margin", so not an error here.
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
        """Take the wired services and the loop's bounds. See `ToolLoop.__init__`.

        Args:
            store: the run's artifact store, where the retrieved dataset is written.
            tools: this agent's toolset, from `agents/toolsets.py`.

        Raises:
            ValueError: an empty toolset or a non-positive bound, checked by the loop.
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

        # Split by tool name, not response shape: the two are the same to the loop and
        # different fields here — rows to read, and provenance.
        datasets = [
            _table(outcome.call, outcome.response)
            for outcome in result.kept
            if outcome.call.name == FETCH_DATA_TOOL
        ]
        sources = [
            _source(outcome.call, outcome.response)
            for outcome in result.kept
            if outcome.call.name == SEARCH_TOOL
        ]

        if not datasets and not sources:
            # A summary with no data behind it is the invented answer the design forbids —
            # better a failed subtask the report can name (§8).
            raise TaskFailure(
                f"Retrieval for {context.subtask.id!r} finished without retrieving anything."
            )

        dataset = RetrievedDataset(
            instruction=context.subtask.instruction,
            summary=result.summary,
            datasets=datasets,
            sources=sources,
        )
        # `to_thread` because the store is blocking I/O; blocking the loop would serialise
        # the engine's concurrent dispatch (§10).
        return await asyncio.to_thread(
            self._store.put_text, f"{context.subtask.id}.json", dataset.model_dump_json(indent=2)
        )


def _source(call: ToolCall, response: ToolResponse) -> RetrievalSource:
    """Pair a search call with its result, defensively about the argument's shape.

    `call.arguments` is model output, so `query` may be absent or not a string even after
    the tool accepted the call; recording it is not worth a second validation pass.
    """
    return RetrievalSource(query=str(call.arguments.get("query", "")), result=response.content)


def _table(call: ToolCall, response: ToolResponse) -> RetrievedTable:
    """Pair a `fetch_data` call with what it returned: the rows, or a pointer to them.

    Arguments render to sorted JSON rather than staying a mapping, so the field does not
    become a second schema for #6 to know about. The pointer is read from `metadata`, not
    parsed out of prose written for the model (§6), and a file that was not inlined leaves
    `csv` empty rather than storing its summary as if it were rows.
    """
    inlined = response.metadata.get(INLINED_KEY) == "true"
    return RetrievedTable(
        query=json.dumps(call.arguments, sort_keys=True),
        csv=response.content if inlined else "",
        pointer=response.metadata.get(POINTER_KEY, ""),
    )
