"""The second real worker: computation over what an earlier step retrieved (#6).

**The model writes the code.** The ticket's fallback — a fixed set of typed pandas helpers
— is a worse pandas with a schema: analysis is open-ended, so every question outside the
enumeration comes back unanswerable. `run_python` makes a written script affordable (§6).

**The loop is `tool_loop.ToolLoop`**, so this module is only what makes the analysis agent
that agent: its executor, its artifact, and what "computed nothing" means.

**One artifact**, like retrieval's: numbers and the scripts behind them under the single
pointer `Worker.run` returns, as JSON so the aggregator and #7 read it with `json.loads`.

**Every figure leads.** The aggregator sees a *preview*, so field order is a budget —
see `AnalysisResult`.
"""

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from orchestra.agents.workers.tool_loop import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TOKEN_BUDGET,
    ToolLoop,
)
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import ArtifactPointer, SubtaskContext, TaskEvent
from orchestra.prompts import ANALYTICS_SYSTEM_PROMPT
from orchestra.providers.base import Provider
from orchestra.tools.base import BaseTool, ToolCall, ToolResponse


class Computation(BaseModel):
    """One script that ran, and what it printed.

    `stdout` before `code`, like `AnalysisResult`: the output is the result, the script is
    provenance for it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stdout: str
    code: str


class AnalysisResult(BaseModel):
    """The artifact this worker writes. The payload behind the pointer it returns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Field order is load-bearing. The aggregator (#8) sees `ArtifactStore.preview`, which
    # elides past 800 characters — less than one escaped pandas script — so *every* output
    # goes ahead of *any* script. `figures` guarantees the numbers survive; `computations`
    # repeats them beside their code, which is provenance and may be elided. `instruction`
    # is last, unlike `RetrievedDataset`: the aggregator already has the plan's.
    summary: str
    figures: list[str] = Field(default_factory=list)
    computations: list[Computation] = Field(default_factory=list)
    instruction: str


class AnalyticsWorker:
    """Computes over an earlier step's data and stores the results. Built in `app.py` (§3.3)."""

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
            store: the run's artifact store, where the analysis is written. The executor
                holds the same store via `agents/toolsets.py` — this one writes the output,
                that one resolves the inputs.
            tools: this agent's toolset, from `agents/toolsets.py`.

        Raises:
            ValueError: an empty toolset or a non-positive bound, checked by the loop.
        """
        self._store = store
        self._loop = ToolLoop(
            provider=provider,
            broker=broker,
            tools=tools,
            system_prompt=ANALYTICS_SYSTEM_PROMPT,
            label="Analysis",
            max_turns=max_turns,
            token_budget=token_budget,
        )

    async def run(self, context: SubtaskContext) -> ArtifactPointer:
        """Compute what the subtask asks for and store it. See `Worker.run`.

        Raises:
            TaskFailure: the loop hit a bound, or it ended having computed nothing.
            asyncio.CancelledError: propagated from the provider or the store (§10).
        """
        result = await self._loop.run(context)

        # No split by tool name, unlike retrieval's two: one tool, so everything kept is a
        # script that ran and printed.
        computations = [_computation(outcome.call, outcome.response) for outcome in result.kept]

        if not computations:
            # A summary with no figures behind it is the invented answer the design
            # forbids — better a failed subtask the report can name (§8).
            raise TaskFailure(
                f"Analysis for {context.subtask.id!r} finished without computing anything."
            )

        analysis = AnalysisResult(
            summary=result.summary,
            figures=[item.stdout for item in computations],
            computations=computations,
            instruction=context.subtask.instruction,
        )
        # `to_thread` because the store is blocking I/O; blocking the loop would serialise
        # the engine's concurrent dispatch (§10).
        return await asyncio.to_thread(
            self._store.put_text, f"{context.subtask.id}.json", analysis.model_dump_json(indent=2)
        )


def _computation(call: ToolCall, response: ToolResponse) -> Computation:
    """Pair a script's output with the script, defensively about the argument's shape.

    `call.arguments` is model output, so `code` may be absent or not a string even after
    the tool accepted the call; recording it is not worth a second validation pass.
    """
    return Computation(stdout=response.content, code=str(call.arguments.get("code", "")))
