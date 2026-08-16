"""The second real worker: computation over what an earlier step retrieved (#6).

**The model writes the code.** The ticket's fallback is a fixed set of typed pandas
helpers — sum, group, pct_change — and that is a worse pandas with a schema: analysis is
open-ended, the useful operation is whichever one the subtask happens to name, and every
question outside the enumeration would come back unanswerable. A script the model writes
covers all of them, and `run_python` is what makes that affordable to run (§6).

**The loop itself is `tool_loop.ToolLoop`** — bounds, retries and transcript are the same
for every worker that runs tools, so this module is only what makes the analysis agent
that agent: its executor, its artifact, and what "computed nothing" means.

**One artifact**, like retrieval's: the numbers and the scripts behind them live under
the single pointer `Worker.run` returns, as JSON so the aggregator and #7 read it with
`json.loads` instead of learning a second format.

**Every figure leads.** The aggregator sees a *preview* of this file, not the file, so
field order is a budget: every number the run computed goes first, and the scripts behind
them go after, where the elision can take them without costing the report a figure.
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

    `stdout` before `code` for the same reason `AnalysisResult` orders its fields that
    way: the output is the result, the script is the provenance for it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stdout: str
    code: str


class AnalysisResult(BaseModel):
    """The artifact this worker writes. The payload behind the pointer it returns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Field order is load-bearing, not taste. The aggregator (#8) is shown
    # `ArtifactStore.preview`, which elides past 800 characters, and one real pandas
    # script escapes to more than that on its own — so *every* output goes ahead of *any*
    # script. Interleaving them is not enough: with the pairs first, the prompt permitting
    # two scripts is enough for the first script's code to push the second one's figures
    # past the cut. `figures` is what guarantees the numbers survive; `computations`
    # repeats them beside the code that produced each one, which is provenance and can be
    # elided. `instruction` sits last on purpose, unlike `RetrievedDataset`: the
    # aggregator is handed the plan's instructions already, so re-reading this one costs
    # the preview a figure and tells it nothing new.
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
        """Take the wired services and the loop's bounds.

        Args:
            provider: the model provider to run the tool-use conversation with.
            store: the run's artifact store, where the analysis is written. The executor
                gets its own handle on the same store, from `agents/toolsets.py` — this
                one writes the output, that one resolves the inputs.
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

        # No split by tool name, unlike retrieval's two: this agent has one tool, and
        # everything the loop kept is by definition a script that ran and printed.
        computations = [_computation(outcome.call, outcome.response) for outcome in result.kept]

        if not computations:
            # Every script failed, printed nothing, or none was written. Either way the
            # step computed no figures, and a summary with nothing behind it is exactly
            # the invented answer the design forbids — better a failed subtask the report
            # can name (§8).
            raise TaskFailure(
                f"Analysis for {context.subtask.id!r} finished without computing anything."
            )

        analysis = AnalysisResult(
            summary=result.summary,
            figures=[item.stdout for item in computations],
            computations=computations,
            instruction=context.subtask.instruction,
        )
        # `to_thread` because the store is synchronous filesystem I/O, and blocking the
        # event loop would serialise the engine's concurrent dispatch (§10).
        return await asyncio.to_thread(
            self._store.put_text, f"{context.subtask.id}.json", analysis.model_dump_json(indent=2)
        )


def _computation(call: ToolCall, response: ToolResponse) -> Computation:
    """Pair a script's output with the script, defensively about the argument's shape.

    `call.arguments` is model output, so `code` may be absent or not a string even though
    the tool accepted the call — recording it is not worth a second validation pass, and
    `str()` cannot fail here.
    """
    return Computation(stdout=response.content, code=str(call.arguments.get("code", "")))
