"""The second real worker: computation over what an earlier step retrieved (#6).

**The model writes the code.** The ticket's fallback — a fixed set of typed pandas helpers
— is a worse pandas with a schema: analysis is open-ended, so every question outside the
enumeration comes back unanswerable. `run_python` makes a written script affordable (§6).

**The loop is `tool_loop.ToolLoop`**, so this module is only what makes the analysis agent
that agent: its executor, its artifact, and what "computed nothing" means.

**One artifact**, like retrieval's: numbers and the scripts behind them under the single
pointer `Worker.run` returns, as JSON so the aggregator and #7 read it with `json.loads`.

**Every figure leads, and cites.** Each number is recorded against the artifact its script
read, so the report cites what was computed rather than what the aggregator guessed (#9).
The aggregator sees a *preview*, so field order is a budget — see `AnalysisResult`.
"""

import asyncio
from collections.abc import Collection

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.agents.workers.tool_loop import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TOKEN_BUDGET,
    ToolLoop,
)
from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import ArtifactPointer, KeyFigure, SubtaskContext, TaskEvent
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
    # elides past 800 characters — less than one escaped pandas script — so *every* figure
    # goes ahead of *any* script — the more so now a figure carries its source pointer too,
    # which spends more of that budget than the bare string it replaced. `computations`
    # repeats each number beside its code, which is provenance and may be elided.
    # `instruction` is last, unlike `RetrievedDataset`: the aggregator already has the plan's.
    summary: str
    figures: list[KeyFigure] = Field(default_factory=list)
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
        # What an earlier step produced, which is the only thing a figure may cite. A
        # script may stage more than that — a raw data file `fetch_data` registered — and
        # those pointers are no step's output.
        upstream = frozenset(context.inputs.values())
        figures = [
            figure
            for outcome in result.kept
            if (figure := _figure(outcome.call, outcome.response, upstream)) is not None
        ]

        if not computations:
            # A summary with no computation behind it is the invented answer the design
            # forbids — better a failed subtask the report can name (§8). On `computations`,
            # not `figures`: a script that ran and printed is work done, unsourced or not.
            raise TaskFailure(
                f"Analysis for {context.subtask.id!r} finished without computing anything."
            )

        analysis = AnalysisResult(
            summary=result.summary,
            figures=figures,
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


def _figure(call: ToolCall, response: ToolResponse, upstream: Collection[str]) -> KeyFigure | None:
    """Pair a script's output with the upstream artifact its step was given — the first
    of them the call named.

    Not "the artifact it read", which is what the script's own code decides and this
    cannot know, and not `inputs[0]`: since #40 a script may also stage the raw data file
    `fetch_data` registered, and that pointer is no subtask's output, so
    `TaskState.backed_figures` drops a figure citing it and the report loses the number.

    `None` when the call named no upstream pointer: a number the plan cannot trace to a
    step is dropped rather than sourced to a guess, which is what #9 exists to stop.
    """
    inputs = call.arguments.get("inputs")
    named = [item for item in inputs if item in upstream] if isinstance(inputs, list) else []
    first: object = named[0] if named else None
    try:
        # Validated, not trusted: `call.arguments` is model output, so the pointer's shape
        # is checked here even though the tool accepted the call (§7).
        return KeyFigure.model_validate({"value": response.content, "source": first})
    except ValidationError:
        return None
