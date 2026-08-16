"""Composition root: the one place every service is constructed and wired (§3.1).

Nothing below this module builds a provider, a store, a broker or a worker — they are
handed one. That is what lets a test run the whole application against `FakeProvider`
without patching anything, and what keeps `cli/` free of business logic (§4).

Phase B replaces the stub role by role (#5-#7). Data Retrieval is real; Analytics and
Visualization still echo. Nothing else in the application changed when the first one
landed, which is the property the mapping in `build_orchestra` exists to have.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack

from orchestra.agents.aggregator import Aggregator
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.toolsets import data_retrieval_tools
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.data_retrieval import DataRetrievalWorker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.artifacts import ArtifactStore
from orchestra.config import Config, load_config
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, TaskEvent, TaskState
from orchestra.providers.base import Provider, create_provider


class Orchestra:
    """The application: plan a request, execute the plan, report on it, hand back the ledger."""

    def __init__(
        self,
        *,
        planner: Planner,
        engine: ExecutionEngine,
        aggregator: Aggregator,
        provider: Provider,
        broker: Broker[TaskEvent],
    ) -> None:
        """Take the wired services.

        Args:
            planner: turns the request into a plan.
            engine: executes it.
            aggregator: synthesises the artifacts into the run's final report.
            provider: held only so the run can release it; agents get it injected.
            broker: the run's event stream, exposed for the renderer to subscribe to (#11).
        """
        self._planner = planner
        self._engine = engine
        self._aggregator = aggregator
        self._provider = provider
        self._broker = broker

    @property
    def broker(self) -> Broker[TaskEvent]:
        """The event stream a dashboard subscribes to before calling `run_task`."""
        return self._broker

    async def run_task(self, prompt: str) -> TaskState:
        """Plan `prompt`, execute the plan, and write the run's report.

        Args:
            prompt: the user's plain-language request.

        Returns:
            The ledger, with `final_result` set, whether the run completed, partly
            failed, or stopped short. The caller reads `failed` for the exit code (§8)
            and `failure_reason` for why.

        Raises:
            TaskFailure: planning failed — no plan, so nothing to report on. An
                execution failure does *not* reach here.
            ProviderError: the provider failed while planning or synthesising.
            asyncio.CancelledError: the run was cancelled; propagated (§10).
        """
        state = TaskState(user_request=prompt)
        await self._planner.create_plan(state)
        try:
            await self._engine.run(state)
        except TaskFailure as exc:
            # The step cap, or a role with no worker. Recorded rather than raised, so the
            # report still names the artifacts on disk instead of exiting 5 in silence.
            # `CancelledError` is not an `Exception`, so a cancelled run unwinds untouched.
            state.failure_reason = str(exc)
        await self._aggregator.write_report(state)
        return state

    async def aclose(self) -> None:
        """Release the provider's connections. Idempotent."""
        await self._provider.aclose()


def build_orchestra(config: Config) -> Orchestra:
    """Construct the application from validated configuration.

    Args:
        config: the run's settings, already loaded and validated.

    Returns:
        A wired `Orchestra`. Substitute a service by calling the constructor directly.

    Raises:
        ConfigError: the artifact directory is unusable (§9 — fail before work starts).
    """
    provider = create_provider(api_key=config.anthropic_api_key, model=config.anthropic_model)
    store = ArtifactStore(config.artifact_dir)
    broker: Broker[TaskEvent] = Broker()
    # The mapping is the seam Phase B swaps role by role, which is why it is a mapping
    # and not a conditional in the engine: a real worker lands as one reassignment.
    # `dict.fromkeys` first so a role added later still runs, as a stub, rather than
    # failing `_check_roles` before the run starts.
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, EchoWorker(store))
    workers[AgentRole.DATA_RETRIEVAL] = DataRetrievalWorker(
        provider=provider,
        store=store,
        tools=data_retrieval_tools(config.data_dir, search_api_key=config.tavily_api_key),
    )
    return Orchestra(
        planner=Planner(provider),
        engine=ExecutionEngine(workers=workers, broker=broker),
        # The workers' own store: the aggregator resolves the pointers they minted.
        aggregator=Aggregator(provider, store),
        provider=provider,
        broker=broker,
    )


type RunObserver = Callable[[Broker[TaskEvent]], AbstractAsyncContextManager[object]]
"""Something that watches a run: given the broker, it stays attached for the run's
duration. `cli/render.py`'s dashboard is one (#11).

A parameter rather than an import because the layer rule runs one way (§3.2): `cli/`
may import `app.py`, so `app.py` may not name the renderer.

`[object]`, not `[None]`: the type is covariant in what it yields, so `[None]` would
reject every observer that yields something — `dashboard` hands back its `RunView`.
`run_once` discards the value, so nothing here depends on what it was.
"""


async def run_once(prompt: str, *, observer: RunObserver | None = None) -> TaskState:
    """Load configuration, run `prompt` once, and release the provider.

    The entry point `cli/app.py` delegates to, so the command body stays a parse, a
    delegation and an exit code (§4).

    Args:
        prompt: the user's plain-language request.
        observer: entered around the run, so it is subscribed before the first event is
            published and torn down after the last. `None` runs headless.

    Returns:
        The run's ledger, carrying the report the command prints.

    Raises:
        OrchestraError: configuration or planning failed. A run that started and then
            stopped short returns its ledger instead.
        asyncio.CancelledError: the run was cancelled; propagated after both the
            observer and the provider are released (§10).
    """
    orchestra = build_orchestra(load_config())
    try:
        # An exit stack rather than an `if`: the one that duplicated the `run_task` call
        # across both branches is how the two copies drift.
        async with AsyncExitStack() as stack:
            if observer is not None:
                await stack.enter_async_context(observer(orchestra.broker))
            return await orchestra.run_task(prompt)
    finally:
        # Runs on cancellation too: Ctrl-C must not leak the provider's sockets.
        await orchestra.aclose()
