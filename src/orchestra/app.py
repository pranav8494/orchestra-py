"""Composition root: the one place every service is constructed and wired (§3.1).

Nothing below builds a provider, store, broker or worker — they are handed one, which
is what lets a test run the whole application against `FakeProvider` without patching.

Phase B replaced the stub role by role (#5-#7), each landing as one reassignment in
`build_orchestra`'s mapping and nothing else — which was its point. `EchoWorker` stays
as the mapping's default, so a role added later runs rather than failing at startup.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path

from orchestra.agents.aggregator import Aggregator
from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.toolsets import analytics_tools, data_retrieval_tools
from orchestra.agents.workers.analytics import AnalyticsWorker
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.data_retrieval import DataRetrievalWorker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.agents.workers.visualization import VisualizationWorker
from orchestra.artifacts import ArtifactStore
from orchestra.config import Config, load_config
from orchestra.core.errors import TaskFailure
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, TaskEvent, TaskState
from orchestra.providers.base import Provider, create_provider

# UTC and no colons: sortable as a plain string, and legal on Windows and in a shell.
RUN_DIR_FORMAT = "%Y-%m-%dT%H-%M-%SZ"


def _run_directory() -> str:
    """This run's subdirectory name.

    Generated here, not in `config.py`: a timestamp is not configuration. Second
    granularity keeps the name readable, at the price of two runs starting in the same
    second sharing a directory.
    """
    return datetime.now(UTC).strftime(RUN_DIR_FORMAT)


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
        artifact_dir: Path,
    ) -> None:
        """Take the wired services. `provider` is held only so the run can release it;
        `artifact_dir` is the store's root, recorded on each ledger this creates."""
        self._planner = planner
        self._engine = engine
        self._aggregator = aggregator
        self._provider = provider
        self._broker = broker
        self._artifact_dir = artifact_dir

    @property
    def broker(self) -> Broker[TaskEvent]:
        """The event stream a dashboard subscribes to before calling `run_task`."""
        return self._broker

    async def run_task(self, prompt: str) -> TaskState:
        """Plan `prompt`, execute the plan, and write the run's report.

        Returns the ledger whether the run completed, partly failed, or stopped short;
        the caller reads `failed` for the exit code (§8).

        Raises:
            TaskFailure: planning failed — an execution failure does *not* reach here.
            ProviderError: the provider failed while planning or synthesising.
        """
        state = TaskState(user_request=prompt, artifact_dir=self._artifact_dir)
        await self._planner.create_plan(state)
        try:
            await self._engine.run(state)
        except TaskFailure as exc:
            # Recorded, not raised, so the report still names the artifacts on disk
            # instead of exiting 5 in silence.
            state.failure_reason = str(exc)
        await self._aggregator.write_report(state)
        return state

    async def aclose(self) -> None:
        """Release the provider's connections. Idempotent."""
        await self._provider.aclose()


def build_orchestra(config: Config) -> Orchestra:
    """Construct the application from validated configuration.

    Substitute a service by calling `Orchestra` directly.

    Raises:
        ConfigError: the artifact directory is unusable (§9 — fail before work starts).
    """
    provider = create_provider(
        api_key=config.anthropic_api_key,
        model=config.anthropic_model,
        max_tokens=config.anthropic_max_tokens,
    )
    # Per run, not one flat directory for all time, where a repeated plan lands as
    # `_reserve`'s `-1` variants with nothing recording which run wrote which.
    store = ArtifactStore(config.artifact_dir / _run_directory())
    broker: Broker[TaskEvent] = Broker()
    # A mapping, not a conditional in the engine: a real worker lands as one
    # reassignment. `fromkeys` first so a role added later runs as a stub rather than
    # failing `_check_roles` before the run starts.
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, EchoWorker(store))
    # The same bounds for both: one budget per subtask, not per role, so `WORKER_MAX_TURNS`
    # means the same thing wherever the operator reads it. Passed by name rather than
    # unpacked from a dict, which mypy would not check against either constructor.
    workers[AgentRole.DATA_RETRIEVAL] = DataRetrievalWorker(
        provider=provider,
        store=store,
        broker=broker,
        tools=data_retrieval_tools(config.data_dir, search_api_key=config.tavily_api_key),
        max_turns=config.worker_max_turns,
        token_budget=config.worker_token_budget,
    )
    workers[AgentRole.ANALYTICS] = AnalyticsWorker(
        provider=provider,
        store=store,
        broker=broker,
        tools=analytics_tools(store),
        max_turns=config.worker_max_turns,
        token_budget=config.worker_token_budget,
    )
    # No loop bounds: one structured call, not a tool-use conversation, so there is no
    # iteration count or budget for the operator to set (§10).
    workers[AgentRole.VISUALIZATION] = VisualizationWorker(
        provider=provider, store=store, broker=broker
    )
    return Orchestra(
        planner=Planner(provider),
        engine=ExecutionEngine(
            workers=workers,
            broker=broker,
            max_concurrency=config.max_concurrency,
            subtask_attempts=config.subtask_attempts,
        ),
        # The workers' own store: the aggregator resolves the pointers they minted.
        aggregator=Aggregator(provider, store),
        provider=provider,
        broker=broker,
        artifact_dir=store.root,
    )


type RunObserver = Callable[[Broker[TaskEvent]], AbstractAsyncContextManager[object]]
"""Watches a run: given the broker, stays attached for its duration (`cli/render.py`'s
dashboard is one, #11).

A parameter rather than an import because the layer rule runs one way (§3.2): `cli/`
may import `app.py`, so `app.py` may not name the renderer. `[object]`, not `[None]`,
so an observer yielding something — `dashboard` yields its `RunView` — still fits.
"""


async def run_once(prompt: str, *, observer: RunObserver | None = None) -> TaskState:
    """Load configuration, run `prompt` once, and release the provider.

    What `cli/app.py` delegates to, so the command body stays parse, delegate, exit (§4).
    `observer` is entered around the run, so it is subscribed before the first event and
    torn down after the last; `None` runs headless.

    Raises:
        OrchestraError: configuration or planning failed. A run that started and then
            stopped short returns its ledger instead.
    """
    orchestra = build_orchestra(load_config())
    try:
        # An exit stack rather than an `if`, which would duplicate the `run_task` call
        # across both branches.
        async with AsyncExitStack() as stack:
            if observer is not None:
                await stack.enter_async_context(observer(orchestra.broker))
            return await orchestra.run_task(prompt)
    finally:
        # Runs on cancellation too: Ctrl-C must not leak the provider's sockets.
        await orchestra.aclose()
