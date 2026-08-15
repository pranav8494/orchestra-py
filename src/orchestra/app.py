"""Composition root: the one place every service is constructed and wired (§3.1).

Nothing below this module builds a provider, a store, a broker or a worker — they are
handed one. That is what lets a test run the whole application against `FakeProvider`
without patching anything, and what keeps `cli/` free of business logic (§4).

Phase A wires the stub worker into every role. Phase B replaces those entries one at a
time (#5-#7); nothing else in the application changes when it does.
"""

from orchestra.agents.engine import ExecutionEngine
from orchestra.agents.planner import Planner
from orchestra.agents.workers.base import Worker
from orchestra.agents.workers.stub import EchoWorker
from orchestra.artifacts import ArtifactStore
from orchestra.config import Config, load_config
from orchestra.core.events import Broker
from orchestra.core.state import AgentRole, TaskEvent, TaskState
from orchestra.providers.base import Provider, create_provider


class Orchestra:
    """The application: plan a request, execute the plan, hand back the ledger."""

    def __init__(
        self,
        *,
        planner: Planner,
        engine: ExecutionEngine,
        provider: Provider,
        broker: Broker[TaskEvent],
    ) -> None:
        """Take the wired services.

        Args:
            planner: turns the request into a plan.
            engine: executes it.
            provider: held only so the run can release it; agents get it injected.
            broker: the run's event stream, exposed for the renderer to subscribe to (#11).
        """
        self._planner = planner
        self._engine = engine
        self._provider = provider
        self._broker = broker

    @property
    def broker(self) -> Broker[TaskEvent]:
        """The event stream a dashboard subscribes to before calling `run_task`."""
        return self._broker

    async def run_task(self, prompt: str) -> TaskState:
        """Plan and execute `prompt`.

        Args:
            prompt: the user's plain-language request.

        Returns:
            The ledger, whether or not every subtask succeeded — a partially failed run
            still has artifacts worth reporting. The caller reads `failed_subtasks` to
            decide the exit code (§8).

        Raises:
            OrchestraError: planning failed, or the run hit a limit that ends it.
            asyncio.CancelledError: the run was cancelled; propagated (§10).
        """
        state = TaskState(user_request=prompt)
        await self._planner.create_plan(state)
        await self._engine.run(state)
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
    # Every role gets the same stub in Phase A; the mapping is the seam Phase B swaps
    # role by role, which is why it is a mapping and not a conditional in the engine.
    echo = EchoWorker(store)
    workers: dict[AgentRole, Worker] = dict.fromkeys(AgentRole, echo)
    return Orchestra(
        planner=Planner(provider),
        engine=ExecutionEngine(workers=workers, broker=broker),
        provider=provider,
        broker=broker,
    )


async def run_once(prompt: str) -> TaskState:
    """Load configuration, run `prompt` once, and release the provider.

    The entry point `cli/app.py` delegates to, so the command body stays a parse, a
    delegation and an exit code (§4).

    Args:
        prompt: the user's plain-language request.

    Returns:
        The run's ledger.

    Raises:
        OrchestraError: configuration, planning or execution failed.
    """
    orchestra = build_orchestra(load_config())
    try:
        return await orchestra.run_task(prompt)
    finally:
        # Runs on cancellation too: Ctrl-C must not leak the provider's sockets.
        await orchestra.aclose()
