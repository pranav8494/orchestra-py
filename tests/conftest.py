"""Shared test fixtures.

`FakeProvider` is the substitute that lets the whole suite run without touching the
network — the payoff for keeping vendor SDKs behind the provider port. It answers from
a queue and records what it was asked, so a test asserts on the conversation as well as
on the result.

See CONVENTIONS.md §12.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import BaseModel

from orchestra.providers.base import Provider, ProviderMessage, StructuredT

# Every setting `Config` reads. Listed once so a new field cannot silently start
# leaking the developer's shell into the suite.
_SETTING_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ARTIFACT_DIR")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cut every test off from the ambient environment and any real `.env` (§9).

    `Config` reads the environment and a `.env` resolved against the *current working
    directory*, so without both halves of this a developer with an exported key gets a
    different result from CI. Autouse because that hazard is suite-wide, not per-test.
    """
    for var in _SETTING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


@dataclass(frozen=True, slots=True)
class ParseCall:
    """One recorded `parse_structured` call."""

    system: str
    messages: tuple[ProviderMessage, ...]
    output_format: type[BaseModel]


@dataclass
class FakeProvider:
    """A `Provider` that answers from a queue and never opens a socket (§12).

    Queue entries are returned in order: a pydantic model is handed back as the parsed
    output, `None` stands for a refusal or a truncated reply, and an exception is
    raised. Running the queue dry is a test-authoring bug, so it fails loudly rather
    than repeating the last answer.
    """

    responses: list[BaseModel | BaseException | None] = field(default_factory=list)
    model: str = "fake-model"
    # Set to hold every call open until the test releases it — how a cancellation test
    # gets a request in flight to cancel.
    blocker: asyncio.Event | None = None
    calls: list[ParseCall] = field(default_factory=list)
    closed: bool = False

    async def parse_structured(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        output_format: type[StructuredT],
    ) -> StructuredT | None:
        """Record the call and answer from the queue. See `Provider.parse_structured`."""
        self.calls.append(ParseCall(system, tuple(messages), output_format))
        if self.blocker is not None:
            await self.blocker.wait()
        if not self.responses:
            raise AssertionError(f"FakeProvider has no queued response for call {len(self.calls)}")
        answer = self.responses.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        # The queue is heterogeneous by design; the test decides what the call returns.
        return cast("StructuredT | None", answer)

    async def aclose(self) -> None:
        """Nothing to release; the flag lets a test assert the caller closed it."""
        self.closed = True


if TYPE_CHECKING:
    # Conformance is checked by mypy, not by `isinstance`: `runtime_checkable` compares
    # attribute names only, so it would pass a fake whose signature had drifted.
    _PROTOCOL_CHECK: Provider = FakeProvider()
