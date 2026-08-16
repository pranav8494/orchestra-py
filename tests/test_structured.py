"""Tests for the shared validate-and-retry helper.

Everything runs against `FakeProvider`. The assertions are about the loop around the
provider — how many calls it makes, what a retry turn carries, what it returns when
nothing validates — never about the model's judgement.
"""

import asyncio

import pytest
from pydantic import BaseModel

from conftest import FakeProvider
from orchestra.agents.structured import DEFAULT_MAX_RETRIES, parse_validated
from orchestra.core.errors import ProviderError
from orchestra.providers.base import MessageRole, ProviderMessage

SYSTEM = "Count what you are shown."
BRIEFING = "Three quarters of revenue."
INSTRUCTION = "That reply was rejected; send a corrected one."


class _Draft(BaseModel):
    """A schema-valid draft that may still be unusable — the case the helper exists for."""

    value: int


def _positive(draft: _Draft) -> int:
    """The caller's conversion: shape is pydantic's, the rule is the caller's."""
    if draft.value <= 0:
        raise ValueError(f"value must be positive, got {draft.value}")
    return draft.value


async def _parse(provider: FakeProvider, **overrides: int) -> tuple[int | None, str]:
    return await parse_validated(
        provider=provider,
        system=SYSTEM,
        messages=[ProviderMessage(role=MessageRole.USER, content=BRIEFING)],
        output_format=_Draft,
        validate=_positive,
        instruction=INSTRUCTION,
        **overrides,
    )


@pytest.mark.asyncio
async def test_parse_validated_valid_first_reply_returns_it_without_a_retry_turn() -> None:
    provider = FakeProvider(responses=[_Draft(value=3)])

    value, rejection = await _parse(provider)

    assert (value, rejection) == (3, "")
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call.system == SYSTEM
    assert call.output_format is _Draft
    assert [message.content for message in call.messages] == [BRIEFING]


@pytest.mark.asyncio
async def test_parse_validated_rejected_reply_retries_with_the_instruction_and_the_reason() -> None:
    """The model cannot see its own last message as data, so the retry carries both."""
    provider = FakeProvider(responses=[_Draft(value=-1), _Draft(value=4)])

    value, rejection = await _parse(provider)

    assert (value, rejection) == (4, "")
    assert len(provider.calls) == 2
    retry = provider.calls[1].messages
    assert [message.content for message in retry][:1] == [BRIEFING]  # the first turn is kept
    assert len(retry) == 2
    assert retry[1].role is MessageRole.USER
    assert INSTRUCTION in retry[1].content
    assert "value must be positive, got -1" in retry[1].content
    assert '"value": -1' in retry[1].content  # the rejected draft goes back with it


@pytest.mark.asyncio
async def test_parse_validated_refusal_retries_and_feeds_back_that_nothing_was_returned() -> None:
    """`None` is a refusal or a reply truncated before the JSON closed."""
    provider = FakeProvider(responses=[None, _Draft(value=7)])

    value, _ = await _parse(provider)

    assert value == 7
    assert len(provider.calls) == 2
    assert INSTRUCTION in provider.calls[1].messages[1].content


@pytest.mark.asyncio
async def test_parse_validated_every_attempt_invalid_returns_the_last_rejection() -> None:
    provider = FakeProvider(responses=[_Draft(value=-1)] * (DEFAULT_MAX_RETRIES + 1))

    value, rejection = await _parse(provider)

    assert value is None
    assert "value must be positive" in rejection
    assert len(provider.calls) == DEFAULT_MAX_RETRIES + 1
    # One turn added per retry, not one conversation restarted per attempt.
    assert len(provider.calls[-1].messages) == DEFAULT_MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_parse_validated_with_zero_retries_calls_the_provider_once() -> None:
    provider = FakeProvider(responses=[_Draft(value=-1)])

    value, rejection = await _parse(provider, max_retries=0)

    assert value is None
    assert rejection
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_parse_validated_with_negative_retries_rejects_the_wiring() -> None:
    """A bound below zero is a programmer error, as `ExecutionEngine`'s bounds are."""
    provider = FakeProvider(responses=[])

    with pytest.raises(ValueError, match="max_retries"):
        await _parse(provider, max_retries=-1)

    assert provider.calls == []


@pytest.mark.asyncio
async def test_parse_validated_propagates_a_provider_failure_without_retrying() -> None:
    """A transport failure is not unusable output; retrying here would double every outage."""
    provider = FakeProvider(responses=[ProviderError("401 authentication_error")])

    with pytest.raises(ProviderError, match="authentication_error"):
        await _parse(provider)

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_parse_validated_propagates_cancellation() -> None:
    """§10: a run the user cannot stop is a defect, so cancellation must propagate."""
    provider = FakeProvider(responses=[_Draft(value=3)], blocker=asyncio.Event())  # never set

    task = asyncio.create_task(_parse(provider))
    await asyncio.sleep(0)  # let the task reach the provider call
    assert len(provider.calls) == 1
    task.cancel()

    # Bounded: a helper that swallowed the cancellation would hang on the blocker.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
