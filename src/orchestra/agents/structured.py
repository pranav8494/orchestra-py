"""One validate-and-retry path for every structured call (#9).

Structured output guarantees JSON *shape*; `validate` is where the caller checks what a
schema cannot say, and its rejection goes back to the model as the next turn. Unusable
output is retried; a provider outage is not — `ProviderError` propagates.

Whether an exhausted call fails the run is the caller's policy, so nothing is raised here.
"""

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from orchestra.providers.base import MessageRole, Provider, ProviderMessage

# Two extra calls: enough for a model to act on feedback, few enough that a user waiting
# on a run does not pay for a fourth.
DEFAULT_MAX_RETRIES = 2

# A refusal and off-schema JSON are indistinguishable at the adapter, so the reason has
# to fit either. Not prompt text (§11) — the caller's `instruction` says what to do.
_NO_OUTPUT = "No usable structured output was returned."


# `StructuredT` mirrors the port's bound; `ResultT` is whatever the caller's ledger type is.
async def parse_validated[StructuredT: BaseModel, ResultT](
    *,
    provider: Provider,
    system: str,
    messages: Sequence[ProviderMessage],
    output_format: type[StructuredT],
    validate: Callable[[StructuredT], ResultT],
    instruction: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[ResultT | None, str]:
    """Request `output_format` until `validate` accepts it, feeding each rejection back.

    Args:
        validate: converts the draft into the caller's type. Raises `ValueError` (which
            `pydantic.ValidationError` is) when the draft is well-formed but unusable;
            anything else propagates as the programmer error it is.
        instruction: prepended to the rejection on every retry turn.
        max_retries: extra calls after the first.

    Returns:
        The validated value and an empty string, or `None` and the last rejection.

    Raises:
        ValueError: a negative `max_retries` — a wiring bug, not a user-facing error.
        ProviderError: the request itself failed. Retrying output is not retrying an outage.
        asyncio.CancelledError: propagated, never swallowed (§10).
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be at least 0, got {max_retries}")

    turns = list(messages)
    rejection = ""
    for attempt in range(max_retries + 1):
        if attempt:
            turns.append(
                ProviderMessage(role=MessageRole.USER, content=f"{instruction}\n\n{rejection}")
            )
        draft = await provider.parse_structured(
            system=system, messages=turns, output_format=output_format
        )
        if draft is None:
            rejection = _NO_OUTPUT
            continue
        try:
            return validate(draft), ""
        except ValueError as exc:
            # The rejected draft goes back with the reason: the model cannot see its own
            # previous message as data, and "fix this" needs a "this".
            rejection = f"{exc}\n\nThe rejected reply was:\n{draft.model_dump_json(indent=2)}"
    return None, rejection
