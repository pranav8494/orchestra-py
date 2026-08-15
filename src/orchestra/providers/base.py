"""The provider port and its factory (CONVENTIONS.md §3.3, §6).

Callers name a model and a message list; what a vendor calls those, and which
exceptions it raises, stops here. A provider returns our types or raises from the
taxonomy in `core/errors.py` — never an SDK type.

**Only structured parsing, for now.** §6 sketches `send()`/`stream()` as well, and they
arrive with the worker agents (#5-#7) that need a free-text turn and a token stream.
Declaring them here first would force every fake and every adapter to implement methods
nothing calls, so the port grows when the first caller does.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import BaseModel, SecretStr

# Bound to BaseModel because structured output is a trust boundary: the schema the
# model fills in and the validation of what comes back are the same object (§7).
StructuredT = TypeVar("StructuredT", bound=BaseModel)


class MessageRole(StrEnum):
    """Who authored a turn. A closed set, so an enum rather than bare strings (§7)."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    """One conversation turn — an internal value object, not a trust boundary (§7)."""

    role: MessageRole
    content: str


class Provider(Protocol):
    """A model provider. Implemented once per vendor, plus `FakeProvider` in tests."""

    @property
    def model(self) -> str:
        """The model identifier requests are sent to."""
        ...

    async def parse_structured(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        output_format: type[StructuredT],
    ) -> StructuredT | None:
        """Ask the model for a response matching `output_format`.

        Args:
            system: the system prompt, from `orchestra.prompts`.
            messages: the conversation so far, oldest first.
            output_format: the pydantic model the response must populate.

        Returns:
            The validated model, or `None` when the response carried no usable
            structured output — a refusal, or a reply truncated before the JSON
            closed. Callers decide whether that is worth another attempt; it is not
            an error the provider can fix.

        Raises:
            ProviderError: the request itself failed (auth, transport, rate limit).
        """
        ...

    async def aclose(self) -> None:
        """Release the connections the provider holds. Idempotent.

        On the port because the sockets are the vendor's, so only the adapter can close
        them and no caller can reach past `Provider` to do it. A process that exits drops
        them anyway; a long-lived one, or a test suite that turns `ResourceWarning` into
        an error, does not. Callers should use `contextlib.aclosing`.
        """
        ...


def create_provider(*, api_key: SecretStr, model: str) -> Provider:
    """Build the provider for `model` — the one place a vendor is chosen (§3.3).

    Args:
        api_key: the vendor credential, kept in a `SecretStr` so it cannot leak
            through a repr or a log record until the adapter unwraps it (§9).
        model: the model identifier, from `Config.anthropic_model`.

    Returns:
        A `Provider`. Adding a vendor means one module and one branch here.
    """
    # Imported here rather than at module scope so the port has no import edge to the
    # SDK: `agents/` importing `Provider` must not pull `anthropic` into the process.
    from orchestra.providers.anthropic import AnthropicProvider

    return AnthropicProvider(api_key=api_key, model=model)
