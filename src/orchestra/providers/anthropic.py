"""The Anthropic adapter — the only module in the codebase that imports `anthropic`.

Two things happen at this boundary and nowhere else: SDK failures become `ProviderError`
(exit 4, §8), and the SDK's message shape becomes ours. Note the module is
`orchestra.providers.anthropic`; `import anthropic` below is absolute and resolves to
the vendor package, not to this file.
"""

from collections.abc import Sequence
from typing import Literal, assert_never

import anthropic
from anthropic.types import MessageParam
from pydantic import SecretStr, ValidationError

from orchestra.core.errors import ProviderError
from orchestra.providers.base import MessageRole, ProviderMessage, StructuredT

# claude-opus-5 thinks by default and `max_tokens` caps thinking *and* output together,
# so a budget sized for the answer alone truncates the JSON and yields no parsed output.
DEFAULT_MAX_TOKENS = 16_000


class AnthropicProvider:
    """Talks to the Anthropic Messages API. Constructed by `create_provider` (§3.3).

    Deliberately sends no `temperature`, `top_p`, `top_k` or `thinking` block: the
    default model rejects all four with a 400.
    """

    def __init__(
        self, *, api_key: SecretStr, model: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        """Create the client.

        Args:
            api_key: the credential, unwrapped here and nowhere above (§9).
            model: model identifier, from `Config.anthropic_model`.
            max_tokens: combined thinking + output budget for one request.
        """
        self._client = anthropic.AsyncAnthropic(api_key=api_key.get_secret_value())
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        """The model identifier requests are sent to."""
        return self._model

    async def parse_structured(
        self,
        *,
        system: str,
        messages: Sequence[ProviderMessage],
        output_format: type[StructuredT],
    ) -> StructuredT | None:
        """Send one structured-output request. See `Provider.parse_structured`."""
        try:
            response = await self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[_to_sdk_message(message) for message in messages],
                output_format=output_format,
            )
        except anthropic.AnthropicError as exc:
            # The SDK's own base class, so transport, status and retryable failures are
            # all covered without a bare `except` (§8). Cancellation is a BaseException
            # and passes straight through (§10).
            raise ProviderError(f"Anthropic request to {self._model} failed: {exc}") from exc
        except ValidationError:
            # The SDK validates the text block against `output_format`, so a reply that
            # is JSON but not *this* schema lands here. For the caller that is the same
            # condition as a refusal — no usable structured output — so it gets the same
            # answer rather than a second error path that means the same thing.
            return None
        return response.parsed_output


def _to_sdk_message(message: ProviderMessage) -> MessageParam:
    """Translate one turn into the SDK's shape. The reverse never happens: callers get
    parsed models, so no SDK type crosses back out (§6)."""
    # Exhaustive rather than an else-branch: the SDK types `role` as a literal, and a
    # fourth `MessageRole` member must fail under mypy here, not be silently relabelled
    # `user` at runtime. `assert_never` is what makes that a compile-time guarantee.
    role: Literal["user", "assistant"]
    match message.role:
        case MessageRole.USER:
            role = "user"
        case MessageRole.ASSISTANT:
            role = "assistant"
        case unhandled:
            assert_never(unhandled)
    return MessageParam(role=role, content=message.content)
