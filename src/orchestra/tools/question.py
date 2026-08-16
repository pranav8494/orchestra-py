"""Asking the user, surfaced as a tool the model can call (§3.1, #10).

- **The question *is* the schema** — `core.question.Question` is published verbatim as
  `input_schema`, so there is no second params model to drift from the type the `Asker`
  and the renderer already agree on (§1.5, §2.2).
- **The `Asker` is injected** — this module never learns that the answer comes from a
  terminal, which keeps `tools/` below `cli/` (§3.2) and lets a test answer from a list.
- **Declining is not failing** — a blank answer comes back `is_empty`, not `is_error`:
  the user was asked and chose not to say, so a retry would just ask again (§6).
"""

from pydantic import ValidationError

from orchestra.core.question import Asker, Question
from orchestra.tools.base import (
    ToolCall,
    ToolResponse,
    ToolSpec,
    format_validation_error,
)

TOOL_NAME = "ask_user"

# What a decline comes back as. A sentence, not "": every other tool pairs `is_empty` with
# something the model can act on, and an empty `tool_result` block is rejected by the API.
DECLINED = "The user declined to answer. Continue with a stated assumption; do not ask again."

# A prompt (§6). The cost being described is a human turn-around, so most of this is about
# when *not* to call it; the field-level guidance lives on `Question` itself, which the
# model reads as this tool's schema.
DESCRIPTION = (
    "Put one question to the user and wait for their answer. Use it only when the request "
    "is genuinely ambiguous and guessing wrong would waste the whole run — an unstated "
    "period, an undefined metric, a word with two readings here. Do not use it for "
    "anything a tool can find out, for confirmation of a decision you have already made, "
    "or to reword a question that was just answered. Ask the narrowest kind that fits: "
    "yes_no for a decision, single_choice or multi_choice when you can list the options, "
    "free_text only when you cannot. One question per call, in one plain sentence. "
    "Returns the answer as text; an empty answer means the user declined, so continue "
    "with a stated assumption rather than asking again."
)


class AskUserTool:
    """One question to the user, per call. Implements `BaseTool` (§6).

    Stateless: the answer belongs to the caller's ledger (`state.Clarification`), not to
    the tool, so a second call to the same question asks it again.
    """

    def __init__(self, asker: Asker) -> None:
        """Store the injected `Asker`; nothing is asked until `run` (§3.3)."""
        self._asker = asker

    def info(self) -> ToolSpec:
        """See `BaseTool.info`. Pure: it runs on every turn and asks nobody anything."""
        return ToolSpec(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=Question.model_json_schema(),
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        """Ask the question and return the answer as text. See `BaseTool.run`.

        The only thing that may leave here is `CancelledError` from a user who is still at
        the prompt when the run is cancelled (§10).
        """
        try:
            question = Question.model_validate(call.arguments)
        except ValidationError as exc:
            # Includes the choices/kind rule, so a `free_text` question with options comes
            # back as a sentence the model can act on rather than an unhandled error (§6).
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {format_validation_error(exc)}",
                is_error=True,
            )

        answer = await self._asker.ask(question)
        # Whitespace counts as blank: someone who pressed space and Enter declined as much
        # as someone who pressed Enter. `is_empty`, not `is_error` — see the module docstring.
        if not answer.strip():
            return ToolResponse(content=DECLINED, is_empty=True)
        return ToolResponse(content=answer)
