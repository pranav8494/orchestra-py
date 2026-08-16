"""The typed clarification request the planner emits instead of a plan (§3.3, #10).

A question is data, not prose: `kind` decides how `cli/prompt.py` renders it, so nobody
has to parse a sentence to learn that "which quarter?" wanted one of four choices.

The answered half is `state.Clarification` — it lives on the ledger, this does not.
"""

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Enough to disambiguate a request, few enough that answering is not an interview. Also
# the planner's cap: a longer draft is rejected before anyone is prompted.
MAX_QUESTIONS = 3

# Below this a "choice" is not one, and a model that emits a single option has usually
# meant `yes_no`.
MIN_CHOICES = 2

# Above this the list stops being a menu someone can read at a prompt. Also what lets the
# renderer letter every option (A, B, C...) without running out of alphabet.
MAX_CHOICES = 8


class QuestionKind(StrEnum):
    """How the answer is collected. A closed set, so an enum (§7)."""

    YES_NO = "yes_no"
    SINGLE_CHOICE = "single_choice"
    MULTI_CHOICE = "multi_choice"
    FREE_TEXT = "free_text"

    @property
    def needs_choices(self) -> bool:
        """Does this kind pick from a list? Asked by the validator and the renderer."""
        return self in {QuestionKind.SINGLE_CHOICE, QuestionKind.MULTI_CHOICE}


class Question(BaseModel):
    """One thing to ask the user before the run starts.

    This is model output *and* the `ask_user` tool's parameter schema, so every
    `description` here is prompt text (§6). Frozen and `extra="forbid"`: a question is
    asked as written, and an invented field means the schema drifted (§7).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: QuestionKind = Field(description="How the user answers. Choose the narrowest that fits.")
    text: str = Field(
        min_length=1,
        description="The question itself, in one plain sentence.",
    )
    description: str = Field(
        default="",
        description="Optional one-line context shown under the question. Empty for none.",
    )
    choices: list[str] = Field(
        default_factory=list,
        description=(
            f"The options, for single_choice and multi_choice only. Between {MIN_CHOICES} "
            f"and {MAX_CHOICES}, mutually exclusive, and short — the user picks one by "
            "letter, and a long option is one they have to read twice. Empty for every "
            "other kind."
        ),
    )

    @model_validator(mode="after")
    def _check_choices(self) -> "Question":
        """Tie `choices` to `kind`: a choice question with nothing to choose from cannot be
        rendered, and choices on a free-text one would be silently dropped."""
        if self.kind.needs_choices and len(self.choices) < MIN_CHOICES:
            raise ValueError(f"{self.kind} needs at least {MIN_CHOICES} choices")
        if len(self.choices) > MAX_CHOICES:
            raise ValueError(f"{self.kind} takes at most {MAX_CHOICES} choices")
        if not self.kind.needs_choices and self.choices:
            raise ValueError(f"{self.kind} takes no choices")
        duplicates = sorted({choice for choice in self.choices if self.choices.count(choice) > 1})
        if duplicates:
            raise ValueError(f"duplicate choices: {duplicates}")
        return self


class ClarificationRequest(BaseModel):
    """What the planner returns in place of a plan: the run cannot start as asked.

    A model rather than a bare list so the planner's two outcomes are two types the
    caller can branch on without inspecting contents.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: list[Question] = Field(min_length=1, max_length=MAX_QUESTIONS)


class Asker(Protocol):
    """Whoever can put a question to the user. Implemented by `cli/prompt.py` (§7).

    A port, so `core/` and `tools/` never learn that the answer comes from a terminal —
    and a test can answer from a list.
    """

    async def ask(self, question: Question) -> str:
        """Put `question` to the user and return their answer as text.

        Returns the empty string when they declined to answer; the caller decides what
        that means. Never raises for an unusable answer — only cancellation leaves it.
        """
        ...
