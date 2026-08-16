"""Tests for the clarification schema (#10).

The validator is the contract the renderer and the `ask_user` tool both rely on, so it is
asserted here rather than through either consumer.
"""

import pytest
from pydantic import ValidationError

from orchestra.core.question import (
    MAX_QUESTIONS,
    ClarificationRequest,
    Question,
    QuestionKind,
)

QUESTION = Question(kind=QuestionKind.FREE_TEXT, text="Which period should it cover?")


@pytest.mark.parametrize("kind", [QuestionKind.SINGLE_CHOICE, QuestionKind.MULTI_CHOICE])
def test_question_a_choice_kind_without_enough_choices_is_rejected(kind: QuestionKind) -> None:
    """One option is not a choice, and none is unrenderable."""
    with pytest.raises(ValidationError, match="at least 2 choices"):
        Question(kind=kind, text="Which metric?", choices=["revenue"])


@pytest.mark.parametrize("kind", [QuestionKind.FREE_TEXT, QuestionKind.YES_NO])
def test_question_a_non_choice_kind_carrying_choices_is_rejected(kind: QuestionKind) -> None:
    """Accepting them would drop them at the prompt, unseen by the user."""
    with pytest.raises(ValidationError, match="takes no choices"):
        Question(kind=kind, text="Which metric?", choices=["revenue", "profit"])


def test_question_duplicate_choices_are_rejected() -> None:
    """The same option twice is a menu the user cannot answer unambiguously."""
    with pytest.raises(ValidationError, match="duplicate choices"):
        Question(
            kind=QuestionKind.SINGLE_CHOICE,
            text="Which metric?",
            choices=["revenue", "revenue", "profit"],
        )


def test_question_is_frozen_so_what_was_asked_is_what_was_answered() -> None:
    with pytest.raises(ValidationError):
        QUESTION.text = "something else"  # type: ignore[misc]  # frozen by design


def test_clarification_request_needs_at_least_one_question() -> None:
    with pytest.raises(ValidationError):
        ClarificationRequest(questions=[])


def test_clarification_request_caps_the_round() -> None:
    """An interview is not a clarification round (§10 — bound every loop)."""
    with pytest.raises(ValidationError):
        ClarificationRequest(questions=[QUESTION] * (MAX_QUESTIONS + 1))
