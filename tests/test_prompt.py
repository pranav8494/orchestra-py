"""Tests for the console asker.

Answers are scripted through `stream=`, Rich's own seam for input, so nothing here touches
a terminal or real stdin (§12). What is asserted is the string handed back and the stream
the prompt landed on — never what Rich drew.
"""

import io
from typing import TextIO, cast

import pytest

from orchestra.cli.prompt import ConsoleAsker, match_choices, question_text
from orchestra.core.question import Question, QuestionKind

CHOICES = ["Q1", "Q2", "Q3"]


class ClosedStdin:
    """stdin at EOF. Rich reads through `stream.readline()`, and without a stream it reads
    through the builtin `input()`, which raises `EOFError` — so this stands in for the
    production path the guard exists for. Not a `StringIO` subclass: `readline` is typed
    `bytes` on `IOBase`, and only the one method is ever called."""

    def readline(self) -> str:
        raise EOFError


# ------------------------------------------------------------------ the pure helpers


def test_question_text_free_text_is_just_the_question() -> None:
    text = question_text(Question(kind=QuestionKind.FREE_TEXT, text="Which years?"))

    assert text == "Which years?"


def test_question_text_description_becomes_a_second_line() -> None:
    text = question_text(
        Question(
            kind=QuestionKind.FREE_TEXT,
            text="Which years?",
            description="The report covers one period.",
        )
    )

    assert text == "Which years?\nThe report covers one period."


def test_question_text_multi_choice_lists_its_options() -> None:
    """Rich draws the options for `single_choice` but not for a free-text field, so
    `multi_choice` would otherwise be an unanswerable question."""
    text = question_text(
        Question(kind=QuestionKind.MULTI_CHOICE, text="Which quarters?", choices=CHOICES)
    )

    assert "Q1, Q2, Q3" in text
    assert "commas" in text


def test_question_text_single_choice_leaves_the_options_to_rich() -> None:
    """Listing them here as well would print them twice."""
    text = question_text(
        Question(kind=QuestionKind.SINGLE_CHOICE, text="Which quarter?", choices=CHOICES)
    )

    assert text == "Which quarter?"


def test_match_choices_exact_entries_return_the_choices() -> None:
    assert match_choices("Q1, Q3", CHOICES) == "Q1, Q3"


def test_match_choices_is_case_insensitive_and_returns_the_question_spelling() -> None:
    """The planner reads the answer back, so it must get strings it wrote."""
    assert match_choices("q1,  q2 ", CHOICES) == "Q1, Q2"


def test_match_choices_keeps_the_order_entered() -> None:
    assert match_choices("Q3, Q1", CHOICES) == "Q3, Q1"


def test_match_choices_drops_an_unknown_entry_beside_a_known_one() -> None:
    assert match_choices("Q1, Q9", CHOICES) == "Q1"


def test_match_choices_repeated_entry_appears_once() -> None:
    assert match_choices("Q1, q1", CHOICES) == "Q1"


def test_match_choices_nothing_matched_falls_back_to_the_raw_answer() -> None:
    """Someone who typed a fifth option meant it; returning "" would be recorded as a
    decline and the planner would never see what they said."""
    assert match_choices("  the first half  ", CHOICES) == "the first half"


def test_match_choices_empty_answer_stays_empty() -> None:
    assert match_choices("", CHOICES) == ""


# ------------------------------------------------------------------ one test per kind


@pytest.mark.asyncio
async def test_console_asker_yes_no_returns_yes() -> None:
    """Normalised to a word, not "True": the planner reads this as the user's answer."""
    question = Question(kind=QuestionKind.YES_NO, text="Include 2024?")

    assert await ConsoleAsker(io.StringIO("y\n")).ask(question) == "yes"


@pytest.mark.asyncio
async def test_console_asker_yes_no_returns_no() -> None:
    question = Question(kind=QuestionKind.YES_NO, text="Include 2024?")

    assert await ConsoleAsker(io.StringIO("n\n")).ask(question) == "no"


@pytest.mark.asyncio
async def test_console_asker_single_choice_returns_the_question_spelling() -> None:
    """Matched case-insensitively, answered in the question's own casing."""
    question = Question(kind=QuestionKind.SINGLE_CHOICE, text="Which quarter?", choices=CHOICES)

    assert await ConsoleAsker(io.StringIO("q2\n")).ask(question) == "Q2"


@pytest.mark.asyncio
async def test_console_asker_single_choice_reasks_after_an_invalid_option() -> None:
    """Rich owns the re-ask; this pins that we let it, rather than returning the junk."""
    question = Question(kind=QuestionKind.SINGLE_CHOICE, text="Which quarter?", choices=CHOICES)

    assert await ConsoleAsker(io.StringIO("Q9\nQ3\n")).ask(question) == "Q3"


@pytest.mark.asyncio
async def test_console_asker_multi_choice_returns_the_matched_choices() -> None:
    question = Question(kind=QuestionKind.MULTI_CHOICE, text="Which quarters?", choices=CHOICES)

    assert await ConsoleAsker(io.StringIO("q3, q1\n")).ask(question) == "Q3, Q1"


@pytest.mark.asyncio
async def test_console_asker_multi_choice_unmatched_text_comes_back_verbatim() -> None:
    question = Question(kind=QuestionKind.MULTI_CHOICE, text="Which quarters?", choices=CHOICES)

    assert await ConsoleAsker(io.StringIO("the first half\n")).ask(question) == "the first half"


@pytest.mark.asyncio
async def test_console_asker_free_text_returns_what_was_typed() -> None:
    question = Question(kind=QuestionKind.FREE_TEXT, text="Which years?")

    assert await ConsoleAsker(io.StringIO("2024 and 2025\n")).ask(question) == "2024 and 2025"


# ------------------------------------------------------------------ declining and streams


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [QuestionKind.YES_NO, QuestionKind.SINGLE_CHOICE, QuestionKind.MULTI_CHOICE],
    ids=str,
)
async def test_console_asker_a_user_who_answers_nothing_declines(kind: QuestionKind) -> None:
    """Without a default, Rich re-asks a `yes_no` or `single_choice` forever, and a user
    who cannot answer has no way out but Ctrl-C. `default=DECLINED` ends it: no answer
    means "no answer", which the tool reports as `is_empty`.

    Reached here through the exhausted stream rather than the bare newline — Rich's
    `stream` path does not strip it, where the builtin `input()` production uses does, so
    a real Enter takes the default one iteration sooner than this does.
    """
    choices = CHOICES if kind.needs_choices else []
    question = Question(kind=kind, text="Which quarter?", choices=choices)

    assert await ConsoleAsker(io.StringIO("\n")).ask(question) == ""


@pytest.mark.asyncio
async def test_console_asker_closed_stdin_returns_an_empty_answer() -> None:
    """A closed stdin is a decline, not a crash: the CLI may be run in a pipeline where
    nobody is there to answer (§8 — no traceback for a foreseeable condition)."""
    question = Question(kind=QuestionKind.FREE_TEXT, text="Which years?")

    assert await ConsoleAsker(cast("TextIO", ClosedStdin())).ask(question) == ""


@pytest.mark.asyncio
async def test_console_asker_writes_the_prompt_to_stderr_and_nothing_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§5: stdout carries the run's result alone, so a prompt on it would corrupt
    `--output json`. Asserted on both streams separately, as §12 requires."""
    question = Question(
        kind=QuestionKind.FREE_TEXT, text="Which years?", description="One period only."
    )

    await ConsoleAsker(io.StringIO("2025\n")).ask(question)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Which years?" in captured.err
    assert "One period only." in captured.err
