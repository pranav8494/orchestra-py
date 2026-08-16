"""Tests for the console asker.

Answers are scripted through `stream=`, Rich's own seam for input, so nothing here touches
a terminal or real stdin (§12). What is asserted is the string handed back and the stream
the prompt landed on — never what Rich drew.
"""

import io
from typing import TextIO, cast

import pytest

from orchestra.cli.prompt import ConsoleAsker, match_choices, question_text, resolve_choice
from orchestra.core.question import Question, QuestionKind

CHOICES = ["Q1", "Q2", "Q3"]


class ClosedStdin:
    """stdin at EOF, standing in for the production path where Rich falls back to the
    builtin `input()` and it raises. Not a `StringIO` subclass: `readline` is typed
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


def test_question_text_multi_choice_says_several_are_allowed() -> None:
    """The one thing the lettered menu does not say by itself."""
    text = question_text(
        Question(kind=QuestionKind.MULTI_CHOICE, text="Which quarters?", choices=CHOICES)
    )

    assert "commas" in text


@pytest.mark.parametrize("kind", [QuestionKind.SINGLE_CHOICE, QuestionKind.MULTI_CHOICE])
def test_question_text_letters_every_option_on_its_own_line(kind: QuestionKind) -> None:
    """Rich's inline `[Q1/Q2/Q3]` is only typeable when the options are one word. A live
    run rejected "all", "all three" and "2024" against options like "All three (revenue,
    costs, and profit)" — the letter is what the user actually types."""
    text = question_text(Question(kind=kind, text="Which quarter?", choices=CHOICES))

    assert text.startswith("Which quarter?\n")
    for line, (letter, choice) in zip(
        text.splitlines()[1:], zip("ABC", CHOICES, strict=True), strict=False
    ):
        assert line == f"  {letter}. {choice}"


@pytest.mark.parametrize(
    ("kind", "tail"),
    [
        (QuestionKind.SINGLE_CHOICE, "Answer A-C"),
        (QuestionKind.MULTI_CHOICE, "Answer A-C, or several separated by commas"),
    ],
)
def test_question_text_ends_on_a_line_for_the_answer(kind: QuestionKind, tail: str) -> None:
    """Rich appends its ": " to the last line, so without this the caret sits after an
    option and that option reads as the question."""
    text = question_text(Question(kind=kind, text="Which quarter?", choices=CHOICES))

    assert text.endswith(f"\n{tail}")


def test_resolve_choice_takes_the_letter_beside_the_option() -> None:
    """The point of the menu: "B" is what a user types when the option is a sentence."""
    assert resolve_choice("b", CHOICES) == "Q2"
    assert resolve_choice(" C ", CHOICES) == "Q3"


def test_resolve_choice_still_takes_the_option_text() -> None:
    assert resolve_choice("q3", CHOICES) == "Q3"


def test_resolve_choice_prefers_the_option_over_the_letter_that_collides() -> None:
    """Options spelled "A" and "B" answer to themselves, not to each other's letters."""
    assert resolve_choice("B", ["A", "B"]) == "B"


def test_resolve_choice_returns_nothing_for_an_entry_that_names_neither() -> None:
    assert resolve_choice("Q9", CHOICES) == ""


def test_match_choices_takes_letters_and_texts_together() -> None:
    assert match_choices("A, Q3", CHOICES) == "Q1, Q3"


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
    """Someone who typed a fifth option meant it; "" would be recorded as a decline."""
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
async def test_console_asker_single_choice_takes_the_letter() -> None:
    """End to end through the real Rich prompt: a letter is accepted first time and comes
    back as the option, so a sentence-long option costs one keystroke."""
    long_options = ["Revenue", "Costs", "All three (revenue, costs, and profit)"]
    question = Question(kind=QuestionKind.SINGLE_CHOICE, text="Which metric?", choices=long_options)

    assert await ConsoleAsker(io.StringIO("c\n")).ask(question) == long_options[2]


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
async def test_console_asker_multi_choice_takes_letters() -> None:
    question = Question(kind=QuestionKind.MULTI_CHOICE, text="Which quarters?", choices=CHOICES)

    assert await ConsoleAsker(io.StringIO("a, c\n")).ask(question) == "Q1, Q3"


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
    """Without a default, Rich re-asks a `yes_no` or `single_choice` forever and the user's
    only way out is Ctrl-C. `default=DECLINED` ends it; the tool reports "" as `is_empty`.

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
