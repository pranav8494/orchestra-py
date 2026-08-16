"""The terminal end of a clarification: renders a `Question` and reads the answer (#10).

- **The one module importing `rich.prompt`** — like `render.py` for `rich.live` (§3.1).
  Everything above it holds the `Asker` port and never learns there is a terminal.
- **Prompts are diagnostics** — they go to `err_console`, so a piped run still has only
  its result on stdout (§5). No `Console` is constructed here.
- **The pure parts are separate** — `question_text` and `match_choices` are plain
  functions, so the wording and the matching are tested as data, not as drawing (§12).
"""

from collections.abc import Sequence
from string import ascii_uppercase
from typing import TextIO

from rich.prompt import Confirm, Prompt
from rich.text import Text

from orchestra.cli.console import err_console
from orchestra.core.question import Question, QuestionKind

# Returned by Rich when nothing was entered, and by us when the user declined. One
# spelling for both, because pressing Enter at a question *is* declining.
DECLINED = ""


def choice_letters(choices: Sequence[str]) -> list[str]:
    """`["A", "B", ...]`, one per option. Bounded by `Question`'s `MAX_CHOICES`."""
    return list(ascii_uppercase[: len(choices)])


def question_text(question: Question) -> str:
    """The whole prompt as plain text: the question, its context, and its options lettered
    one per line.

    Lettered and stacked rather than left to Rich's inline `[Revenue/Costs/...]`, which is
    only an answer someone can type when the options are one word. A live run had a user
    rejected three times for "all", "all three" and "2024" against options like "All three
    (revenue, costs, and profit)".
    """
    lines = [question.text]
    if question.description:
        lines.append(question.description)
    letters = choice_letters(question.choices)
    lines.extend(
        f"  {letter}. {choice}" for letter, choice in zip(letters, question.choices, strict=True)
    )
    if letters:
        # A line of its own, because Rich appends its ": " to the last one: without this
        # the caret sits after an option, which reads as though that option is the question.
        span = f"{letters[0]}-{letters[-1]}"
        lines.append(
            f"Answer {span}, or several separated by commas"
            if question.kind is QuestionKind.MULTI_CHOICE
            else f"Answer {span}"
        )
    return "\n".join(lines)


def question_prompt(question: Question) -> Text:
    """`question_text` as Rich `Text`, with the context line dimmed.

    Constructed, never `Text.from_markup`: the question is model output, and a `[q1]` in
    it would otherwise be eaten as a style tag or raise mid-prompt (§7 — untrusted input).
    Only the description dims; the options are what the user is reading.
    """
    prompt = Text(question_text(question))
    if question.description:
        context_at = len(question.text) + 1
        prompt.stylize("dim", context_at, context_at + len(question.description))
    return prompt


def resolve_choice(answer: str, choices: Sequence[str]) -> str:
    """The option `answer` names — its own text or the letter beside it — or `""`.

    Both, because either is a reasonable thing to type in front of a lettered menu. The
    option's own spelling wins over the letter, so a question whose options are literally
    "A" and "B" still answers to them.
    """
    entry = answer.strip()
    for choice in choices:
        if entry.lower() == choice.lower():
            return choice
    letters = choice_letters(choices)
    return choices[letters.index(entry.upper())] if entry.upper() in letters else DECLINED


def match_choices(answer: str, choices: Sequence[str]) -> str:
    """Map a comma-separated answer onto `choices`, by letter or by text.

    Returns the matched options in the question's own spelling, in the order entered, so
    the planner reads back strings it wrote. Unmatched entries are dropped; an answer that
    matched nothing falls back to itself — someone who typed a fifth option meant it, and
    "" would be recorded as a decline.
    """
    matched = [
        resolved for part in answer.split(",") if (resolved := resolve_choice(part, choices))
    ]
    # Deduplicated: `choices` is unique by the model validator, but "a, a" is not.
    return ", ".join(dict.fromkeys(matched)) if matched else answer.strip()


class ConsoleAsker:
    """Puts a `Question` to the terminal. Implements `core.question.Asker` (§7)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        """`stream` is Rich's own seam for scripted input (`Prompt.ask(stream=...)`).

        `None` — the default, and the only value production passes — reads real stdin;
        tests hand in a `StringIO`, so the suite never touches a terminal (§12).
        """
        self._stream = stream

    async def ask(self, question: Question) -> str:
        """Ask `question` and return the answer, or "" if the user declined.

        Async by the `Asker` contract, but the prompt below blocks — a deliberate §10
        deviation: `asyncio.to_thread` uses non-daemon threads, so a Ctrl-C while the user
        sits at the prompt could not end the process. Blocking directly lets
        `KeyboardInterrupt` reach the CLI boundary and exit 130 (§8).

        The trade only holds at a pre-engine call site, where the sole other task on the
        loop is a dashboard consumer parked on its queue with no region open. Offering
        this tool to a worker's model (#12) would freeze the `Live` region and every
        concurrent subtask, and needs a different answer.
        """
        try:
            return self._read(question)
        except (EOFError, OSError):
            # stdin closed — a pipe, or Ctrl-D — or stderr went away mid-prompt. Declining,
            # not a crash: a dead stream costs the interaction, never the run (§5).
            return DECLINED

    def _read(self, question: Question) -> str:
        """Blocking. Split out so the EOF guard wraps every kind exactly once."""
        prompt = question_prompt(question)
        match question.kind:
            case QuestionKind.YES_NO:
                # `default=DECLINED` so Enter declines instead of re-asking forever: Rich
                # returns the default untouched, and a str here is how we tell it apart
                # from an answered `False`.
                answer = Confirm.ask(
                    prompt,
                    console=err_console,
                    default=DECLINED,
                    show_default=False,
                    stream=self._stream,
                )
                if isinstance(answer, str):
                    return DECLINED
                # Normalised to words, not "True"/"False": this string is read back by the
                # planner as the user's answer.
                return "yes" if answer else "no"
            case QuestionKind.SINGLE_CHOICE:
                # Letters *and* texts, so Rich accepts either and still owns the re-ask.
                # `show_choices=False`: the menu is already in the prompt, and Rich's
                # inline bracket would repeat it in the form that was hard to type.
                answer = Prompt.ask(
                    prompt,
                    console=err_console,
                    choices=[*choice_letters(question.choices), *question.choices],
                    case_sensitive=False,
                    show_choices=False,
                    default=DECLINED,
                    show_default=False,
                    stream=self._stream,
                )
                return resolve_choice(answer, question.choices)
            case QuestionKind.MULTI_CHOICE:
                # Free text rather than repeated prompts: Rich has no multi-select, and one
                # line the user can review beats N confirmations they cannot revise.
                return match_choices(
                    Prompt.ask(prompt, console=err_console, stream=self._stream),
                    question.choices,
                )
            case QuestionKind.FREE_TEXT:
                return Prompt.ask(prompt, console=err_console, stream=self._stream)
