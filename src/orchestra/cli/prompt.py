"""The terminal end of a clarification: renders a `Question` and reads the answer (#10).

- **The one module importing `rich.prompt`** — like `render.py` for `rich.live` (§3.1).
  Everything above it holds the `Asker` port and never learns there is a terminal.
- **Prompts are diagnostics** — they go to `err_console`, so a piped run still has only
  its result on stdout (§5). No `Console` is constructed here.
- **The pure parts are separate** — `question_text` and `match_choices` are plain
  functions, so the wording and the matching are tested as data, not as drawing (§12).
"""

from collections.abc import Sequence
from typing import TextIO

from rich.prompt import Confirm, Prompt
from rich.text import Text

from orchestra.cli.console import err_console
from orchestra.core.question import Question, QuestionKind

# Returned by Rich when nothing was entered, and by us when the user declined. One
# spelling for both, because pressing Enter at a question *is* declining.
DECLINED = ""


def question_text(question: Question) -> str:
    """The whole prompt as plain text: the question, its context, and the options Rich
    will not draw itself.

    Rich draws the choices for `single_choice` and the y/n for `yes_no`; `multi_choice` is
    a free-text field, so its options only appear if this puts them there.
    """
    lines = [question.text]
    if question.description:
        lines.append(question.description)
    if question.kind is QuestionKind.MULTI_CHOICE:
        lines.append(f"Options: {', '.join(question.choices)} — separate several with commas")
    return "\n".join(lines)


def question_prompt(question: Question) -> Text:
    """`question_text` as Rich `Text`, with everything below the first line dimmed.

    Constructed, never `Text.from_markup`: the question is model output, and a `[q1]` in
    it would otherwise be eaten as a style tag or raise mid-prompt (§7 — untrusted input).
    """
    body = question_text(question)
    prompt = Text(body)
    context_at = body.find("\n")
    if context_at != -1:
        prompt.stylize("dim", context_at)
    return prompt


def match_choices(answer: str, choices: Sequence[str]) -> str:
    """Map a comma-separated answer onto `choices`, case-insensitively.

    Returns the matched options in the question's own spelling, in the order entered, so
    the planner reads back strings it wrote. Unmatched entries are dropped; an answer that
    matched nothing falls back to itself — someone who typed a fifth option meant it, and
    "" would be recorded as a decline.
    """
    canonical = {choice.lower(): choice for choice in choices}
    matched = [
        canonical[entry]
        for part in answer.split(",")
        if (entry := part.strip().lower()) in canonical
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
                # Rich validates and re-asks; `case_sensitive=False` returns the question's
                # own spelling, so "q2" comes back as "Q2".
                return Prompt.ask(
                    prompt,
                    console=err_console,
                    choices=list(question.choices),
                    case_sensitive=False,
                    default=DECLINED,
                    show_default=False,
                    stream=self._stream,
                )
            case QuestionKind.MULTI_CHOICE:
                # Free text rather than repeated prompts: Rich has no multi-select, and one
                # line the user can review beats N confirmations they cannot revise.
                return match_choices(
                    Prompt.ask(prompt, console=err_console, stream=self._stream),
                    question.choices,
                )
            case QuestionKind.FREE_TEXT:
                return Prompt.ask(prompt, console=err_console, stream=self._stream)
