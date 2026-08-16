"""The Data Retrieval agent's second tool: keyword lookup over a bundled offline corpus.

**Why a canned corpus.** The whole run must work with no network and no key beyond the
model's own, so "search" here means a few background notes shipped in `data/`. The
description says exactly that (§6) — a tool that implies internet access gets asked for
today's share price, and answers with fiction.

**Why two tools and not one.** Giving the agent a choice is the point: `query_csv` holds
this company's figures, this holds industry context. Each description names the other, so
the boundary is in the prompt the model actually reads rather than in a comment.

**Why the corpus is validated.** A JSON file on disk is a trust boundary like any other
(§7), so it is parsed through a pydantic model on first use. A malformed corpus is
`is_error=True` content naming the file, never a `KeyError` unwinding the agent loop.
"""

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from orchestra.tools.base import ToolCall, ToolResponse, ToolSpec

TOOL_NAME = "search"

# Ceiling on `limit`. Stated in the schema so the model reads it before choosing, rather
# than discovering it in an error: the corpus is small and ten notes is already more
# background than a subtask's answer can use.
MAX_RESULTS = 10

# Words, numbers and apostrophes; everything else is a separator. Keywords are matched as
# whole tokens, so "margin" matches "margins?" but not "marginal" — the corpus carries
# both spellings of a word rather than the matcher guessing at stems.
_TOKEN = re.compile(r"[a-z0-9']+")

# The description is a prompt (§6). Its job is to be honest about the two things this
# tool is not: online, and informed about this company. See `query_csv.DESCRIPTION` for
# why the text lives beside the params model instead of in `prompts/` (§11).
DESCRIPTION = (
    "Search a small offline corpus of background notes bundled with this agent: industry "
    "growth benchmarks, typical margin ranges, what drives a one-quarter cost change, "
    "quarter-over-quarter reporting conventions, seasonality and the macro backdrop. Use "
    "it to interpret or contextualise a figure. It does NOT reach the internet, so it has "
    "no news, prices or anything dated, and it does NOT hold this company's own revenue, "
    "costs or profit — use `query_csv` for those. Returns the best-matching notes, or a "
    "plain message when nothing matches."
)


class SearchParams(BaseModel):
    """The arguments the model may send. Published verbatim as the tool's `input_schema`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="What to look up, in plain words, e.g. 'typical SaaS gross margin'.",
    )
    limit: int = Field(
        default=3,
        gt=0,
        le=MAX_RESULTS,
        description=f"How many notes to return, at most {MAX_RESULTS}.",
    )


class CorpusEntry(BaseModel):
    """One background note as it is stored on disk.

    `keywords` are lowercase single tokens; matching is a set intersection against the
    tokenised query, so the note's own author decides what it answers to.
    """

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)


# One adapter, built at import: compiling the validator per call would be the cost of
# validation without the caching pydantic already offers.
_CORPUS = TypeAdapter(list[CorpusEntry])


class SearchTool:
    """Keyword search over a bundled corpus of background notes. Implements `BaseTool` (§6)."""

    def __init__(self, corpus: Path) -> None:
        """Store the injected corpus path. Nothing is read yet.

        Args:
            corpus: the JSON array of notes, from `app.py` via `agents/toolsets.py`.
        """
        self._corpus = corpus
        self._entries: list[CorpusEntry] | None = None

    def info(self) -> ToolSpec:
        """See `BaseTool.info`. Pure: the corpus is not touched here."""
        return ToolSpec(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=SearchParams.model_json_schema(),
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        """Score the corpus against the query and return the best notes. See `BaseTool.run`."""
        try:
            params = SearchParams.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {_problems(exc)}", is_error=True
            )

        try:
            entries = await self._load()
        except OSError as exc:
            return ToolResponse(
                content=f"Could not read the corpus at {self._corpus}: {exc}", is_error=True
            )
        except ValidationError as exc:
            # Covers both a syntax error and a well-formed file of the wrong shape:
            # `validate_json` reports each as a validation error against `CorpusEntry`.
            return ToolResponse(
                content=f"The corpus at {self._corpus} is malformed: {_problems(exc)}",
                is_error=True,
            )

        matches = _rank(params.query, entries)[: params.limit]
        if not matches:
            # NOT an error. A search that looked and found nothing answered the question
            # correctly; flagging it invites the model to retry the same call, and an
            # `is_error` result reads to it as "the tool broke", not "no such note".
            return ToolResponse(content=_nothing_matched(params.query, entries))
        return ToolResponse(content=_format(params.query, matches))

    async def _load(self) -> list[CorpusEntry]:
        """Read and validate the corpus once, off the event loop (§10).

        Lazy rather than in `__init__`: startup should not pay for a tool the model may
        never call, and a bad path must surface as content to the model, which `__init__`
        has no way to return. Two concurrent first calls may both read the file — the
        work is idempotent and small, so a lock would cost more than the duplicate read.

        Raises:
            OSError: the file could not be read.
            ValidationError: it is not a JSON array of notes.
        """
        if self._entries is None:
            self._entries = await asyncio.to_thread(_read_corpus, self._corpus)
        return self._entries


def _read_corpus(path: Path) -> list[CorpusEntry]:
    """Parse and validate the corpus file. Blocking — call it in a thread."""
    # Bytes, not text: `validate_json` decodes, so invalid UTF-8 arrives as a validation
    # error alongside every other malformed-file case instead of as a separate exception.
    return _CORPUS.validate_json(path.read_bytes())


def _rank(query: str, entries: list[CorpusEntry]) -> list[CorpusEntry]:
    """Entries sharing at least one keyword with the query, best first.

    Score is how many of a note's keywords the query used, so a query naming several
    facets of one note outranks a note caught by a single common word. `sorted` is
    stable and does not reverse ties, so equal scores keep the corpus's own order —
    which makes the top result reproducible instead of dependent on dict ordering.
    """
    tokens = set(_TOKEN.findall(query.lower()))
    scored = [(len(tokens.intersection(entry.keywords)), entry) for entry in entries]
    hits = [pair for pair in scored if pair[0] > 0]
    return [entry for _, entry in sorted(hits, key=lambda pair: pair[0], reverse=True)]


def _format(query: str, matches: list[CorpusEntry]) -> str:
    """Render matches as text the model can read and quote.

    The provenance line is not decoration: without it the model has read a paragraph
    about margins with nothing saying it is generic background rather than our numbers.
    """
    sections = [
        f"{len(matches)} background note(s) matching {query!r}, from the offline corpus "
        f"(generic industry context, not this company's figures):"
    ]
    sections += [
        f"[{position}] {entry.title}\n{entry.snippet}"
        for position, entry in enumerate(matches, start=1)
    ]
    return "\n\n".join(sections)


def _nothing_matched(query: str, entries: list[CorpusEntry]) -> str:
    """Say nothing matched, and list what is there so the retry can be better.

    The titles come from the loaded corpus rather than a hardcoded summary — a list that
    goes stale the first time a note is added is worse than no list.
    """
    return "\n".join(
        [
            f"Nothing in the offline corpus matched {query!r}. It holds only these notes:",
            *(f"- {entry.title}" for entry in entries),
            "Try one of those topics, or use `query_csv` for this company's own figures.",
        ]
    )


def _problems(exc: ValidationError) -> str:
    """Flatten a pydantic error into one line the model can act on.

    Deliberately duplicated from `query_csv.py` (§2.3) — see the note there.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors()
    )
