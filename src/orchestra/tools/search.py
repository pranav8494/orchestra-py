"""The Data Retrieval agent's second tool: web search, or a bundled corpus without a key.

- **Two backends, one tool** — which one answers is the operator's deployment choice, not
  the model's.
- **The corpus is the floor, not a stub** — it keeps a run reproducible with no key and no
  network, so a failed live request falls back and says so rather than failing the subtask.
- **Provenance is labelled** — corpus notes carry invented specifics, live results carry a
  URL. A figure in the report must trace to something.

Both backends are trust boundaries, so both are validated through pydantic (§7).
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from orchestra.tools.base import (
    ToolCall,
    ToolResponse,
    ToolSpec,
    format_validation_error,
)

TOOL_NAME = "search"

# One live provider, not configurable: a second would need a second response shape, and
# that choice would belong in `agents/toolsets.py`.
TAVILY_URL = "https://api.tavily.com/search"

# Bounded because the agent loop is bounded (§10): a hung request would spend the
# subtask's wall clock without spending its token budget.
DEFAULT_TIMEOUT = 10.0

# The whole anti-hallucination contract: one set of notes may be quoted as fact, the
# other may not.
_CORPUS_PROVENANCE = (
    "from the bundled offline corpus (illustrative sample data, not this company's "
    "figures and not sourced research — do not quote its numbers as fact)"
)
_LIVE_PROVENANCE = "from a live web search (each result cites the page it came from)"

# Ceiling on `limit`, stated in the schema so the model reads it before choosing rather
# than discovering it in an error.
MAX_RESULTS = 10

# Whole-token matching: "margin" matches "margins?" but not "marginal". The corpus carries
# both spellings of a word rather than the matcher guessing at stems.
_TOKEN = re.compile(r"[a-z0-9']+")

# A prompt (§6). One description covers both backends, so it promises background context
# rather than live data it may not have. See `query_csv.DESCRIPTION` for why it lives
# beside the params model instead of in `prompts/` (§11).
DESCRIPTION = (
    "Search for background and industry context: growth benchmarks, typical margin "
    "ranges, what drives a one-quarter cost change, reporting conventions, seasonality, "
    "the macro backdrop. Use it to interpret or contextualise a figure. It does NOT hold "
    "this company's own revenue, costs or profit — use `query_csv` for those. Results say "
    "where they came from; treat anything marked illustrative as context, not as fact to "
    "quote. Returns the best-matching notes, or a plain message when nothing matches."
)

# What the planner is told this puts within reach (`ToolSpec.provides`). Written as prose
# and no figures, because that is what it returns: a plan that needs numbers to compute
# with cannot be built on this one.
PROVIDES = (
    "written background and industry context from the web — definitions, benchmarks, "
    "market backdrop; prose, never a table of figures to compute over"
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


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One result, whichever backend produced it — so both render the same way.

    A dataclass, not a model: built from already-validated input on both sides, so it is
    an internal value object (§7).
    """

    title: str
    snippet: str
    source: str = ""  # a URL from the live backend; empty for a bundled note


class CorpusEntry(BaseModel):
    """One background note as it is stored on disk.

    `keywords` are single tokens, matched as a set intersection against the tokenised
    query — the note's own author decides what it answers to.
    """

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)

    @field_validator("keywords")
    @classmethod
    def _lowercase(cls, keywords: list[str]) -> list[str]:
        """Fold keywords to lowercase, because `_rank` matches a lowercased token set.

        Normalised rather than rejected: `"Margin"` is a typo that would silently never
        match, and failing the whole tool over it costs the model every other note too.
        """
        return [keyword.lower() for keyword in keywords]

    def as_result(self) -> SearchResult:
        """Render this note in the shape both backends share. No source: it has no URL."""
        return SearchResult(title=self.title, snippet=self.snippet)


# Built at import: compiling the validator per call would cost validation without the
# caching pydantic already offers.
_CORPUS = TypeAdapter(list[CorpusEntry])


class _LiveResult(BaseModel):
    """One result from the live backend, as much of it as this tool uses.

    `extra="ignore"`, unlike every other model here: the payload is a vendor's, and
    forbidding new fields would turn a routine API addition into a broken tool. The
    opposite of the rule for model output, where an extra field means our schema drifted.
    """

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    url: str = ""
    content: str = ""


class _LiveResponse(BaseModel):
    """The live backend's reply. Only `results` is read; the rest is metadata."""

    model_config = ConfigDict(extra="ignore")

    results: list[_LiveResult] = Field(default_factory=list)


class _LiveSearchError(Exception):
    """The live backend could not answer.

    Never escapes `run`: it becomes the notice on a fallback to the corpus, because an
    outage should not end a subtask that has a working offline source (§6).
    """


class SearchTool:
    """Web search, falling back to a bundled corpus. Implements `BaseTool` (§6)."""

    def __init__(
        self,
        corpus: Path,
        *,
        api_key: SecretStr | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Store the injected corpus path and, if there is one, the live backend's key.

        `api_key` is `None` to search the corpus only. Injected, not read from the
        environment: only `config.py` may do that, and a tool that read it could not be
        pointed at a fixture (§6, §9).
        """
        self._corpus = corpus
        self._api_key = api_key
        self._timeout = timeout
        self._entries: list[CorpusEntry] | None = None

    def info(self) -> ToolSpec:
        """See `BaseTool.info`. Pure: no disk, no network."""
        return ToolSpec(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=SearchParams.model_json_schema(),
            provides=PROVIDES,
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        """Search the web if a key was configured, else the corpus. See `BaseTool.run`."""
        try:
            params = SearchParams.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {format_validation_error(exc)}",
                is_error=True,
            )

        warning = ""
        if self._api_key is not None:
            try:
                results = await self._live(params)
            except _LiveSearchError as exc:
                # Degraded, not failed: the corpus can still answer. Reported twice on
                # purpose — in `content` for the model, on `warning` for the agent, which
                # surfaces it to the operator who can fix the key or the network.
                warning = f"Live search was unavailable: {exc}. Answered from the corpus."
            else:
                if not results:
                    return ToolResponse(content=_nothing_found(params.query), is_empty=True)
                return ToolResponse(content=_format(params.query, results, _LIVE_PROVENANCE))

        return await self._corpus_search(params, warning)

    async def _corpus_search(self, params: "SearchParams", warning: str) -> ToolResponse:
        """Rank the bundled notes. The offline path, and the live path's fallback.

        `warning` is empty on the offline path — nothing degraded, this *is* the backend.
        """
        notice = f"({warning})\n\n" if warning else ""
        try:
            entries = await self._load()
        except OSError as exc:
            return ToolResponse(
                content=f"{notice}Could not read the corpus at {self._corpus}: {exc}",
                is_error=True,
                warning=warning,
            )
        except ValidationError as exc:
            # Covers a syntax error too: `validate_json` reports both as validation errors.
            return ToolResponse(
                content=f"{notice}The corpus at {self._corpus} is malformed: {format_validation_error(exc)}",
                is_error=True,
                warning=warning,
            )

        matches = _rank(params.query, entries)[: params.limit]
        if not matches:
            # Neither an error nor a result: a search that found nothing answered
            # correctly. `is_error` would read as "the tool broke" and invite the same
            # call again; `is_empty` stops a caller counting it as retrieved (§6).
            return ToolResponse(
                content=notice + _nothing_matched(params.query, entries),
                is_empty=True,
                warning=warning,
            )
        results = [entry.as_result() for entry in matches]
        return ToolResponse(
            content=notice + _format(params.query, results, _CORPUS_PROVENANCE), warning=warning
        )

    async def _live(self, params: "SearchParams") -> list["SearchResult"]:
        """Ask the live backend, and validate what comes back (§7).

        A client per call: `BaseTool` has no close hook, so a pooled one would leak its
        sockets, and an agent makes a handful of searches per run.

        Raises:
            _LiveSearchError: the request failed, was refused, or came back the wrong
                shape. All three mean the same thing to `run`: use the corpus.
        """
        # Unwrapped in exactly one place (§9), before the request so mypy narrows it.
        key = self._api_key.get_secret_value() if self._api_key is not None else ""
        payload = {
            "query": params.query,
            "max_results": params.limit,
            "search_depth": "basic",  # 1 credit; `advanced` costs 2 and reranks
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    TAVILY_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = _LiveResponse.model_validate_json(response.content)
        except httpx.HTTPStatusError as exc:
            # The status only, never the body: an auth failure echoes the request, and
            # the request carries the key (§9).
            raise _LiveSearchError(f"HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            # Timeouts, DNS, resets. httpx's own base class, so not a bare `except` (§8);
            # `CancelledError` is a BaseException and passes through (§10).
            raise _LiveSearchError(f"{type(exc).__name__}") from exc
        except ValidationError as exc:
            raise _LiveSearchError(
                f"unexpected response shape ({format_validation_error(exc)})"
            ) from exc

        return [
            SearchResult(title=item.title or item.url, snippet=item.content, source=item.url)
            for item in body.results
            if item.content
        ]

    async def _load(self) -> list[CorpusEntry]:
        """Read and validate the corpus once, off the event loop (§10).

        Lazy, not in `__init__`: a bad path must reach the model as content, which
        `__init__` cannot return. Two concurrent first calls may both read the file — the
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
    # error alongside every other malformed-file case.
    return _CORPUS.validate_json(path.read_bytes())


def _rank(query: str, entries: list[CorpusEntry]) -> list[CorpusEntry]:
    """Entries sharing at least one keyword with the query, best first.

    Score is how many of a note's keywords the query used, so a query naming several
    facets of one note outranks one caught by a single common word. Ties keep the corpus's
    own order — `sorted` is stable — which makes the top result reproducible.
    """
    tokens = set(_TOKEN.findall(query.lower()))
    scored = [(len(tokens.intersection(entry.keywords)), entry) for entry in entries]
    hits = [pair for pair in scored if pair[0] > 0]
    return [entry for _, entry in sorted(hits, key=lambda pair: pair[0], reverse=True)]


def _format(query: str, results: list[SearchResult], provenance: str) -> str:
    """Render results as text the model can read.

    The provenance line is not decoration. Corpus notes carry invented specifics like
    "25% to 40%", and an unsourced figure reaching the report is the one thing this design
    forbids; live results carry their URL instead and may be quoted. Labelled rather than
    blunted, which would leave the corpus useless as context.
    """
    sections = [f"{len(results)} result(s) matching {query!r}, {provenance}:"]
    sections += [
        f"[{position}] {result.title}"
        + (f"\n{result.source}" if result.source else "")
        + f"\n{result.snippet}"
        for position, result in enumerate(results, start=1)
    ]
    return "\n\n".join(sections)


def _nothing_found(query: str) -> str:
    """Say a live search returned nothing. No corpus listing — the web is not a menu."""
    return (
        f"The web search for {query!r} returned no usable results. Try different wording, "
        f"or use `query_csv` for this company's own figures."
    )


def _nothing_matched(query: str, entries: list[CorpusEntry]) -> str:
    """Say nothing matched, and list what is there so the retry can be better.

    Titles come from the loaded corpus, not a hardcoded summary: a list that goes stale
    the first time a note is added is worse than no list.
    """
    return "\n".join(
        [
            f"Nothing in the offline corpus matched {query!r}. It holds only these notes:",
            *(f"- {entry.title}" for entry in entries),
            "Try one of those topics, or use `query_csv` for this company's own figures.",
        ]
    )
