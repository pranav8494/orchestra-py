"""The Data Retrieval agent's second tool: web search, or a bundled corpus without a key.

**Two backends, one tool.** With `TAVILY_API_KEY` set this searches the web, without it
the notes in `data/`. One tool because which backend answers is the operator's deployment
choice, not something the model can see or should be picking between.

**The corpus is the floor, not a stub.** It is what makes a run reproducible with no key
and no network, so the live path degrades into it — a failed request falls back and says
so rather than failing the subtask.

**Provenance differs and the model is told which it got.** Corpus notes carry invented
specifics and are labelled illustrative; live results carry a URL. A figure in the report
must trace to something, which only works if the model can tell the two apart.

Both are trust boundaries, so both are validated through a pydantic model (§7).
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

from orchestra.tools.base import ToolCall, ToolResponse, ToolSpec

TOOL_NAME = "search"

# The live backend. One provider, named here rather than made configurable: a second
# would need a second response shape, and `agents/toolsets.py` is where that choice
# would belong if there were ever two.
TAVILY_URL = "https://api.tavily.com/search"

# Seconds for one live search. Bounded because the agent loop is bounded (§10) and a
# hung request would spend the subtask's wall clock without spending its token budget.
DEFAULT_TIMEOUT = 10.0

# The two provenance clauses `_format` chooses between. Separate constants because the
# difference between them is the whole anti-hallucination contract: one set of notes may
# be quoted as fact and the other may not.
_CORPUS_PROVENANCE = (
    "from the bundled offline corpus (illustrative sample data, not this company's "
    "figures and not sourced research — do not quote its numbers as fact)"
)
_LIVE_PROVENANCE = "from a live web search (each result cites the page it came from)"

# Ceiling on `limit`. Stated in the schema so the model reads it before choosing, rather
# than discovering it in an error: the corpus is small and ten notes is already more
# background than a subtask's answer can use.
MAX_RESULTS = 10

# Words, numbers and apostrophes; everything else is a separator. Keywords are matched as
# whole tokens, so "margin" matches "margins?" but not "marginal" — the corpus carries
# both spellings of a word rather than the matcher guessing at stems.
_TOKEN = re.compile(r"[a-z0-9']+")

# The description is a prompt (§6). It must hold for both backends, because the model is
# shown one description whichever is configured — so it promises background context and
# says what this tool is not, rather than promising live data it may not have. Each
# result says which backend produced it. See `query_csv.DESCRIPTION` for why the text
# lives beside the params model instead of in `prompts/` (§11).
DESCRIPTION = (
    "Search for background and industry context: growth benchmarks, typical margin "
    "ranges, what drives a one-quarter cost change, reporting conventions, seasonality, "
    "the macro backdrop. Use it to interpret or contextualise a figure. It does NOT hold "
    "this company's own revenue, costs or profit — use `query_csv` for those. Results say "
    "where they came from; treat anything marked illustrative as context, not as fact to "
    "quote. Returns the best-matching notes, or a plain message when nothing matches."
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
    """One result, whichever backend produced it.

    The shape both paths render through, so the two backends differ in where the text
    came from and not in how it reads. A dataclass rather than a model: it is built from
    already-validated input on both sides, so it is an internal value object (§7).
    """

    title: str
    snippet: str
    source: str = ""  # a URL from the live backend; empty for a bundled note


class CorpusEntry(BaseModel):
    """One background note as it is stored on disk.

    `keywords` are single tokens, lowercased on load; matching is a set intersection
    against the tokenised query, so the note's own author decides what it answers to.
    """

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)

    @field_validator("keywords")
    @classmethod
    def _lowercase(cls, keywords: list[str]) -> list[str]:
        """Fold keywords to lowercase, because `_rank` matches a lowercased token set.

        Normalised rather than rejected: a corpus entry written `"Margin"` is a typo that
        would silently never match, and failing the whole tool over it costs the model
        every other note as well.
        """
        return [keyword.lower() for keyword in keywords]

    def as_result(self) -> SearchResult:
        """Render this note in the shape both backends share. No source: it has no URL,
        which is the whole reason its provenance line reads differently."""
        return SearchResult(title=self.title, snippet=self.snippet)


# One adapter, built at import: compiling the validator per call would be the cost of
# validation without the caching pydantic already offers.
_CORPUS = TypeAdapter(list[CorpusEntry])


class _LiveResult(BaseModel):
    """One result from the live backend, as much of it as this tool uses.

    `extra="ignore"`, unlike every other model here: the payload belongs to a vendor who
    adds fields on their schedule, and forbidding them would turn a routine API addition
    into a broken tool. The opposite of the rule for model output, where an unexpected
    field means our own schema drifted (§7).
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
    """The live backend could not answer. Never escapes `run` — it becomes the notice on
    a fallback to the corpus, because a search outage should not end a subtask that has a
    working offline source (§6)."""


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

        Args:
            corpus: the JSON array of notes, from `app.py` via `agents/toolsets.py`.
            api_key: the live backend's credential, or `None` to search the corpus only.
                Injected like everything else — a tool that read the environment could
                not be pointed at a fixture, and only `config.py` may read it (§6, §9).
            timeout: seconds allowed for one live request.
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
        )

    async def run(self, call: ToolCall) -> ToolResponse:
        """Search the web if a key was configured, else the corpus. See `BaseTool.run`."""
        try:
            params = SearchParams.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {_problems(exc)}", is_error=True
            )

        warning = ""
        if self._api_key is not None:
            try:
                results = await self._live(params)
            except _LiveSearchError as exc:
                # Degraded, not failed: the corpus can still answer, and telling the model
                # what happened is better than a silent downgrade it would read as the
                # tool's normal output. Reported twice on purpose — in `content` for the
                # model, and on `warning` for the agent, which surfaces it to the operator
                # who is the one that can actually go fix the key or the network.
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
            # Covers both a syntax error and a well-formed file of the wrong shape:
            # `validate_json` reports each as a validation error against `CorpusEntry`.
            return ToolResponse(
                content=f"{notice}The corpus at {self._corpus} is malformed: {_problems(exc)}",
                is_error=True,
                warning=warning,
            )

        matches = _rank(params.query, entries)[: params.limit]
        if not matches:
            # NOT an error, but not a result either. A search that looked and found
            # nothing answered the question correctly; flagging it invites the model to
            # retry the same call, and an `is_error` result reads to it as "the tool
            # broke", not "no such note". `is_empty` is what stops a caller counting this
            # as something retrieved (§6).
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

        A client per call rather than one held for the tool's life: `BaseTool` has no
        close hook, so a pooled client would leak its sockets, and an agent makes a
        handful of searches per run — the connection setup is not what costs here.

        Raises:
            _LiveSearchError: the request failed, was refused, or was not the shape this
                tool reads. All three mean the same thing to `run`: use the corpus.
        """
        # Assigned before the request so the type is narrowed for mypy and the secret is
        # unwrapped in exactly one place (§9).
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
            # Timeouts, DNS, connection resets. `httpx.HTTPError` is the SDK's own base
            # class, so this is not a bare `except` (§8), and `CancelledError` is a
            # BaseException that passes straight through (§10).
            raise _LiveSearchError(f"{type(exc).__name__}") from exc
        except ValidationError as exc:
            raise _LiveSearchError(f"unexpected response shape ({_problems(exc)})") from exc

        return [
            SearchResult(title=item.title or item.url, snippet=item.content, source=item.url)
            for item in body.results
            if item.content
        ]

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


def _format(query: str, results: list[SearchResult], provenance: str) -> str:
    """Render results as text the model can read, above all else honestly.

    The provenance line is not decoration, and it is the one thing that differs between
    the backends. Corpus notes are prose carrying invented specifics like "25% to 40%",
    so without the label the model has read a paragraph about margins with nothing saying
    the numbers are neither ours nor sourced — and an unsourced figure reaching the report
    is the one thing this design forbids. Live results are sourced, so they carry their
    URL instead and may be quoted. Said here rather than by blunting the corpus, which
    would leave it useless as context.
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
