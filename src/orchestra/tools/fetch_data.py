"""The Data Retrieval agent's primary tool: any bundled data file, fetched by name.

- **Format-agnostic** — the directory is scanned once at startup and each file probed for
  its shape. Selecting and joining are `run_python`'s job, so no format needs a reader
  here; the model picks a file, not a query. stdlib only: pandas is the Analytics agent's
  dependency, and importing it here would be the second place deciding what a frame is.
- **Small files inline, everything else by pointer** — a tool result is re-sent every later
  turn (#40), so a file that is large, or not text at all, is handed on as an artifact the
  analysis step opens. The two are worded apart: "too large" said of a binary file is a
  claim the retrieval summary would repeat into the report.
- **The catalogue is what the store will accept** — a name the artifact store would refuse
  is repaired at discovery, so a file the planner is offered is one every call can fetch.
- **Every failure is content, not an exception** (§6). Only `CancelledError` leaves `run`.
"""

import asyncio
import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestra.artifacts import ArtifactStore
from orchestra.core.errors import TaskFailure
from orchestra.core.state import ARTIFACT_NAME_PATTERN
from orchestra.tools.base import (
    ToolCall,
    ToolResponse,
    ToolSpec,
    format_validation_error,
)

TOOL_NAME = "fetch_data"

# A tool result is replayed on every later turn against `WORKER_TOKEN_BUDGET` (60k), so
# what is inlined is paid for repeatedly (#40). 16 kB is ~4k tokens: two of them across a
# six-turn loop still leave the loop over half its budget. Bigger files earn a pointer.
INLINE_MAX_BYTES = 16_000

# Bound on every startup probe: no branch reads past this — bytes where the size is
# checked, characters where the read is decoded text — so a 200 MB export on one line
# costs a chunk rather than the run's memory. Generous: anything under it is described
# exactly.
PROBE_MAX_BYTES = 1_000_000

# Keys in `ToolResponse.metadata`. Structured, so the worker reads the pointer instead of
# parsing it back out of prose written for the model (§6).
POINTER_KEY = "pointer"
INLINED_KEY = "inlined"

# The catalogue is embedded in the tool's description (re-sent every retrieval turn) and
# in `provides` (every planner call), so it is capped like any other prompt input: a
# directory of 200 files, or one CSV with 300 columns, must not inflate every request.
MAX_LISTED_DATASETS = 12
MAX_SUMMARY_CHARS = 160

# Probed by suffix. Anything not listed is still offered, described by size alone.
_DELIMITERS = {".csv": ",", ".tsv": "\t"}
_TEXT_SUFFIXES = (".txt", ".md")

# What a probe may raise on a file this process cannot make sense of. `ValueError` covers
# both `json.JSONDecodeError` and `UnicodeDecodeError`.
_UNREADABLE = (OSError, csv.Error, ValueError)

# The complement of `ARTIFACT_NAME_PATTERN`'s allow-list, for repairing a filename the
# store would refuse. `_` is inside the allow-list, so a repaired name always passes.
_UNSAFE_CHARACTERS = re.compile(r"[^\w.\- ]+")


@dataclass(frozen=True, slots=True)
class Dataset:
    """One file in the catalogue: how the model names it, where it is, what is in it."""

    name: str  # the file stem — the key the model sends
    path: Path
    summary: str  # its shape, probed at startup: columns, keys, or size
    store_name: str  # `path.name` as the artifact store will accept it


@dataclass(frozen=True, slots=True)
class _Fetched:
    """What one fetch found. `text` is the file's contents, or `None` with the reason why
    not in `withheld` — a sentence, because the two reasons read differently to a model."""

    pointer: str
    text: str | None = None
    withheld: str = ""


class FetchDataParams(BaseModel):
    """The arguments the model may send, and the check applied to what it sent.

    Published verbatim as `input_schema`, so the `description` is prompt text. `extra` is
    forbidden so a hallucinated `columns` comes back as a readable validation error
    instead of being silently dropped (§7).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        description="Which dataset to fetch, by the name listed in this tool's description.",
    )


class FetchDataTool:
    """Hands over one bundled data file per call. Implements `BaseTool` (§6).

    The catalogue is fixed at construction; `run` looks the model's name up in it and
    never builds a path from model input.
    """

    def __init__(self, store: ArtifactStore, datasets: Sequence[Dataset]) -> None:
        """Take the store and the catalogue, injected (§3.3). The spec is built here,
        not per call: `info()` runs every turn and must stay pure."""
        self._store = store
        self._datasets = {dataset.name: dataset for dataset in datasets}
        # One pointer per dataset per run: registering on every call would leave `x.csv`
        # and `x-1.csv` behind, two pointers for one file and two copies of a large one.
        self._pointers: dict[str, str] = {}
        self._spec = ToolSpec(
            name=TOOL_NAME,
            description=_description(datasets),
            input_schema=FetchDataParams.model_json_schema(),
            provides=_provides(datasets),
        )

    def info(self) -> ToolSpec:
        """See `BaseTool.info`. Pure: no disk is touched, it runs on every turn."""
        return self._spec

    async def run(self, call: ToolCall) -> ToolResponse:
        """Return the named file's contents, or a pointer to it. See `BaseTool.run`."""
        try:
            params = FetchDataParams.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResponse(
                content=f"Invalid arguments for {TOOL_NAME}: {format_validation_error(exc)}",
                is_error=True,
            )

        dataset = self._datasets.get(params.name)
        if dataset is None:
            return ToolResponse(content=self._no_such_dataset(params.name), is_error=True)

        try:
            # The engine dispatches subtasks concurrently, so the copy and the read go to
            # a thread rather than stalling every other agent (§10).
            fetched = await asyncio.to_thread(
                _fetch, self._store, dataset, self._pointers.get(dataset.name, "")
            )
        except (OSError, TaskFailure) as exc:
            # `TaskFailure` is the store refusing the copy, `OSError` the file itself. To
            # the model they are one event: this dataset is unavailable. Nothing else is
            # caught — a `ValueError` here would be a bug in this module, not news for
            # the model (§8).
            return ToolResponse(
                content=f"Could not read {dataset.name} at {dataset.path}: {exc}", is_error=True
            )

        # Cached after the copy, never before: a failed registration must leave no pointer
        # behind. A race between two subtasks costs one extra copy, not correctness.
        self._pointers[dataset.name] = fetched.pointer
        metadata = {
            POINTER_KEY: fetched.pointer,
            INLINED_KEY: str(fetched.text is not None).lower(),
        }

        if fetched.text is None:
            return ToolResponse(
                content=f"{dataset.name}: {dataset.summary}. {fetched.withheld} It is stored as "
                f"{fetched.pointer} — pass that pointer to the analysis step, which reads the "
                "file itself, rather than asking for its contents here.",
                metadata=metadata,
            )
        if not fetched.text.strip():
            # Ran, found nothing: neither a failure to retry — the file will be just as
            # empty next time — nor a result the next step can compute over (§6). Said in
            # a sentence, because empty content reads to the model as a tool that broke.
            return ToolResponse(
                content=f"{dataset.name} is an empty file: it exists but holds no data. "
                "Fetch a different dataset, or report that this one is empty.",
                is_empty=True,
                metadata=metadata,
            )
        return ToolResponse(content=fetched.text, metadata=metadata)

    def _no_such_dataset(self, name: str) -> str:
        """Name what is on file, so the error carries the retry (§6). Capped like the
        description: an unknown name must not cost more prompt than the catalogue does."""
        if not self._datasets:
            return f"There is no dataset named {name!r}; this deployment has no data files at all."
        listed = ", ".join(list(self._datasets)[:MAX_LISTED_DATASETS])
        hidden = len(self._datasets) - MAX_LISTED_DATASETS
        more = f", and {hidden} more" if hidden > 0 else ""
        return f"There is no dataset named {name!r}. The datasets are: {listed}{more}."


def discover_datasets(data_dir: Path) -> tuple[Dataset, ...]:
    """The catalogue: every readable file in `data_dir`, probed for its shape.

    Called once at startup by `agents/toolsets.py`. Sorted, so the planner's roster does
    not depend on directory order. A missing directory yields none rather than raising —
    installed as a wheel the default points at nothing (`config.default_data_dir`) — and
    an unprobeable file is skipped: one bad file must not cost the run its other data.
    """
    try:
        # Dotfiles are the platform's, not the operator's: a stray `.DS_Store` offered as
        # a dataset is noise, and the store's name rules reject a leading dot anyway.
        paths = sorted(
            path for path in data_dir.iterdir() if path.is_file() and not path.name.startswith(".")
        )
    except OSError:
        return ()

    datasets: list[Dataset] = []
    taken: set[str] = set()
    for path in paths:
        # The stem is the key the model sends, so two files sharing one fall back to the
        # whole filename — `sales.csv` and `sales.xlsx` are both fetchable, where dropping
        # the second would need a warning this module has nowhere to emit.
        name = path.name if path.stem in taken else path.stem
        store_name = _storable(path.name)
        if name in taken or not store_name:
            continue
        try:
            summary = _probe(path)
        except _UNREADABLE:
            continue
        taken.add(name)
        datasets.append(Dataset(name=name, path=path, summary=summary, store_name=store_name))
    return tuple(datasets)


def _probe(path: Path) -> str:
    """Describe one file's shape in a clause, reading no more than `PROBE_MAX_BYTES`.

    Raises:
        OSError, csv.Error, ValueError: unreadable or unparseable. The caller skips it.
    """
    suffix = path.suffix.lower()
    if suffix in _DELIMITERS:
        header = _header(path, _DELIMITERS[suffix])
        kind = suffix.removeprefix(".").upper()
        return f"{kind} with columns {', '.join(header)}" if header else f"empty {kind}"
    if suffix == ".jsonl":
        # `object`, not what `json.loads` returns: `Any` is banned in `tools/` (§7).
        first: object = json.loads(_first_line(path) or "null")
        return f"JSON lines, {_shape_of(first)} per line"
    if suffix == ".json":
        return _json_summary(path)
    size = path.stat().st_size
    if suffix in _TEXT_SUFFIXES:
        return f"text, {size} bytes, beginning {_first_line(path)!r}"
    return f"{suffix or 'extensionless'} file, {size} bytes, contents not inspected"


def _fetch(store: ArtifactStore, dataset: Dataset, known: str) -> _Fetched:
    """Register the file and read it if it is small text. Blocking — call in a thread.

    `known` is the pointer an earlier call already minted, or "" to register now.

    Raises:
        OSError: the file could not be read.
        TaskFailure: the store could not take the copy.
    """
    pointer = known or store.put_file(dataset.path, name=dataset.store_name)
    if dataset.path.stat().st_size > INLINE_MAX_BYTES:
        return _Fetched(
            pointer,
            withheld=f"That is more than the {INLINE_MAX_BYTES} bytes this tool returns "
            "inline, so its contents are not shown here.",
        )
    try:
        return _Fetched(pointer, text=dataset.path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # Small, but not text. Said as its own reason: reported as "too large" it would be
        # a false claim, and the retrieval summary repeats what the tool said into the
        # report.
        return _Fetched(
            pointer,
            withheld="The file is not UTF-8 text, so its contents cannot be shown here.",
        )


def _description(datasets: Sequence[Dataset]) -> str:
    """A prompt (§6), beside the params model rather than in `prompts/` (§11 — one module
    per *agent*), because it documents this tool's own catalogue."""
    if not datasets:
        return (
            "Fetch a bundled data file by name. This deployment has no data files, so "
            "every call reports that; use the `search` tool for background instead."
        )
    return (
        "Fetch one of this team's own bundled data files by name. This is the only source "
        "for our own figures — quote what it returns rather than estimating. Available: "
        f"{_catalogue(datasets)}. A small text file comes back in full; anything larger, or "
        "not text, comes back as a summary plus an artifact pointer to hand to the analysis "
        "step, which opens the file itself. Fetch the whole file: selecting, filtering and "
        "joining are the analysis step's job, not this one's. Do not use it for industry "
        "benchmarks, definitions or macro background — that is the `search` tool."
    )


def _provides(datasets: Sequence[Dataset]) -> str:
    """What the planner is told this puts within reach (`ToolSpec.provides`).

    Composed from the catalogue, so a file added to `data/` is one the planner may plan
    against without an edit here; empty when there is none, so `retrievable_data` skips
    the tool. The closing boundary is load-bearing: without it a request for data nobody
    holds — a share price, a headcount — is planned as steps that retrieve nothing (#10).
    """
    if not datasets:
        return ""
    return (
        f"this team's own bundled data files — {_catalogue(datasets)} — and nothing beyond "
        "those files and their own columns: no other entity, measure or period exists here"
    )


def _catalogue(datasets: Sequence[Dataset]) -> str:
    """The catalogue in one clause, capped. Shared by the model's description and the
    planner's roster — two audiences, one list, so they cannot drift apart."""
    listed = "; ".join(_entry(dataset) for dataset in datasets[:MAX_LISTED_DATASETS])
    hidden = len(datasets) - MAX_LISTED_DATASETS
    return f"{listed}; and {hidden} more not listed here" if hidden > 0 else listed


def _entry(dataset: Dataset) -> str:
    """One catalogue line, its summary elided if a very wide file made it long.

    Elided like `ArtifactStore.preview` does a payload, and for the same reason: this text
    is prompt input, so its size is the caller's to bound rather than the file's to decide.
    """
    summary = dataset.summary
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = f"{summary[:MAX_SUMMARY_CHARS]}... [elided]"
    return f"{dataset.name} ({summary})"


def _storable(filename: str) -> str:
    """`filename` as the artifact store will accept it, or "" when nothing usable is left.

    Repaired rather than rejected: `Q3 P&L.csv` is an ordinary filename, and a catalogue
    that advertised it only to fail every call would be the #10 failure the boundary
    clause exists to prevent. Checked against the store's own pattern, not a second
    opinion about it.
    """
    repaired = _UNSAFE_CHARACTERS.sub("_", filename).lstrip(".")
    return repaired if re.fullmatch(ARTIFACT_NAME_PATTERN, repaired) else ""


def _header(path: Path, delimiter: str) -> list[str]:
    """The first row of a delimited file, parsed. Bounded by `_first_line`."""
    return next(csv.reader([_first_line(path)], delimiter=delimiter), [])


def _first_line(path: Path) -> str:
    """The file's first line, from a bounded read. Blocking.

    `read` then split, not `readline`: a file with no newline in it is one line, and
    `readline` would pull all of it into memory to say so.
    """
    with path.open(encoding="utf-8") as handle:
        return handle.read(PROBE_MAX_BYTES).split("\n", 1)[0].strip()


def _json_summary(path: Path) -> str:
    """The top-level shape of a JSON document: a list's first object, or an object's keys."""
    size = path.stat().st_size
    if size > PROBE_MAX_BYTES:
        # No header to read: the shape costs a whole parse, so past the bound the file is
        # described by size alone rather than delaying every run.
        return f"JSON, {size} bytes, too large to probe"
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return f"JSON list of {len(value)} entries, {_shape_of(value[0] if value else None)} each"
    return _shape_of(value)


def _shape_of(value: object) -> str:
    """Describe one decoded JSON value: an object by its keys, anything else by its type."""
    if isinstance(value, dict):
        keys = ", ".join(str(key) for key in value)
        return f"an object with keys {keys}" if keys else "an empty object"
    return f"a JSON {type(value).__name__}"
