# orchestra-py

A CLI multi-agent orchestrator. It turns a plain-language business request into subtasks, routes them
to role-specialised agents, runs real tools, streams live progress, and returns a report whose every
number cites its source.

```bash
uv run orchestra run "Summarize the last 3 quarters financial trends and create a chart"
```

**[Demo recording](docs/demo/demo.mov)** ·
**[Design decisions](#design-decisions)** ·
**[Trade-offs made under the 24 h constraint](#trade-offs-made-under-the-24-h-constraint)** ·
**[How to run and test](#quickstart)** ·
**[Design walkthrough](docs/presentation/orchestra-deck.html)** (HTML deck — download and open in a browser)

## How it works

| Stage | What happens |
|---|---|
| **Input** | A business request in plain text. |
| **Planning** | One orchestrator call turns it into a DAG of subtasks, each with a role. Or asks a clarifying question first. |
| **Execution** | Independent steps run concurrently under `asyncio.TaskGroup`. Each agent gets one toolset and one slice of state. |
| **Aggregation** | Finished artifacts become an executive summary, sourced key figures, and a chart. |
| **Visibility** | A Rich dashboard draws the plan, live spinners and an event log — on stderr, so stdout stays pipeable. |
| **Interrupt** | Press `i` mid-run to pause, tell the orchestrator something new, and have it replan the rest. |

---

## Quickstart

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/). It fetches Python 3.12 itself.

```bash
git clone https://github.com/pranav8494/orchestra-py.git && cd orchestra-py
uv sync
cp .env.example .env                       # then set ANTHROPIC_API_KEY=<your-key>
uv run orchestra run "Summarize the last 3 quarters financial trends and create a chart"
```

`ANTHROPIC_API_KEY` is the only required setting. Without it the run exits 3 at startup, naming the
variable and the fix. `TAVILY_API_KEY` is optional — unset, `search` reads a bundled offline corpus.

### Test

```bash
uv run pytest                 # 653 passed, 6 deselected — offline, never touches the network
uv run pytest -m live         # opt in: real provider, needs ANTHROPIC_API_KEY
uv run ruff check . && uv run mypy
```

The offline suite crosses the CLI boundary with only the provider port substituted. QA results are in
[docs/test-results.md](docs/test-results.md).

---

## Example prompts

Agents read `data/` — six files of one company's figures, 2024Q1–2025Q4, in CSV, JSON and Markdown.
`fetch_data` catalogues the directory at startup, so dropping a file in adds a dataset with no code
change. The planner gets that roster up front and plans against what the team can actually obtain.

The plan's shape follows the request:

| Prompt | Resulting plan |
|---|---|
| `Summarize the last 3 quarters financial trends and create a chart` | 3 steps, linear: retrieval → analytics → visualization |
| `Compare our last 3 quarters of revenue growth against industry benchmarks and chart the trend` | 4–5 steps. Two retrievals with no edge between them, so both agents run at once |
| `Make a chart of performance` | No plan yet — "performance" names no measure, so the planner asks first |
| `Summarize the revenue trend in one paragraph` | 2 steps, **no visualization**. Nothing asked for a chart |

Ambiguous requests are answered with a question, before any plan is drawn:

```
Which metric should the chart show?
  A. revenue
  B. profit
Answer A-B
```

One round per run. Options come from the roster, so it asks only about data the team holds.

Output is text by default, `--output json` for a machine-readable document, `--quiet` to drop the
dashboard. Every key figure carries the `artifact:` pointer of the step that produced it, and
`Chart:` names an HTML file you can open.

---

## Design decisions

| Decision | Why |
|---|---|
| **Centralized orchestrator** over a shared typed ledger | One place to enforce retry caps, step budgets and the clarification round. Peer-to-peer handoffs loop; a blackboard is infrastructure. |
| **Three roles**, one toolset and one slice of state each | A role that can fetch, compute and draw is not a team. |
| **`artifact:<name>` pointers**, never blobs in state | The ledger is serialised into prompts. Pointers keep context small and every figure traceable. |
| **Analytics writes and runs real Python** | Analysis is open-ended. A fixed set of typed helpers cannot answer the question outside it. |
| **A mechanism per named failure** | Hallucination, repetition and runaway loops each get code and a default, not a promise. |

Layers point inward: `cli/` → `app.py` → `agents/` → `{core/, tools/, providers/}`. `core/` imports
nothing upward and no vendor SDK. `app.py` is the one composition root, which is what lets the whole
application run against a `FakeProvider` swapped at a single seam.

### Agent roles

Prompts live in `src/orchestra/prompts/`, one module per agent, never inline. The same role block is
reused verbatim by the mid-run replan.

| Role | Does | Tools |
|---|---|---|
| **Orchestrator** | Decomposes the request into a DAG, or asks a clarifying question. Also replans mid-run. | `ask_user` |
| **data_retrieval** | Finds and loads raw data. The only role that may obtain data the team lacks. | `fetch_data`, `search` |
| **analytics** | Computes over retrieved data by writing Python and running it. | `run_python` |
| **visualization** | Turns computed figures into a Plotly chart plus an ASCII fallback. | — |
| **Aggregator** | Synthesises finished artifacts into the final report. | — |

Agents share context only through the ledger, as pointers — never by passing payloads to each other.

### Guardrails

| Failure | Mechanism | Default |
|---|---|---|
| **Hallucinated figures** | Every key figure carries its artifact pointer. One citing anything this run did not produce is dropped. | — |
| **Malformed output** | Every structured call validates through Pydantic; an unusable reply is retried with the rejection fed back. | 2 retries |
| **Repetition** | Signature over `(tool, input, output)` in a sliding window. Hitting the limit fails the subtask. | window 10, 5 repeats |
| **Runaway agent** | Per-subtask turn cap and token budget. | 6 turns / 60,000 tokens |
| **Runaway plan** | Global step budget. Exceeding it ends the run with a partial report, never a hang. | 15 steps |
| **Question loops** | One clarification round per run, unreachable a second time. | 1 round, ≤3 questions |
| **Unbounded fan-out** | Semaphore over the ready set. | 4 concurrent |

Model-written Python runs in a subprocess with a 15 s wall clock, a scrubbed environment, a throwaway
working directory and a socket guard. **This is isolation, not a sandbox** — it raises the cost of an
accident, it does not contain hostile code. Tool failures return as data, never raised, so the model
can read the error and retry.

---

## Trade-offs made under the 24 h constraint

| Decision | Instead of | Why |
|---|---|---|
| **CLI** (Typer + Rich) | web UI (FastAPI + React + WebSockets) | ~70% of the budget goes to orchestration and prompts rather than 40%. WebSocket debugging was the biggest risk item. |
| **Single-process `asyncio.TaskGroup`** | distributed queue | Concurrency here is a handful of I/O-bound agents. A queue adds Redis and a worker lifecycle for nothing at this scale. |
| **Bundled data + offline search corpus** | live data sources | The suite runs offline and the demo is reproducible. Live search is opt-in. |
| **Plotly HTML + ASCII fallback** | PNG rendering | PNG needs `kaleido`, ~100 MB, for an artifact the terminal cannot show anyway. |
| **Mock data, real tools** | mocked tools | `data/` is fixture data, but `fetch_data`, `search` and `run_python` are real, so the failure modes exercised are real. |

Build time exceeded the research doc's 8–10 h estimate. That estimate assumed thinner tests and no
separate QA pass. What was spent instead bought the offline end-to-end suite.

## Stretch goals

| Goal | Status |
|---|---|
| Mid-execution conversation and dynamic replanning | **Landed.** `i` pauses between steps, so a settled ledger is reshaped rather than one mid-write. Completed steps are kept. POSIX terminals only. |
| Multi-turn refinement after the final output | **Not built.** Timeboxed and cut — [#13](https://github.com/pranav8494/orchestra-py/issues/13). |

## Known limitations

| | |
|---|---|
| [#31](https://github.com/pranav8494/orchestra-py/issues/31) | On a non-UTF-8 stdout the ASCII chart's `█` bars raise `UnicodeEncodeError`; the report is discarded and the run exits 1. |
| [#33](https://github.com/pranav8494/orchestra-py/issues/33) | Visualization can drop an interior category while the title still claims the full range. |
| [#36](https://github.com/pranav8494/orchestra-py/issues/36) | The step cap, `run_python` wall clock and provider timeout never reach `Config`, so an operator cannot raise them. |
| [#40](https://github.com/pranav8494/orchestra-py/issues/40) | Tool output is re-sent whole each turn. `fetch_data` hands anything over 16 kB on as a pointer, which bounds the worst case but does not close it. |

---

[CONVENTIONS.md](CONVENTIONS.md) governs all code here. Design rationale is in
[docs/plan/Multi_Agent_Task_Solver_Research.md](docs/plan/Multi_Agent_Task_Solver_Research.md).
