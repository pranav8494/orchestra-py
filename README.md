# orchestra-py

A CLI multi-agent orchestrator. It breaks a plain-language business request into subtasks, routes
them to role-specialised AI agents, runs real tools, streams live progress, and returns a structured
report whose every number points back at the artifact it came from.

```bash
uv run orchestra run "Summarize the last 3 quarters financial trends and create a chart"
```

## What it does

| Step | What happens |
|---|---|
| **Plan** | One orchestrator call turns the request into a DAG of subtasks — or asks a clarifying question first, if the request is missing something it would otherwise have to invent. |
| **Execute** | Independent steps run concurrently under `asyncio.TaskGroup`; each agent gets only its slice of state and its own toolset. |
| **Report** | Outputs are aggregated into an executive summary, sourced key figures, and a chart. |
| **Watch** | A Rich dashboard draws the plan, spinners for live agents, and an event log — on stderr, so stdout stays pipeable. |
| **Interrupt** | Press `i` mid-run to pause, tell the orchestrator something new, and have it replan what is left. |

---

## Quickstart

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/). It fetches Python 3.12 itself —
no system Python required.

```bash
git clone https://github.com/pranav8494/orchestra-py.git && cd orchestra-py
uv sync                                    # runtime + dev deps, interpreter from .python-version
cp .env.example .env                       # then set ANTHROPIC_API_KEY=<your-key>
uv run orchestra run "Summarize the last 3 quarters financial trends and create a chart"
uv run pytest                              # 622 passed, 6 deselected (live) — offline, no network
```

`ANTHROPIC_API_KEY` is the only required setting. Everything else has a default; `.env.example`
documents each one. The two worth knowing:

| Variable | Default | Effect |
|---|---|---|
| `TAVILY_API_KEY` | unset | Unset, the `search` tool reads the bundled offline corpus. The run works either way, and a live search that fails falls back to the corpus with a warning rather than failing the step. |
| `ARTIFACT_DIR` | `~/.orchestra/artifacts` | Where each run's datasets and charts are written. |

Without a key the run fails at startup with a message naming the variable and the fix, and exits 3 —
it does not get four minutes in before noticing.

---

## Example prompts

The agents read `data/`: one CSV of this company's quarterly revenue, costs and profit for
2024Q1–2025Q4, and a corpus of industry notes the `search` tool falls back to offline. That is the
whole world they can retrieve, and the planner is told so — it will not plan a step whose data has
nowhere to come from.

### 1. Linear — retrieval → analytics → visualization

```bash
uv run orchestra run "Summarize the last 3 quarters financial trends and create a chart"
```

Three steps in sequence. The report cites revenue, cost and profit figures against the artifact the
analysis script read, and `Chart:` names an HTML file you can open.

### 2. Fan-out — two retrievals at once

```bash
uv run orchestra run "Compare our last 3 quarters of revenue growth against industry benchmarks and chart the trend"
```

4–5 steps. Two retrievals — our CSV and web search — with no dependency edge between them, so the
dashboard shows both agents spinning together, then the comparison, then the chart.

### 3. Ambiguous — the clarification flow

```bash
uv run orchestra run "Make a chart of performance"
```

"Performance" names no measure, so the planner answers `clarify` instead of `plan` and asks before
drawing any plan:

```
Which metric should the chart show?
  A. revenue
  B. profit
Answer A-B
```

Answer once and planning proceeds — there is no second round. The options offered are only measures
the team can actually retrieve, because the planner is told the roster up front. With stdin piped
(a script, CI), nobody is at the prompt: the planner is told so and plans against the most
reasonable reading rather than hanging.

### The planner is dynamic

A three-role pipeline can look dynamic while always emitting the same chain. The three prompts above
are the proof that it does not — the plan's shape follows the request:

| Request | Plan |
|---|---|
| Linear | 3 steps, sequential: retrieval → analytics → visualization |
| Fan-out | 4–5 steps, two retrievals with no edge between them, reconverging on the comparison |
| `Summarize the revenue trend in one paragraph` | 2 steps, **no visualization** — the role is dropped, not reordered |

```bash
uv run pytest tests/test_planner_scenarios.py                # shapes, offline
uv run pytest -m live tests/test_planner_scenarios_live.py   # the same shapes against the real model
```

`tests/test_end_to_end.py` asserts the fan-out overlap on the event stream too: the second
`subtask_started` lands before either `subtask_completed`.

---

## Output

### Text (default)

Abridged from a real run of the linear prompt. At a terminal this is framed in a `Report` panel;
piped, it is the bare text:

```
Over the last three quarters (2025Q2-2025Q4), the company's financial performance improved
steadily and at an accelerating pace. Revenue grew from $5.76M to $7.015M (up 21.8%
cumulatively), while costs rose more slowly, from $4.41M to $4.88M (up 10.7%). [...]

Key figures:
  2025Q4 Revenue / Costs / Profit  $7.015M / $4.88M / $2.135M  artifact:fetch_financials.json
  Cumulative revenue growth (Q2→Q4)  +21.8%  artifact:analyze_trends.json
  Cumulative profit growth (Q2→Q4)  +58.2%  artifact:analyze_trends.json

Quarterly Revenue, Costs, and Profit (2025Q2-2025Q4)
x: Quarter    y: Amount ($M)

Revenue:
  2025Q2  █████████████████████████████████         5.76
  2025Q3  ████████████████████████████████████      6.34
  2025Q4  ████████████████████████████████████████  7.01
[...]

Chart: /Users/you/.orchestra/artifacts/2026-08-17T07-29-39Z/chart_trends.html

Artifacts: /Users/you/.orchestra/artifacts/2026-08-17T07-29-39Z
Steps:
done     fetch_financials  artifact:fetch_financials.json
done     analyze_trends  artifact:analyze_trends.json
done     chart_trends  artifact:chart_trends.json
```

Every figure carries the pointer to the artifact it came from. `Chart:` is the absolute path, not
the pointer, so it opens — and it resolves to the `.html`, while the visualization step's own
pointer names its `.json` receipt. `--quiet` drops the `Artifacts:`/`Steps:` block — that is
progress. The report always prints.

### `--output json`

One document on stdout and nothing else. The keys are a published contract, defined by the view
models in `cli/format.py`:

| Key | Type | Notes |
|---|---|---|
| `request` | string | the prompt as given |
| `status` | `completed` \| `failed` | |
| `report` | object \| null | `null` when no report was produced |
| `report.executive_summary` | string | |
| `report.key_figures[]` | `{label, value, source}` | `source` is an `artifact:<name>` pointer |
| `report.chart` | pointer \| null | stays a pointer; resolve it against `artifact_dir` |
| `report.chart_ascii` | string \| null | the inline drawing |
| `subtasks[]` | `{id, role, status, artifact}` | plan order |
| `failure_reason` | string \| null | |
| `artifact_dir` | string \| null | absolute path to this run's directory |

One caveat: when configuration or planning fails there is no plan and no report, so `-o json` writes
nothing to stdout — the human-readable message goes to stderr with exit 3 or 5.

### Artifacts

Each run writes into its own timestamped subdirectory of `ARTIFACT_DIR`
(`~/.orchestra/artifacts/2026-08-17T09-14-02Z/`), so two runs never interleave. State carries
`artifact:<name>` pointers, never blobs — the ledger is serialised into prompts, so payloads stay on
disk. Datasets and analyses land as `.json`, charts as a standalone `.html` that links Plotly from a
CDN, so viewing one needs a network connection.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | unhandled exception (a bug) |
| 2 | usage error |
| 3 | configuration error |
| 4 | provider error after retries |
| 5 | task failure |
| 130 | SIGINT |

A run that fell short still prints its report; only the code says it failed. Ctrl-C cancels the
tasks, tears down the live region, restores the terminal, and exits 130 with no traceback.

---

## Architecture

Centralized orchestrator over a shared typed ledger — pattern A of the
[research doc](docs/plan/Multi_Agent_Task_Solver_Research.md), chosen over peer-to-peer handoffs
(loop-prone), a blackboard (infrastructure), and hierarchical sub-teams (over-engineered here).

```
   request
      │
      ▼
┌──────────────┐   action: clarify   ┌───────────────┐
│   Planner    │────────────────────▶│  ask the user │  at most one round
│ orchestrator │◀────────────────────│   (stderr)    │
└──────┬───────┘      answers        └───────────────┘
       │ Plan — a DAG of Subtasks
       ▼
┌───────────────────────────────────────────┐        ┌──────────────┐
│            Execution engine               │◀──`i`──│ Interrupt +  │
│  ready set → semaphore → TaskGroup        │        │   replan     │
└───┬─────────────────┬──────────────────┬──┘        └──────────────┘
    ▼                 ▼                  ▼
┌────────────┐  ┌────────────┐  ┌───────────────┐
│    Data    │  │ Analytics  │  │ Visualization │   one toolset each,
│ Retrieval  │  │            │  │               │   one slice of state each
└─────┬──────┘  └─────┬──────┘  └───────┬───────┘
      └───────────────┴─────────────────┘
                      │ artifact:<name> pointers
                      ▼
            ┌───────────────────┐        ┌────────────────┐
            │  TaskState        │───────▶│   Broker       │──▶ Rich dashboard
            │  the ledger       │ events └────────────────┘    (stderr)
            └─────────┬─────────┘
                      ▼
            ┌───────────────────┐
            │    Aggregator     │──▶ report ──▶ stdout (text | json)
            └───────────────────┘
```

Layers point inward — `cli/` → `app.py` → `agents/` → `{core/, tools/, providers/}`. `core/` imports
nothing upward and no vendor SDK, so the whole engine runs behind a different front end unchanged.
`app.py` is the one composition root; nothing below it constructs a provider, store or worker, which
is what lets the entire test suite run against a `FakeProvider` without patching. The rules are in
[CONVENTIONS.md](CONVENTIONS.md).

---

## Agent roles and prompts

Prompts live in `src/orchestra/prompts/`, one module per agent, never inline. The planner's `ROLES`
block and the engine's `AgentRole` enum are the same three names, asserted by a test.

| Role | Does | Tools | Prompt |
|---|---|---|---|
| **Orchestrator** (planner) | Decomposes the request into a DAG, or asks a clarifying question. Also replans mid-run. | `ask_user` | `prompts/planner.py`, `prompts/interrupt.py` |
| **data_retrieval** | Finds and loads raw data. The only role that may obtain data the team does not already have. | `query_csv`, `search` | `prompts/data_retrieval.py` |
| **analytics** | Computes over data an earlier step retrieved — aggregations, trends, comparisons — by writing Python and running it. | `run_python` | `prompts/analytics.py` |
| **visualization** | Turns computed figures into a Plotly chart plus an ASCII fallback. One structured call, no tool loop. | — | `prompts/visualization.py` |
| **Aggregator** | Synthesises the finished artifacts into the final report. | — | `prompts/aggregator.py` |

The planner is told what the retrieval tools can actually obtain, so it never plans a step whose
data has nowhere to come from — and never offers a clarifying choice it cannot satisfy.

---

## Guardrails

The rubric names hallucination, repetition and runaway loops, so each gets a mechanism rather than a
promise. Defaults shown; the configurable ones are in `.env.example`.

| Failure | Mechanism | Default |
|---|---|---|
| **Hallucinated figures** | Every key figure carries the pointer to the artifact the analysis script actually read. Unbacked figures are dropped before the report is written. | — |
| **Malformed model output** | Every structured call validates through Pydantic; an unusable reply is retried with the rejection fed back. | 2 retries |
| **Repetition** | Signature over `(tool, input, output)` in a sliding window; a repeating tool cycle ends the turn. | window 10, 5 repeats |
| **Runaway agent** | Per-subtask turn cap and token budget. Both apply: turns catch an agent that keeps calling tools, tokens catch one calling expensive ones. | 6 turns / 60,000 tokens |
| **Failing step** | Per-subtask attempt cap. A deterministic failure — a bound already hit, a plan whose order is wrong — is not retried; it would cost the same three times. | 3 attempts |
| **Runaway plan** | Global step budget across all attempts. Exceeding it ends the run with a partial report, never a hang. | 15 steps |
| **Question loops** | At most one clarification round per run; the ledger holding an answer makes a second unreachable, even after a mid-run replan. | 1 round, ≤3 questions |
| **Unbounded fan-out** | Semaphore over the ready set. | 4 concurrent |

### Running model-written code

The Analytics agent computes by writing Python and running it, rather than by calling a fixed set of
typed helpers — analysis is open-ended, and an enumeration of operations cannot answer the question
outside it. The trade-off:

| Guard | What it buys |
|---|---|
| Subprocess with a 15 s wall clock, killed by process group | a runaway loop ends the call, not the run |
| Scrubbed environment (`-I`, allow-listed vars) | the run's API keys never reach the script |
| Throwaway working directory, also its `HOME`/`TMPDIR` | nothing is left behind for the next script or the user's home |
| `socket` guard in the child | no accidental network call mid-analysis |

**This is isolation, not a sandbox.** It raises the cost of an accident; it does not contain hostile
code — a script can undo a monkeypatch, and real containment means a container, a seccomp profile or
a separate uid. The code being run comes from our own model and our own prompt, and untrusted input
is out of scope for this project.

Tool failures are returned as data, never raised: the model has to be able to read the error and
retry. An exception would unwind the agent loop and deny it that.

---

## Trade-offs made under the 24 h constraint

| Decision | Instead of | Why |
|---|---|---|
| **CLI** (Typer + Rich) | web UI (FastAPI + React + WebSockets) | Roughly 70% of the budget goes to orchestration and prompts rather than 40%; a terminal dashboard buys live progress at a fraction of the effort, and WebSocket debugging was the biggest risk item. |
| **Single-process `asyncio.TaskGroup`** | distributed queue or blackboard | Concurrency here is a handful of I/O-bound agents. A queue would add Redis, a worker lifecycle and non-deterministic ordering, and buy nothing at this scale. |
| **Centralized orchestrator** | peer-to-peer handoffs | One place to enforce retry caps, step budgets and the clarification round. P2P makes A→B→A loops easy and top-level progress hard to show. |
| **Bundled mock CSV + offline search corpus** | live data sources | The whole suite runs offline and the demo is reproducible. Live web search is opt-in via `TAVILY_API_KEY`. |
| **Plotly HTML + ASCII fallback** | PNG rendering | PNG needs `kaleido`, a ~100 MB dependency, for an artifact the terminal cannot show anyway. The ASCII chart goes in the report; the HTML opens in a browser. |
| **Pointer-based artifacts** | blobs in state | The ledger is serialised into prompts. Pointers keep context small and make every figure traceable. |
| **Mock data, real tools** | mocked tools | The CSV is fixture data, but `query_csv`, `search` and `run_python` are the real implementations — the failure modes exercised are the real ones. |

Actual build time exceeded the research doc's 8–10 h estimate. That estimate was the compressed
path — thinner tests, no separate QA pass. What was spent instead bought the offline end-to-end
suite in `tests/test_end_to_end.py`, which crosses the CLI boundary with only the provider port
substituted.

## Stretch goals

| Goal | Status |
|---|---|
| Mid-execution interrupt and dynamic replanning | **Landed.** `i` arms a pause the engine takes up between steps, so a settled ledger is reshaped rather than one mid-write. The orchestrator may replan, restart a step, ask, or continue; completed steps are kept, and a restarted step resets everything downstream of it. POSIX terminals only — with piped stdin or on Windows the key is never watched and the run never pauses. |
| Multi-turn refinement after the final output | **Not built.** Timeboxed and cut — [#13](https://github.com/pranav8494/orchestra-py/issues/13). |

## Known limitations

| | |
|---|---|
| [#31](https://github.com/pranav8494/orchestra-py/issues/31) | On a non-UTF-8 stdout (`PYTHONIOENCODING=ascii`), the ASCII chart's `█` bars raise `UnicodeEncodeError`; the report is discarded and the run exits 1. |
| [#33](https://github.com/pranav8494/orchestra-py/issues/33) | Visualization can drop interior categories — a three-quarter request once charted Q2 and Q4 with the title still claiming Q2–Q4. Values correct, middle point absent. |
| [#36](https://github.com/pranav8494/orchestra-py/issues/36) | The global step cap, the `run_python` wall clock and the provider's timeout never reach `Config`, so an operator cannot raise them. |
| [#40](https://github.com/pranav8494/orchestra-py/issues/40) | `query_csv` and `search` output is stored and re-sent whole; only `run_python` is character-capped. The token budget catches it after it has been paid for once. |

---

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                 # offline; the suite must never touch the network
uv run pytest -m live         # opt in: real provider, needs ANTHROPIC_API_KEY
```

[CONVENTIONS.md](CONVENTIONS.md) governs all code here. Design rationale is in
[docs/plan/Multi_Agent_Task_Solver_Research.md](docs/plan/Multi_Agent_Task_Solver_Research.md);
QA results are in [docs/test-results.md](docs/test-results.md).
