# Test results (#16)

> Recorded at #45. The checklist was rewritten for #46's `fetch_data` and re-run on 2026-08-17:
> see [that pass](#pass-2026-08-17--fetch_data-and-the-six-file-catalogue) for the evidence behind
> every tick, the boxes a non-TTY shell cannot reach, and the defects it found. The offline suite
> reports **653 passed, 6 deselected**; `pytest -m live` is now **green — 6 passed** on the rerun at
> #49, which cleared D2 and did not reproduce #48.

Automated suite (offline, the four gates):

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Live tests opt in: `uv run pytest -m live` (needs `ANTHROPIC_API_KEY`; `TAVILY_API_KEY` optional —
without it the search tool reads the bundled corpus instead of the web).

## Manual live-LLM checklist

Run from a real terminal with `ANTHROPIC_API_KEY` set. `$?` is checked after each run. Ticks carry
the command, the exit code and the observation that satisfied them; an unticked box carries the
reason it was not run.

Evidence is keyed to the runs in [the 2026-08-17 pass](#pass-2026-08-17--fetch_data-and-the-six-file-catalogue):
G the four gates, N the offline checks, R1–R6 the six live model runs.

### Planner scenarios

- [x] `uv run orchestra run "Summarize the last 3 quarters financial trends and create a chart"` → 3 steps, sequential: retrieval → analytics → visualization; exit 0 — **R2, exit 0**: `fetch_quarterly_financials` (data_retrieval) → `analyze_financial_trends` (analytics) → `chart_financial_trends` (visualization); no dashboard frame of the 830 drawn ever held two `running` cells
- [x] `uv run orchestra run "Compare our last 3 quarters of revenue growth against industry benchmarks and chart the trend"` → 4–5 steps, two retrievals with no edge between them, then analytics, then visualization; exit 0 — **R3, exit 0**: 4 steps, `fetch_revenue` + `fetch_benchmarks` both started before either finished, then `compute_growth_comparison`, then `chart_growth_trend`
- [ ] `uv run orchestra run "Summarize the revenue trend in one paragraph"` → 2 steps, **no** visualization step; exit 0 — **not run**: the 6-run budget went to the #46 criteria. R1 proves the *plan* shape live (`test_the_planner_shapes_a_real_plan_to_the_request[role_omission]` PASSED); the CLI run end to end was not made
- [x] `uv run pytest -m live` → live planner and aggregation tests pass — **R7, exit 0: 6 passed, 653 deselected in 82.63s**. Was R1's 2 failed / 4 passed; D2 is fixed at `cb1dc87` and D1/[#48](https://github.com/pranav8494/orchestra-py/issues/48) did not reproduce this time

### The catalogue and `fetch_data` (#46)

- [x] Roster the planner is given lists the **six** datasets and excludes `search_snippets.json`; the boundary clause closes it — **N, exit 0**: `retrievable_data(data_retrieval_tools(Path("data"), store))` printed all six and no corpus, closing with "and nothing beyond those files and their own columns"
- [x] A question that needs two datasets → retrieval fetches both files whole; the slicing happens in the analytics step, not the tool — **R4, exit 0**: "Do the revenue, costs and profit in the Q4 board pack agree with quarterly_financials for 2025Q4, and which expense category was the largest that quarter?" planned three retrieval steps; every file came back **whole** — all 8 quarters of `quarterly_financials`, all 32 rows of `expense_breakdown` — and the 2025Q4 row was picked out in `compare_and_identify_largest`'s pandas, not by the tool
- [x] A question only `product_lines.json` or `project_timeline.md` can answer → the right non-CSV file is fetched and the report's figures come from it — **R5, exit 0**: fetched `{"name": "product_lines"}` and `{"name": "project_timeline"}`; the growth figures trace to the JSON and the milestones explaining them to the Markdown
- [x] A request naming data no file holds → the run says so rather than inventing it; no figure cites a file that has none — **R6, exit 0**: "Break down our 2025 employee attrition rate by department." reported "does not include an employee attrition rate or any department-level headcount breakdown", named all six files as checked, and stated **zero** key figures
- [x] `Q4 Board Pack (final)` fetches under the repaired pointer `artifact:Q4 Board Pack _final_.csv`, and the payload is on disk under that name — **R4, exit 0**: `fetch_boardpack_financials.json` records `query: {"name": "Q4 Board Pack (final)"}` → `pointer: artifact:Q4 Board Pack _final_.csv`, and that 520-byte file is in the run directory. Also **N** against the tool directly
- [x] Drop a new file into `data/` (or a `DATA_DIR` copy) → it appears in the roster and is fetchable with no code change — **N, exit 0**: `regional_bookings.csv` written into a `DATA_DIR` copy appeared as `regional_bookings (CSV with columns region, quarter, bookings)`. Run against a copy, so nothing was added to the committed `data/`
- [x] A file over `INLINE_MAX_BYTES` (16 kB) → summary plus artifact pointer, contents withheld, `inlined=false` — **N, exit 0**: a 41,311-byte `big_ledger.csv` returned "That is more than the 16000 bytes this tool returns inline", `metadata={'pointer': 'artifact:big_ledger.csv', 'inlined': 'false'}`
- [x] An unknown dataset name → error naming the datasets that do exist, returned as content, not raised — **N, exit 0**: `{"name": "share_price"}` → `is_error=True` and "There is no dataset named 'share_price'. The datasets are: …" as content. `search_snippets` gets the same answer, confirming the corpus is not fetchable

### Fan-out concurrency (#17's outstanding criterion)

- [ ] Rerun the fan-out prompt above and watch the dashboard → both retrieval agents show as running **at the same time**, two spinners live before either completes — **not tickable here**: needs a human eye on a painted frame. Only the event stream was read
- [x] Same run, on the event log → the second `subtask_started` appears before the first `subtask_completed` — **R3, exit 0**: stderr's plain sink reads `start fetch_revenue` / `start fetch_benchmarks` / `done fetch_revenue` / `done fetch_benchmarks`, in that order

### Clarification (#10)

- [ ] `uv run orchestra run "Make a chart of performance"` → one clarifying question before any plan is drawn — **not run**: needs a real TTY. With stdin a pipe the reply is rejected and the model plans instead, which tests a different path
- [ ] Answer it once → planning proceeds without a second question; run reaches a report; exit 0 — **not run**: same, needs a TTY to answer at
- [x] Question offers only measures the team can actually retrieve (no metric no tool supplies) — **R1**: the question the live model drew was "Which performance metric should the chart show? … quarterly financials (revenue, costs, profit) / yearly performance (revenue, costs, profit, headcount, customers, net revenue retention) / product line revenue / expense breakdown by category". Every option maps to a real column; none of "stock", "share price", "traffic" was offered. Ticked on the observed question, not on the test — the test around it is red for a stale fixture (D2)

### Mid-run interrupt and replan (#12)

- [ ] Start the linear prompt; stderr shows `Press i to interrupt the run and talk to the orchestrator.` — **not run**: the hint is only printed when stdin is a TTY
- [ ] Press `i` mid-run → dashboard pauses, chat prompt opens, in-flight step is not lost — **not run**: needs a real TTY and a key pressed mid-run
- [ ] Ask for a changed goal (for example "skip the chart, just summarise") → orchestrator replans; completed steps are kept, unfinished ones replaced — **not run**: same
- [ ] Resume → run finishes on the **new** plan; report reflects it; exit 0 — **not run**: same

### Cancellation

- [ ] Start any run, press Ctrl-C mid-execution, then `echo $?` → prints `130` — **not run**: needs a real SIGINT from an operator's own shell
- [ ] Same shell afterwards: type a command → characters echo normally, Enter works (cbreak handed back) — **not run**: line discipline can only be judged in the shell that owned the terminal
- [ ] No traceback on stderr; a single `Interrupted.` line — **not run**: same run as above
- [ ] Terminal cursor is visible and the Live region is torn down (no stuck dashboard frame) — **not run**: needs a human eye on the shell afterwards

### Startup and usage

- [x] `uv run orchestra --help`, `orchestra run --help`, `orchestra --version` → exit 0; `run`'s flags are `-o/--output`, `-q/--quiet`, `--debug` — **N, exit 0** for all three; `--version` printed `0.1.0`
- [x] `ANTHROPIC_API_KEY=` → exit 3, stderr names the variable and the fix, stdout empty, no traceback — **N, exit 3**: stderr was `ANTHROPIC_API_KEY is not set.` plus the `.env` fix line; stdout 0 bytes even under `-o json`

### Output modes

- [x] `uv run orchestra run "<fan-out prompt>" -o json | python -c "import json,sys; json.load(sys.stdin)"` → parses; exactly one document; no progress text on stdout; keys match `cli/format.py` — **R3, exit 0**: `json.load` exit 0, `raw_decode` left only a trailing newline, and the key sets equal `ResultDocument`/`ReportView`/`FigureView`/`SubtaskView`'s fields in order. Progress went to stderr's plain sink
- [x] `uv run orchestra run "<prompt>" -q` → no progress lines; the report still prints on stdout — **R6, exit 0**: stderr **0 bytes**, report on stdout, and no `Artifacts:`/`Steps:` block
- [x] `uv run orchestra run "<linear prompt>" > /tmp/out.txt` → progress still drawn on stderr, report lands in the file — **R2, exit 0**: the report is in the file and the progress in the stderr capture. Caveat, since fixed: this shell has `FORCE_COLOR=3`, so Rich called the redirect a terminal and framed the report in a `Panel`. Filed as [#49](https://github.com/pranav8494/orchestra-py/issues/49), fixed in this commit — see O1 for the before/after evidence

### Artifacts

- [ ] After a charted run, open the chart in `~/.orchestra/artifacts/<run>/*.html` (each run gets its own timestamped subdirectory) → renders in a browser with this run's numbers, axes and title — **not tickable here**: no browser and no human eye. The file's contents were read instead, on the two boxes below
- [x] The charted run wrote its `.html` into its own timestamped subdirectory, and `Chart:` names that path — **R2, exit 0**: `~/.orchestra/artifacts/2026-08-17T10-12-48Z/chart_financial_trends.html`, 8,126 bytes, `<html` present, and the report's `Chart:` line is that absolute path
- [x] Report's ASCII chart matches the HTML chart's shape — **R2, exit 0**: the HTML's three traces are `x:["2025 Q2","2025 Q3","2025 Q4"]` with `y:[5.76,6.34,7.02]`, `[4.41,4.62,4.88]`, `[1.35,1.72,2.135]`, equal to the ASCII bars and labels; titles agree ("Revenue, Costs, and Profit Trends (2025 Q2-Q4)", "Quarter", "Amount (USD Millions)"). All three quarters are present, so #33 did not recur here. Read out of the file, not seen rendered

## Automated results

`uv run pytest` — **622 passed, 6 deselected** (live) on the day of the pass; baseline before this
ticket was 607. `ruff check`, `ruff format --check` and `mypy` (strict, 86 files) all clean.

Everything below runs offline against `FakeProvider`: the real planner, workers, tools over
`data/`, aggregator and Typer command. The provider port is the rule, with two named exceptions —
the executor's clock, patched at the class reference because no `Config` field carries it (#36),
and the CLI's asker, because a `CliRunner` stdin is a pipe and the command would otherwise
correctly decide nobody can be asked.

| AC | Covered by (`tests/test_end_to_end.py` unless noted) | Result |
|---|---|---|
| 1. Three scenarios end to end | `test_scenario_runs_every_step_and_leaves_every_pointer_resolvable[linear\|fan_out\|role_omission]`, `test_fan_out_starts_both_retrievals_before_either_completes`, `test_role_omission_reports_no_chart_and_writes_no_html` | PASS |
| 2. Ambiguous prompt, one answer, then a plan | `test_cli_run_answers_one_clarifying_question_and_then_completes`; `run_once` arm in `test_app.py` | PASS |
| 3. Malformed output / executor timeout → partial report, no hang, no traceback | `test_cli_run_with_an_unusable_plan_exits_task_failure_without_a_traceback`, `test_a_malformed_chart_draft_fails_one_step_and_the_run_still_reports`, `test_a_runaway_script_is_killed_and_costs_only_its_own_step`, `test_cli_run_whose_synthesis_is_refused_degrades_to_the_ledger_and_exits_zero` | PASS |
| 4. Ctrl-C exits 130, terminal restored | `test_cli_run_interrupted_inside_a_step_exits_130_and_releases_the_terminal`; line discipline in `test_chat.py::test_the_terminal_is_handed_back_when_an_exception_unwinds_the_chat` | PASS |
| 5. Automated on `FakeProvider` + manual checklist | whole suite (`-m "not live"` by default); checklist above | PASS |
| 6. Mid-execution interrupt / replanning (#12 landed; #13 did not) | `test_an_interrupt_replans_what_is_left_and_only_the_new_step_runs` | PASS |
| 7. Results recorded | this file | PASS |

Two assertions were mutation-checked to prove they can fail: forcing `MAX_CONCURRENCY=1` breaks
the fan-out overlap check, and collapsing the per-branch turn queues into one FIFO reproduces the
interleaving bug they exist to prevent.

### Verified by hand

Ctrl-C was also checked with a **real SIGINT** into a real `orchestra run` (the automated case
injects the exception into a task, which is not the same code path). Observed: exit 130, empty
stdout, stderr ending in a single `Interrupted.`, the cursor shown again, and no traceback.

### Not covered automatically

| Gap | Why | Where it is covered |
|---|---|---|
| cbreak/raw restored after Ctrl-C *at a real terminal* | `CliRunner` supplies a pipe for stdin, so `_interactive()` is false and no `ConsoleChat` is ever built | `test_chat.py` over a pty; manual checklist above |
| Dashboard visibly showing two agents at once | Asserted on renderer input, never on a painted live run | manual checklist (#17's outstanding criterion) |
| Live-model plan shapes | Suite must not touch the network (§12) | `uv run pytest -m live`; manual checklist |

## Pass 2026-08-17 — `fetch_data` and the six-file catalogue

The first run of the checklist since #46 replaced `query_csv` with `fetch_data` and grew `data/` to
six files in four formats. Machine: macOS 25.5.0, Python 3.12.13, `claude-opus-5`, `TAVILY_API_KEY`
set. Total wall time 3m31s across two staggered chains.

**Gates (G).** `uv run ruff check .` exit 0, "All checks passed!"; `uv run ruff format --check .`
exit 0, "86 files already formatted"; `uv run mypy` exit 0, "Success: no issues found in 86 source
files"; `uv run pytest` exit 0, **651 passed, 6 deselected in 4.50s**.

**Offline checks (N), no model call.** The catalogue and the planner's roster built directly from
`data_retrieval_tools`; `fetch_data` driven by hand for the repaired name, an unknown name, the
search corpus and a 41 kB file; a new file seen through a `DATA_DIR` copy; `--help`/`--version`;
the config-error path; the `-o json` key set against `cli/format.py`. All exit 0 (the config case
exit 3, as specified). Evidence is on each box above.

**Live model runs (R1–R6), the whole budget.** Prompts chosen to tick several boxes each:

| Run | What | Exit | Outcome |
|---|---|---|---|
| R1 | `uv run pytest -m live` | 1 | 4 passed, **2 failed** — D1 and D2 |
| R2 | linear prompt, stdout redirected | 0 | 3 sequential steps, chart HTML written |
| R3 | fan-out prompt, `-o json` | 0 | 4 steps, both retrievals overlapped, document parses |
| R4 | board pack vs `quarterly_financials` vs `expense_breakdown` | 0 | 3 retrievals, repaired pointer used |
| R5 | fastest-growing product line and what shipped | 0 | `product_lines.json` + `project_timeline.md` |
| R6 | 2025 attrition by department, `-q` | 0 | refused for want of data, no figures invented |

**Still unverified, and why.** Everything needing a real terminal or a human eye: pressing `i` and
the replan conversation, Ctrl-C line discipline and the restored cursor, two spinners live on a
painted frame, and a chart rendered in a browser. No pty was used, so nothing above is ticked on a
simulated terminal. The role-omission CLI run was not made either — the budget went to the #46
criteria, and R1 covers its plan shape live.

### Rerun the same day, at #49

**Gates after the #49 fix.** `uv run ruff check .` exit 0, "All checks passed!"; `uv run ruff format
--check .` exit 0, "86 files already formatted"; `uv run mypy` exit 0, "Success: no issues found in
86 source files"; `uv run pytest` exit 0, **653 passed, 6 deselected in 4.36s** — 651 plus #49's two
regression arms.

**R7 — `uv run pytest -m live`, exit 0, 6 passed, 653 deselected in 82.63s.** All six green, the
first time in this record: `test_the_report_cites_this_run_s_own_numbers_and_chart`,
`test_the_planner_shapes_a_real_plan_to_the_request[linear|fan_out|role_omission]`,
`test_the_planner_asks_only_about_data_the_team_holds`,
`test_the_planner_asks_nothing_it_could_answer_itself`. So D2 is cleared live, and #48 did not
reproduce on this run — it stays open on R1's single observation. One billed run; an earlier attempt
died 6/6 on a `401 … "API key is invalid."` before the key was replaced and produced no model
output, so it is evidence of nothing.

### Defects

**D1 — a key figure can cite an artifact that holds only some of its numbers.** Filed as
[#48](https://github.com/pranav8494/orchestra-py/issues/48). Severity: medium;
it is the traceability claim the README leads with.

```bash
uv run pytest -m live -k aggregation     # exit 1
```

The report stated `Revenue, 2025Q2 to 2025Q4 = "$5.76M → $6.34M → $7.02M (+21.8% overall)"` sourced
to `artifact:fetch_financials.json`. The three revenues are in that retrieval artifact; the
`+21.8%` is not — it was computed by the analytics step. The aggregator's guard checks only that
the pointer names an artifact this run produced, not that the artifact contains the number, so a
compound figure mixing retrieved and computed values gets one pointer and it is wrong for part of
it. Model-output dependent: R2 wrote the same figures citing `analyze_financial_trends.json` and
would have passed. Filed as #48, not fixed. **R7 did not reproduce it**: the same test passed, so
the rate stands at one occurrence in two live runs of that test.

**D2 — the live planner test's `UNAVAILABLE` fixture is stale against #46.** Severity: low, but it
made `pytest -m live` red.

```bash
uv run pytest -m live -k asks_only_about_data     # exit 1
```

`tests/test_planner_scenarios_live.py::UNAVAILABLE` still lists `"headcount"` as a measure no tool
supplies. #46 added `yearly_performance.csv`, whose columns include `headcount_year_end`, so the
planner offering headcount is now correct and the assertion is wrong. The product behaved as
intended; the fixture did not follow the data. Fixed at `cb1dc87`, which drops `"headcount"` from
the list, and confirmed by R7: that test now passes against the live planner.

### Observations

**O1 — `FORCE_COLOR` puts box characters down a pipe.** With `FORCE_COLOR=3` exported (as on this
machine) `console.is_terminal` is `True` for a redirected stdout, so `orchestra run > file` writes
a `Panel`-framed report — the corruption §5 and the README say a pipe must never see. Same root
cause as [#38](https://github.com/pranav8494/orchestra-py/issues/38), whose title only claims the
tests are affected; the runtime symptom is wider. `-o json` is unaffected — JSON is never framed,
and R3's stdout parsed clean.

Filed as [#49](https://github.com/pranav8494/orchestra-py/issues/49) and **fixed in this commit**:
framing reads `console.stdout_is_tty()`, colour still reads `FORCE_COLOR`. Reproduced first in a
real process with a real `> file` redirect and only the model stubbed — the file opened
`╭─ Report ─…`; after the fix the same command writes the bare report, zero box characters.
Regression test `test_cli.py::test_run_frames_the_report_only_on_a_real_tty_even_with_colour_forced`
sets `FORCE_COLOR` itself and fails on the piped arm against the old code.

**O2 — the same variable makes a redirected stderr look interactive.** Outside #49, which is about
stdout, and in different lines: `cli/app._render_mode` and `_interactive()` read
`err_console.is_terminal`. With `FORCE_COLOR=3` and `2>log`, `_render_mode` returns `LIVE`, so a
`Live` region is redrawn into the file; with a tty stdin, `_interactive()` returns `True`, so the
`i` hint and any clarifying question go to the log where nobody can answer them. Both observed
directly on the two functions. Not filed, not fixed — reported for triage.

## Known issues

| Issue | Summary | Status |
|---|---|---|
| [#31](https://github.com/pranav8494/orchestra-py/issues/31) | Non-ASCII on a non-UTF-8 stdout (`PYTHONIOENCODING=ascii`) raises `UnicodeEncodeError`; the whole report is discarded and the run exits 1. The ASCII chart's `█` bars make it deterministic on such a terminal. | Pre-existing, filed, unscheduled |
| [#33](https://github.com/pranav8494/orchestra-py/issues/33) | Visualization can drop an interior category: a three-quarter request charted Q2 and Q4 with the title still claiming Q2–Q4. Values correct, middle point absent. | Filed, unscheduled |
| [#36](https://github.com/pranav8494/orchestra-py/issues/36) | The executor's wall clock never reaches `Config`, so an operator cannot raise it for a legitimately slow analysis. Confirmed while testing the timeout: the only seam is the class reference in `agents/toolsets.py`. | Filed, unscheduled |
| [#48](https://github.com/pranav8494/orchestra-py/issues/48) | A key figure can cite an artifact holding only part of it: a compound "retrieved → computed" figure gets one pointer (D1). Model-output dependent — seen once in two live runs of the test. | Filed, unscheduled |
| [#40](https://github.com/pranav8494/orchestra-py/issues/40) | Tool output is stored and re-sent whole each turn. `fetch_data` now hands anything over 16 kB, or not text, to the analysis step as a pointer, which bounds the worst case; an inlined file and every `search` result are still replayed in full. | Reduced by #46, not closed |
