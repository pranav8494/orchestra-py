# Test results (#16)

Automated suite (offline, the four gates):

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Live tests opt in: `uv run pytest -m live` (needs `ANTHROPIC_API_KEY`; `TAVILY_API_KEY` optional —
without it the search tool reads the bundled corpus instead of the web).

## Manual live-LLM checklist

Run from a real terminal with `ANTHROPIC_API_KEY` set. `$?` is checked after each run.

### Planner scenarios

- [ ] `uv run orchestra run "Summarize the last 3 quarters financial trends and create a chart"` → 3 steps, sequential: retrieval → analytics → visualization; exit 0
- [ ] `uv run orchestra run "Compare our last 3 quarters of revenue growth against industry benchmarks and chart the trend"` → 4–5 steps, two retrievals with no edge between them, then analytics, then visualization; exit 0
- [ ] `uv run orchestra run "Summarize the revenue trend in one paragraph"` → 2 steps, **no** visualization step; exit 0
- [ ] `uv run pytest -m live` → live planner and aggregation tests pass

### Fan-out concurrency (#17's outstanding criterion)

- [ ] Rerun the fan-out prompt above and watch the dashboard → both retrieval agents show as running **at the same time**, two spinners live before either completes
- [ ] Same run, on the event log → the second `subtask_started` appears before the first `subtask_completed`

### Clarification (#10)

- [ ] `uv run orchestra run "Make a chart of performance"` → one clarifying question before any plan is drawn
- [ ] Answer it once → planning proceeds without a second question; run reaches a report; exit 0
- [ ] Question offers only measures the team can actually retrieve (no metric no tool supplies)

### Mid-run interrupt and replan (#12)

- [ ] Start the linear prompt; stderr shows `Press i to interrupt the run and talk to the orchestrator.`
- [ ] Press `i` mid-run → dashboard pauses, chat prompt opens, in-flight step is not lost
- [ ] Ask for a changed goal (for example "skip the chart, just summarise") → orchestrator replans; completed steps are kept, unfinished ones replaced
- [ ] Resume → run finishes on the **new** plan; report reflects it; exit 0

### Cancellation

- [ ] Start any run, press Ctrl-C mid-execution, then `echo $?` → prints `130`
- [ ] Same shell afterwards: type a command → characters echo normally, Enter works (cbreak handed back)
- [ ] No traceback on stderr; a single `Interrupted.` line
- [ ] Terminal cursor is visible and the Live region is torn down (no stuck dashboard frame)

### Output modes

- [ ] `uv run orchestra run "<linear prompt>" -o json | python -c "import json,sys; json.load(sys.stdin)"` → parses; exactly one document; no progress text on stdout
- [ ] `uv run orchestra run "<linear prompt>" -q` → no progress lines; the report still prints on stdout
- [ ] `uv run orchestra run "<linear prompt>" > /tmp/out.txt` → progress still drawn on stderr, report lands in the file

### Artifacts

- [ ] After a charted run, open the chart in `~/.orchestra/artifacts/<run>/*.html` (each run gets its own timestamped subdirectory) → renders in a browser with this run's numbers, axes and title
- [ ] Report's ASCII chart matches the HTML chart's shape

## Automated results

`uv run pytest` — **622 passed, 6 deselected** (live). Baseline before this ticket was 607.
`ruff check`, `ruff format --check` and `mypy` (strict, 86 files) all clean.

Everything below runs offline against `FakeProvider`: the real planner, workers, tools over
`data/`, aggregator and Typer command, with only the provider port substituted.

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

## Known issues

| Issue | Summary | Status |
|---|---|---|
| [#31](https://github.com/pranav8494/orchestra-py/issues/31) | Non-ASCII on a non-UTF-8 stdout (`PYTHONIOENCODING=ascii`) raises `UnicodeEncodeError`; the whole report is discarded and the run exits 1. The ASCII chart's `█` bars make it deterministic on such a terminal. | Pre-existing, filed, unscheduled |
| [#36](https://github.com/pranav8494/orchestra-py/issues/36) | The executor's wall clock never reaches `Config`, so an operator cannot raise it for a legitimately slow analysis. Confirmed while testing the timeout: the only seam is the class reference in `agents/toolsets.py`. | Filed, unscheduled |
