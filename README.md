# orchestra-py
A CLI multi-agent orchestrator that breaks a plain-language business request into subtasks, routes them to role-specialised AI agents, executes real tools, and streams live progress to a structured result.

## The planner is dynamic

A three-role pipeline can look dynamic while always emitting the same chain. These three requests are the proof that it does not — the plan's shape follows the request:

| Request | Expected plan |
|---|---|
| `Summarize the last 3 quarters financial trends and create a chart` | 3 steps, sequential: retrieval → analytics → visualization |
| `Compare our last 3 quarters of revenue growth against industry benchmarks and chart the trend` | 4–5 steps: two retrievals with no edge between them — our CSV and web search — running at once, then the comparison (one step or two), then visualization |
| `Summarize the revenue trend in one paragraph` | 2 steps, **no visualization** — the role is dropped, not reordered |

The fan-out request names two subjects held in two places, because the planner is told what the team can retrieve: asking for two date ranges of one CSV is legitimately one step, and it plans it as one.

```bash
uv run pytest tests/test_planner_scenarios.py            # shapes, offline
uv run pytest -m live tests/test_planner_scenarios_live.py   # the same shapes against the real model
```

During the fan-out run the dashboard shows both retrieval agents spinning together; `tests/test_engine.py` asserts the same thing on the event stream, where the second `subtask_started` lands before either `subtask_completed`.

## Running model-written code

The Analytics agent computes by writing Python and running it, rather than by calling a fixed set of typed helpers — analysis is open-ended, and an enumeration of operations cannot answer the question outside it. The trade-off:

| Guard | What it buys |
|---|---|
| Subprocess with a 15 s wall clock, killed by process group | a runaway loop ends the call, not the run |
| Scrubbed environment (`-I`, allow-listed vars) | the run's API keys never reach the script |
| Throwaway working directory, also its `HOME`/`TMPDIR` | nothing is left behind for the next script or the user's home |
| `socket` guard in the child | no accidental network call mid-analysis |

**This is isolation, not a sandbox.** It raises the cost of an accident; it does not contain hostile code — a script can undo a monkeypatch, and real containment means a container, a seccomp profile or a separate uid. The code being run comes from our own model and our own prompt, and untrusted input is out of scope for this project.
