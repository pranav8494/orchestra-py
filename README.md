# orchestra-py
A CLI multi-agent orchestrator that breaks a plain-language business request into subtasks, routes them to role-specialised AI agents, executes real tools, and streams live progress to a structured result.

## Running model-written code

The Analytics agent computes by writing Python and running it, rather than by calling a fixed set of typed helpers — analysis is open-ended, and an enumeration of operations cannot answer the question outside it. The trade-off:

| Guard | What it buys |
|---|---|
| Subprocess with a 15 s wall clock, killed by process group | a runaway loop ends the call, not the run |
| Scrubbed environment (`-I`, allow-listed vars) | the run's API keys never reach the script |
| Throwaway working directory, also its `HOME`/`TMPDIR` | nothing is left behind for the next script or the user's home |
| `socket` guard in the child | no accidental network call mid-analysis |

**This is isolation, not a sandbox.** It raises the cost of an accident; it does not contain hostile code — a script can undo a monkeypatch, and real containment means a container, a seccomp profile or a separate uid. The code being run comes from our own model and our own prompt, and untrusted input is out of scope for this project.
