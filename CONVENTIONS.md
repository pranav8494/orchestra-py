# Conventions — orchestra-py

`orchestra-py` (distribution) · `orchestra` (import) · Python 3.12, pinned in `.python-version`

Rules for all code in this repo. The structural patterns behind them are catalogued in §3.3.

**MUST** = blocks merge. **SHOULD** = the default; deviate only with a comment saying why.
If a rule blocks correct work, change it in a PR. Don't route around it.

---

## 1. Golden rules

1. **Search before you write** (§2). The most important rule here.
2. **Dependencies point inward**: `cli/` → `app.py` → `agents/` → `{core/, tools/, providers/}`.
3. **`core/` is pure** — no SDK, no Rich, no Typer, no I/O.
4. **stdout = output. stderr = diagnostics. Logs = file.** Never mix.
5. **One of each thing** — one `Console`, one `Config`, one tool interface, one provider port.
   A second parallel abstraction is a bug.
6. **Tool failures are data, not exceptions** — the model must be able to read and retry.
7. **Prompts live in `prompts/`**, never inline.
8. **A module you can't describe in one sentence is two modules.**

---

## 2. Search before you write

**MUST** run this before adding any function, class, module, or dependency.

### 2.1 The search

```bash
rg -n "def (parse_|format_|render_)" src/orchestra   # 1. the name you were about to use
rg -ni "retry|backoff|truncat|redact" src/orchestra  # 2. the concept, not the name
ls src/orchestra/core/ && cat src/orchestra/tools/base.py \
   src/orchestra/cli/render.py src/orchestra/config.py   # 3. where shared code lives
# 4. Does an existing function extend to your case? A new parameter beats a new function.
```

Run all four. Stopping early is how duplicates get in.

### 2.2 Reuse ladder

Work down; only reach the bottom when nothing above fits.

| | Action | When |
|---|---|---|
| 1 | **Use** existing | It already does this |
| 2 | **Parameterise** existing | 90% match, difference is a value |
| 3 | **Compose** existing | Your need is a sequence of existing parts |
| 4 | **Extend** the abstraction (new `BaseTool`, `Provider`) | A variant of a known kind |
| 5 | **Write new** | Genuinely new concept, correct layer |

**Never** add a second interface overlapping an existing one. If `BaseTool` doesn't fit, change
`BaseTool` for every implementer in one PR — don't add `BaseTool2`.

### 2.3 When duplication is correct

Don't extract when the copies are on **opposite sides of a layer boundary** (independence outranks
DRY), or are **coincidentally similar** and will diverge. Apply the **rule of three**: extract on the
third instance, when the axis of variation is visible. Deliberate duplication gets a comment.

### 2.4 Declare it

**MUST** — any PR adding a public helper, module, or dependency says what it searched for:

```
Duplicate check: rg'd `truncat|elide|shorten` — nothing. core.format.wrap_text works on
paragraphs, this on single-line tool output. Different axis, kept separate.
```

---

## 3. Layout and layers

### 3.1 Tree

```
src/orchestra/
  app.py             # composition root — wires every service, ONE place
  config.py          # one typed Config; defaults < file < env < flags
  cli/
    app.py           # Typer. Parse, delegate, exit code. NO business logic.
    console.py       # THE two Console objects. Constructed nowhere else.
    render.py        # all Live/Table/Progress; only module importing rich.live
    format.py        # text | json switch
  core/              # PURE — no SDK, no rich, no typer, no network
    state.py         # TaskState, Plan, Subtask (Pydantic)
    events.py        # typed Broker[T]
    loop.py          # repetition detector over tool-call signatures
    question.py      # typed clarification request/answer
    permission.py    # request/grant, auto-approve when non-interactive
    logging.py       # structured logs -> file/stderr, NEVER stdout
    errors.py        # error taxonomy -> exit codes
  providers/         # ONLY place vendor SDKs may be imported
    base.py          # Provider Protocol + factory
  agents/
    planner.py       # decomposition / replanning
    workers/
    toolsets.py      # which tools each agent gets, ONE place
  tools/
    base.py          # BaseTool Protocol + ToolResponse
    question.py      # clarification surfaced as a tool the model can call
  prompts/           # one module per agent + registry
tests/               # conftest.py holds FakeProvider
```

### 3.2 Dependency rule

```
cli/  ──▶  app.py  ──▶  agents/  ──▶  core/ , tools/ , providers/
```

- `core/` imports nothing upward and no third-party SDK, `rich`, or `typer`.
- `tools/`, `providers/` may import `core/`, never `agents/`/`app.py`/`cli/`.
- `cli/` may import anything but holds no business logic.

Enforced by `src/orchestra/core/.ruff.toml` (§14.3). This is what keeps `core/` reusable behind a
different front end.

### 3.3 Pattern catalogue

Each module owns one pattern. If you can't name the pattern, the module is in the wrong place.

| Module | Pattern |
|---|---|
| `app.py` | composition root, constructor injection, no singletons |
| `core/events.py` | typed broker; lossy for progress, must-deliver for lifecycle (§6) |
| `core/state.py` | shared typed ledger, pointers not blobs |
| `core/loop.py` | tool-call signature window, trips on repetition |
| `core/question.py` | typed clarification request, blocks until answered |
| `tools/base.py` | `info()` + `run()`, failures as data |
| `providers/base.py` | port + factory, SDKs quarantined |
| `agents/toolsets.py` | toolset composition in one place |
| `core/permission.py` | request/grant + auto-approve |
| `config.py` | one typed config, layered, loaded once |
| `prompts/` | prompt modules + registry |
| `core/logging.py` | logs off stdout |
| `cli/format.py` | format switch separate from rendering |

---

## 4. CLI layer

**MUST** — commands parse, delegate, and map the result to an exit code. A loop, a domain
conditional, or an LLM call in a command body means it's in the wrong place.

```python
app = typer.Typer(no_args_is_help=True, add_completion=False)

@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="The task to solve.")],
    output: Annotated[OutputFormat, typer.Option("--output", "-o")] = OutputFormat.TEXT,
) -> None:
    result = get_app().run_task(prompt)      # all logic lives there
    emit(result, output)
    raise typer.Exit(result.exit_code)
```

- **MUST** set `no_args_is_help=True`.
- **MUST NOT** call `sys.exit()` — raise `typer.Exit(code)` so cleanup runs.
- **SHOULD** use `Annotated[...]` for anything with help text, a short flag, or validation.
- **SHOULD** keep flags consistent: `--output/-o`, `--quiet/-q`, `--debug`, `--yes/-y`.

---

## 5. Output streams

| Stream | Carries | Never |
|---|---|---|
| stdout | results, JSON — anything pipeable | logs, progress, spinners, warnings |
| stderr | diagnostics, progress, errors | anything a script parses |
| log file | structured, levelled records | — |

`rich.live.Live` owns the terminal while rendering; a concurrent write from a worker corrupts it.
Hence the strictness.

- **MUST NOT** use `print()` in `src/` — enforced by ruff `T20`.
- **MUST** construct `Console` only in `cli/console.py` — two instances, `console` and
  `err_console(stderr=True)`. Never elsewhere.
- **MUST** confine `rich.live` / `rich.progress` to `cli/render.py`; one `Live` at a time. Agents
  publish events to the broker; `render.py` subscribes and draws.
- **MUST** keep pipes working: gate styling on `console.is_terminal`, honour `NO_COLOR` and
  `TERM=dumb`, support `--output {text,json}`. `--output json` emits one JSON document and nothing
  else. `--quiet` suppresses progress, never the result or exit code.

---

## 6. Core abstractions — extend, never parallel

A new tool, provider, or agent means implementing one of these.

**`tools/base.py` — `BaseTool`** · `info()` + `async run(ctx, call) -> ToolResponse`
- **MUST** return failures as `ToolResponse(is_error=True, ...)`, not raise — an exception unwinds
  the agent loop and denies the model its retry. Raise only for programmer errors.
- **MUST** write `info()`'s description as a prompt, not a docstring: when to use it, when not, limits.
- **SHOULD** put structured detail in `metadata`, not in prose the model must parse.

**`providers/base.py` — `Provider`** · `send()`, `stream()`, `model`
- **MUST** import vendor SDKs only here. Other layers speak our types, never the SDK's.
- Adding a provider = one module + one factory branch. If it requires touching `agents/`, the
  abstraction leaked — fix that.

**`core/events.py` — `Broker[T]`** · two publish modes, deliberately
- **MUST** publish progress non-blocking: drop to full subscribers rather than stall the agent loop.
  Deliberate — don't "fix" it by awaiting.
- **MUST** publish lifecycle events (`plan_created`, `subtask_completed`, `subtask_failed`,
  `run_finished`) must-deliver: bounded blocking with a timeout. A dropped completion event strands
  the dashboard on a spinner forever.
- **MUST** unsubscribe on cancellation.

**`config.py` — `Config`** · one typed object, loaded once, injected
- Precedence: `defaults < file < env < flags`.
- **MUST NOT** read `os.environ` outside `config.py`.

**`core/state.py` — `TaskState`** · the shared ledger
- **MUST** be Pydantic and fully typed.
- **SHOULD** hold pointers to large artifacts, never inline blobs — state gets serialised and fed to
  models.
- **SHOULD** give each agent only its slice.

---

## 7. Typing

| Use | For |
|---|---|
| Pydantic model | trust boundaries: LLM output, tool params, config, persisted state |
| `@dataclass(frozen=True, slots=True)` | internal value objects |
| `Protocol` | ports (`BaseTool`, `Provider`, `Agent`) |
| `StrEnum` | any closed set of strings — never bare literals for states or roles |

- **MUST** annotate every public function; mypy runs strict (§14.2).
- **MUST NOT** use `Any` in `core/`, `tools/`, `agents/`. At an adapter it's fine at the SDK edge —
  parse into our types immediately, never let it travel inward.
- **MUST** validate LLM output through Pydantic before it touches state.
- **SHOULD NOT** subclass for reuse; prefer composition and `Protocol`.

---

## 8. Errors and exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | unhandled exception (a bug) |
| 2 | usage error (Typer default) |
| 3 | configuration error |
| 4 | provider error after retries |
| 5 | task failure |
| 130 | SIGINT |

All exceptions derive from `OrchestraError` in `core/errors.py` and carry their exit code.

- **MUST** exit non-zero on failure.
- **MUST** catch and render at one place — the CLI boundary. Don't format messages in `core/`.
- **MUST NOT** show a traceback unless `--debug`. Give the message, the cause, the fix.
- **MUST NOT** swallow exceptions — no bare `except:`, no `except Exception: pass`.
- **MUST** handle `KeyboardInterrupt`: cancel tasks, restore the terminal, exit 130. An un-exited
  `Live` region corrupts the user's shell.

---

## 9. Config and secrets

- **MUST** load config once at startup into one typed object.
- **MUST** read secrets from env only; never commit them or write them to a generated config file.
- **MUST NOT** log a secret or print an unredacted config under `--debug`.
- **SHOULD** store user data under `~/.orchestra/`, never the working directory.
- **SHOULD** fail fast — validate required config at startup, not four minutes into a run.

---

## 10. Async

- **SHOULD** be async at I/O boundaries; keep `core/` logic synchronous.
- **MUST NOT** block the event loop — use `asyncio.to_thread`.
- **MUST** bound concurrency with a semaphore; never unbounded fan-out.
- **MUST** make long operations cancellable via `asyncio.TaskGroup`. A run the user can't stop is a
  defect.
- **MUST** propagate `CancelledError` — clean up and re-raise.
- **MUST** cap every agent loop with max iterations and a token budget; exceeding either is a
  `TaskFailure`, not an infinite retry.

---

## 11. Prompts

- **MUST** live in `prompts/`, one module per agent, via the registry. Never inline.
- **SHOULD** compose as `base + role + context` rather than duplicating preamble. §2 applies to
  prompt text — copy-pasted prompt blocks drift and are the hardest duplication to spot.
- **SHOULD** keep runtime formatting out of prompt modules.

---

## 12. Testing

- **MUST NOT** hit the network — use the `FakeProvider` fixture in `tests/conftest.py`.
- **MUST** test commands with `CliRunner`, asserting exit code, stdout, and stderr **separately**:

  ```python
  result = runner.invoke(app, ["run", "task", "--output", "json"])
  assert result.exit_code == 0
  json.loads(result.stdout)   # stdout must be JSON, alone
  ```

  Asserting only stdout is how stream-contract regressions ship.
- **MUST** add a failing regression test with every bug fix.
- **SHOULD** test orchestration logic, state transitions, tool contracts, error paths, and
  cancellation. Don't test Rich's rendering — assert on the data handed to the renderer.
- **SHOULD** name tests `test_<unit>_<condition>_<expected>`.

---

## 13. Dependencies

- **MUST** justify each new runtime dependency in the PR: what it does, why not stdlib, its weight.
  Runtime deps ship to users; dev deps are judged more leniently.
- **MUST** commit `uv.lock`.
- **MUST NOT** import a vendor LLM SDK outside `providers/`.
- **SHOULD** prefer stdlib — `pathlib`, `dataclasses`, `enum`, `asyncio`, `tomllib`.

---

## 14. Tooling

### 14.1 Commands

`.python-version` pins the interpreter and uv downloads it — don't use system Python. Versions come
from the committed `uv.lock`, so local, pre-commit, and CI resolve identically.

```bash
uv sync                            # env + deps
uv run orchestra --help            # run
uv run pytest                      # tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run pre-commit run --all-files
```

### 14.2 What the gates enforce

| Rule | Enforces |
|---|---|
| ruff `T20` | no `print()` (§5) |
| ruff `TID` | layer rule (§3.2) |
| ruff `F401`/`F841`/`ARG`/`ERA` | dead and commented-out code |
| ruff `ASYNC`, `B`, `ANN` | async footguns, common bugs, missing annotations |
| mypy `strict` | types — relaxed for `orchestra.providers.*` only, scoped by module path |

### 14.3 Layer rule

`src/orchestra/core/.ruff.toml` extends the root config and bans upward and vendor imports in
`core/`, failing at commit rather than review. If contracts outgrow banned-imports, move to
[import-linter](https://import-linter.readthedocs.io/).

### 14.4 Duplicate gate

pre-commit runs pylint R0801 over `src/`. It catches mechanical copy-paste only — it cannot catch a
reimplementation under a different name. **It is a backstop, not a substitute for §2.**

---

## 15. Git and review

Imperative commit subjects under 72 chars, optionally Conventional Commits. One logical change per
commit; formatting-only changes get their own.

**PR checklist**

- [ ] Duplicate check declared (§2.4)
- [ ] Layer rule respected; `core/` still clean
- [ ] No `print()`; correct stream (§5)
- [ ] Right exit code; no bare `except`
- [ ] New tools implement `BaseTool`; new providers implement `Provider`
- [ ] Tests cover happy path, one error path, cancellation if async
- [ ] `ruff`, `mypy`, `pytest` pass
- [ ] New runtime deps justified

**Review order**: correctness → layer boundaries → duplication → style. Style is last; ruff has it.
