# CLAUDE.md

**[CONVENTIONS.md](./CONVENTIONS.md) governs all code here. Read it before writing anything.**

## Non-negotiables

1. **Search before you write** (§2). Run it before any new function, class, or module — small
   additions are how duplication accumulates. State what you searched for and didn't find.
2. **Layers point inward** (§3.2): `cli/` → `app.py` → `agents/` → `{core/, tools/, providers/}`.
   `core/` imports nothing upward and no vendor SDK.
3. **Never `print()`** (§5). stdout = output, stderr = diagnostics, logs = file. Use the console
   objects in `cli/console.py`; never construct another `Console`.
4. **Extend the existing abstractions, never parallel them** (§6). New tool → `BaseTool`. New
   provider → `Provider`. If the contract doesn't fit, change it for every implementer in one PR.
5. **Tool failures return as data**, not exceptions — the model must be able to read and retry (§6).

## Before claiming done

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

Don't report work complete without seeing these pass.

## Writing style

Keep everything you write short and concise — chat replies, commit messages, PR descriptions, docs,
and tickets. No over-explanation, no verbose overload. State the change and the why in a line or
two; the reader will ask if they want more.

- **Commit**: imperative subject under 72 chars, plus at most 2–3 lines of body. Skip the body when
  the subject says it.
- **PR**: what changed, why, how to verify. Nothing else.
- **Docs**: prefer a table or a short list over prose.

## Context

Rationale and the pattern catalogue are in CONVENTIONS.md. Multi-agent design research is in
`docs/plan/Multi_Agent_Task_Solver_Research.md`.
