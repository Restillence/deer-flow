# Review Preferences

> Durable rules for how code should be reviewed in this project.
> Agents MUST read this page before generating or reviewing code.

## What reviewers care about

1. **Correctness first** — does it do what the task requires? Tests prove it.
2. **Small diffs** — target < 400 lines (excluding lock files). Split if bigger.
3. **No drive-by edits** — don't fix unrelated issues in the same PR. File a follow-up issue.
4. **Error handling** — every external call (HTTP, DB, file I/O) handles failure explicitly.
5. **Type safety** — all new code uses Python type hints. `str | None`, not `Optional[str]`.

## Python-specific

- Use `uv` for all dependency management. Never edit `uv.lock` manually.
- Functions ≤ 40 lines. If longer, extract.
- No `import *` — explicit imports only.
- f-strings for formatting, not `.format()` or `%`.
- Dataclasses or Pydantic models for structured data, not raw dicts.
- `logging` module, never `print()` for production code.

## Monorepo (content-ecosystem)

- Each package (`packages/*/`) owns its tests.
- Cross-package changes need tests in ALL affected packages.
- Workspace dependencies: `from comfy_agent import ...` (not relative paths across packages).

## What blocks a review

- ❌ Missing tests for new behavior
- ❌ Secrets in diff
- ❌ Unrelated changes mixed in
- ❌ "Trust me" claims without test output or CI link
- ❌ Migration without rollback
- ❌ New dependency without approval

## What speeds up review

- ✅ PR description with files changed + reasons
- ✅ Test output pasted or linked
- ✅ Risk assessment filled in
- ✅ Screenshots for UI changes
- ✅ Rollback plan for non-low-risk changes
