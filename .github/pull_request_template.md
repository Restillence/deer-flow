<!-- AI-Assisted PR Template — content-ecosystem -->
<!-- This template is loaded automatically when a PR is created. -->

## 🤖 AI Assistance Disclosure

- [ ] This PR was generated or assisted by an AI tool
- **Tool used**: <!-- opencode / claude-code / codex / manual -->
- **Task type**: <!-- code_change / dependency_bump / ui_change / docs_or_blog -->

## 🧠 Knowledge Context
- [ ] Ran `memory_search` / `gbrain query` before implementation, or marked N/A below
- **Context query / reason N/A**: <!-- query used, relevant result, or why not applicable -->
- [ ] Repo-local vector index is fresh (`python scripts/build-ai-vector-index.py --check`)

## 🧰 gstack Loop
- [ ] Spec/plan review run, or N/A
- [ ] Review/health/QA run as applicable, or N/A
- [ ] Ship/canary/docs path selected, or deferred with reason
- **gstack notes**: <!-- commands used or reason N/A -->

## 📋 Files Changed
<!-- List each file with a one-line reason -->
-
-

## 🧪 Tests
- [ ] Existing tests pass (`uv run pytest`)
- [ ] New tests cover the changed behavior
- **Test output**: <!-- paste summary or link to CI run -->
- **Coverage**: <!-- % on changed lines -->

## ✅ CI
- [ ] All required checks green (test, lint, security)
- **CI run**: <!-- link -->

## ⚠️ Risk Assessment
<!-- Check all that apply -->
- [ ] Low — no DB/auth/job/API changes
- [ ] Medium — changes a non-critical path
- [ ] High — changes DB schema, auth, or external API

**Affected areas**: <!-- DB / auth / jobs / API / none -->

## 🔄 Rollback Plan
<!-- Required if risk > low. Include tested rollback steps. -->
-

## 🔍 Review
- [ ] Code reviewed by human
- [ ] No secrets or private data in diff
- [ ] Diff is small and reviewable (< 400 lines excluding lock files)

## 📝 Notes
<!-- Anything else reviewers should know -->
