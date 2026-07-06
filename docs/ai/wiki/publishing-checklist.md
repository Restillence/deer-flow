# Publishing Checklist

> Rules for publishing docs, blog posts, and public content.
> Agents MUST read this page before generating public-facing content.

## Before publishing

1. **Source trust ladder** — every claim must link to a source at trust level ≤ 3:
   - Level 0: Private raw data → ❌ NEVER in public content
   - Level 1: Redacted internal note → ❌ Not for public
   - Level 2: Wiki page with refs → ⚠️ Internal only
   - Level 3: Public primary source → ✅ Cite directly
   - Level 4: Generated summary → ⚠️ Back with Level 3 source
   - Level 5: Unsourced model output → ❌ NEVER

2. **No secrets or private data** — run the safety gate eval before publishing:
   ```bash
   python3 ai/evals/test_safety_gate.py <content_file>
   ```

3. **No client/customer names** — use "a customer", "a client", or get explicit permission.

4. **Link check** — all links resolve (use CI link checker or `lychee`).

5. **License check** — only include code/images you have rights to.

6. **Fact check** — verify all technical claims against current docs (not memory).

## Content types

| Type | Reviewer | Additional checks |
|---|---|---|
| Blog post | Human editor | Tone, accuracy, no client data |
| README/docs | Tech reviewer | Code examples run correctly |
| Chelog | Maintainer | Matches actual git history |
| Social media | Human approval | No private context, no unverified claims |

## Redaction rules

See `docs/ai/wiki/data-redaction-rules.md` for what must be removed before publishing.
