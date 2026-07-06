# Source Trust Ladder

> Lower trust level wins. When two sources conflict, the one with the lower number is more trustworthy.
> Agents MUST cite sources at level ≤ 3 for public claims.

## The Ladder

| Level | Source type | Example | Usable in public? |
|---|---|---|---|
| 0 | Private raw data | DB dumps, user logs, internal Slack | ❌ NEVER |
| 1 | Redacted internal note | Memory file with PII removed | ❌ Internal only |
| 2 | Wiki page with references | `docs/ai/wiki/review-preferences.md` | ⚠️ Internal |
| 3 | Public primary source | Official docs, RFC, source code, changelog | ✅ Yes — cite directly |
| 4 | Generated summary | AI-generated overview backed by Level 3 | ⚠️ Back with Level 3 |
| 5 | Unsourced model output | "I think this is correct" with no source | ❌ Never cite |

## Rules

1. **Every factual claim in public content links to a Level ≤ 3 source.**
2. **When Level 4 (generated) and Level 3 (primary) conflict, Level 3 wins.**
3. **Level 5 (unsourced) is never acceptable as the sole basis for a claim.**
4. **Internal decisions can cite Level 1-2, but must note the trust level.**
5. **When researching with web_search**: the search result is Level 4 until you fetch the primary source (Level 3).

## How to use this in practice

```python
# ❌ BAD — Level 5, unsourced
"This framework is the fastest option."

# ⚠️ OK — Level 4, needs a source
"According to benchmarks, this framework is fast (source: <link to benchmark>)."

# ✅ BEST — Level 3, primary source
"The official documentation states this method returns a list [1]."
# [1] https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob
```

## Integration

- `ai/router.yml` requires source links for `docs_or_blog` task class
- `docs/ai/wiki/publishing-checklist.md` enforces the ladder before publishing
- `ai/evals/test_fake_certainty.py` catches Level 5 claims in PR descriptions
