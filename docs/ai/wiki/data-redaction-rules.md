# Data Redaction Rules

> Rules for removing private/sensitive data before it enters prompts, logs, PRs, or channels.

## What counts as private data

| Category | Examples | Action |
|---|---|---|
| API keys | `sk-...`, `ghp_...`, `AKIA...` | NEVER in any file, log, or message |
| Passwords | `password=...`, `secret=...` | Use env vars or secret manager |
| Personal names | Client/customer names | Replace with "the client" or initials |
| Emails | Personal emails | Replace with `user@example.com` |
| Phone numbers | Any phone number | Remove entirely |
| Internal URLs | `staging.internal.company.com` | Remove or replace with `<internal-url>` |
| Database DSNs | `postgres://user:pass@host` | Remove credentials, keep host if non-sensitive |
| Client identifiers | `client_id=...`, `customer_id=...` | Hash or replace with `<redacted>` |
| Financial data | Revenue, MRR, customer counts | Round to nearest order of magnitude or omit |
| Source code snippets | From private repos | Only include if you have permission |

## Redaction process

1. **Before sending to any AI tool**: scan with `python3 ai/evals/test_safety_gate.py <file>`
2. **Before committing**: pre-commit hook scans diff
3. **Before merging**: PR template requires "no secrets" checkbox

## Exemptions

- Environment variables (`.env` files) are gitignored and never committed
- Test fixtures may use fake data (`test_user@example.com`, `sk-test-123`)
- Public documentation may reference public URLs
