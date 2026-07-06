# Private AI Boundaries

> Defines what AI agents MAY and MAY NOT do in this workspace.
> Read before any autonomous action.

## ✅ ALLOWED (without asking)

- Read files, search code, understand the codebase
- Run tests, lint, type checks, evals
- Write code, docs, config on feature branches
- Commit and push to feature branches
- Open PRs
- Merge PRs when CI is green AND human has approved
- Run research (web_search, web_fetch)
- Deploy behind a feature flag with rollback plan
- Run cron jobs for monitoring, health checks, updates

## 🛑 HARD BLOCKED (always need explicit Telegram approval)

- Any DB migration or schema change (requires tested forward + backward rollback)
- Anything that could expose a secret or private client data
- Deleting or force-pushing to `main` or `dev`
- Installing new system packages without approval
- Changing IAM/permissions/access controls
- Sending messages to external services (email, Slack, public APIs)
- Merging to `main` without human review
- Anything costing money (cloud resources, API calls beyond free tier)

## ⚠️ SOFT BLOCKED (ask once, then autonomous within scope)

- Adding a new dependency (ask first, then fine for similar packages)
- Creating new GitHub branches/repos
- Modifying CI workflow files
- Large refactors (> 400 lines)

## Kill switch

If `~/.openclaw/workspace/.openclaw-stop` marker file exists, ALL autonomous work halts.
Create it with: `touch ~/.openclaw/workspace/.openclaw-stop`
Remove it with: `rm ~/.openclaw/workspace/.openclaw-stop`

See also: `docs/ai/wiki/data-redaction-rules.md` for what constitutes private data.
