#!/usr/bin/env python3
"""
Audit trail logger — append-only JSONL for every autonomous AI run.
Log format matches the spec: tool, task_type, prompt_sha, source_refs,
branch, commit_sha, merged_sha, duration, tokens, human_approved.

Usage (from any agent/tool):
    python3 scripts/audit-log.py log --tool opencode --task-type code_change \
        --branch feat/xyz --commit-sha abc123

    python3 scripts/audit-log.py list --limit 10
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / ".ai" / "ai_runs.jsonl"


def ensure_log():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.touch()


def log_run(args):
    ensure_log()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": args.tool,
        "task_type": args.task_type,
        "prompt_sha": hashlib.sha256(args.prompt.encode()).hexdigest()[:16]
        if args.prompt
        else None,
        "source_refs": args.source_refs.split(",") if args.source_refs else [],
        "branch": args.branch,
        "commit_sha": args.commit_sha,
        "merged_sha": args.merged_sha,
        "duration_s": int(args.duration) if args.duration else None,
        "tokens": int(args.tokens) if args.tokens else None,
        "human_approved": args.human_approved,
        "status": args.status or "unknown",
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Logged: {entry['tool']} / {entry['task_type']} → {entry['status']}")


def list_runs(args):
    ensure_log()
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        print("(no runs logged yet)")
        return
    lines = LOG_FILE.read_text().strip().split("\n")
    entries = [json.loads(line) for line in lines if line.strip()]
    # Sort by timestamp descending
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    limit = int(args.limit)
    for e in entries[:limit]:
        ts = e.get("timestamp", "?")[:19]
        tool = e.get("tool", "?")
        task = e.get("task_type", "?")
        status = e.get("status", "?")
        branch = e.get("branch", "-")
        approved = "✅" if e.get("human_approved") else "⏳"
        print(
            f"{ts} | {tool:15s} | {task:18s} | {status:8s} | {branch:25s} | {approved}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI run audit log")
    sub = parser.add_subparsers(dest="command")

    log_p = sub.add_parser("log", help="Log a run")
    log_p.add_argument("--tool", required=True)
    log_p.add_argument("--task-type", required=True)
    log_p.add_argument("--prompt", default="", help="Prompt text (hashed, not stored)")
    log_p.add_argument("--source-refs", default="", help="Comma-separated source links")
    log_p.add_argument("--branch", default="")
    log_p.add_argument("--commit-sha", default="")
    log_p.add_argument("--merged-sha", default="")
    log_p.add_argument("--duration", default="", help="Duration in seconds")
    log_p.add_argument("--tokens", default="", help="Token count")
    log_p.add_argument("--human-approved", action="store_true")
    log_p.add_argument("--status", default="", help="success/failed/blocked")

    list_p = sub.add_parser("list", help="List recent runs")
    list_p.add_argument("--limit", default="20")

    args = parser.parse_args()
    if args.command == "log":
        log_run(args)
    elif args.command == "list":
        list_runs(args)
    else:
        parser.print_help()
