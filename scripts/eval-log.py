#!/usr/bin/env python3
"""
Eval logger — append-only JSONL for every pre-commit and CI eval run.

Tracks fake-certainty and safety-gate violations over time so you can
see if the AI stack is actually reducing issues.

Usage:
    python3 scripts/eval-log.py log \\
        --gate pre-commit \\
        --eval safety_gate \\
        --findings '{"secrets": [], "private_context": [], ...}' \\
        --files "src/auth.py,tests/test_auth.py" \\
        --status pass

    python3 scripts/eval-log.py log \\
        --gate pre-commit \\
        --eval safety_gate \\
        --findings '{"secrets": [{"type": "OpenAI key", "match": "sk-..."}]}' \\
        --files "config.py" \\
        --status fail

    python3 scripts/eval-log.py summary
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / ".ai" / "eval_log.jsonl"


def ensure_log() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.touch()


def log_eval(args: argparse.Namespace) -> None:
    ensure_log()
    findings = json.loads(args.findings) if args.findings else {}

    # Count total issues across all finding categories
    issue_count = 0
    if isinstance(findings, dict):
        for v in findings.values():
            if isinstance(v, list):
                issue_count += len(v)
            elif isinstance(v, bool) and v:
                issue_count += 1

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate": args.gate,  # pre-commit, ci, manual
        "eval": args.eval,  # safety_gate, fake_certainty
        "status": args.status,  # pass, fail
        "issue_count": issue_count,
        "files": args.files.split(",") if args.files else [],
        "branch": args.branch or "",
        "commit_sha": args.commit_sha or "",
        "findings": findings,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"✅ Eval logged: {args.gate}/{args.eval} → {args.status} ({issue_count} issues)"
    )


def summary(args: argparse.Namespace) -> None:
    ensure_log()
    raw = LOG_FILE.read_text().strip()
    if not raw:
        print("(no eval runs logged yet)")
        return

    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    total = len(entries)
    passed = sum(1 for e in entries if e.get("status") == "pass")
    failed = total - passed
    total_issues = sum(e.get("issue_count", 0) for e in entries)

    # Breakdown by gate
    by_gate: dict[str, dict[str, int]] = {}
    for e in entries:
        gate = e.get("gate", "?")
        if gate not in by_gate:
            by_gate[gate] = {"total": 0, "pass": 0, "fail": 0, "issues": 0}
        by_gate[gate]["total"] += 1
        by_gate[gate]["pass" if e.get("status") == "pass" else "fail"] += 1
        by_gate[gate]["issues"] += e.get("issue_count", 0)

    # Breakdown by eval type
    by_eval: dict[str, dict[str, int]] = {}
    for e in entries:
        ev = e.get("eval", "?")
        if ev not in by_eval:
            by_eval[ev] = {"total": 0, "pass": 0, "fail": 0}
        by_eval[ev]["total"] += 1
        by_eval[ev]["pass" if e.get("status") == "pass" else "fail"] += 1

    # Recent trend (last 10 runs)
    recent = entries[-10:]
    recent_fail_rate = (
        sum(1 for e in recent if e.get("status") == "fail") / len(recent) * 100
        if recent
        else 0
    )

    print("=" * 60)
    print("  AI Eval Trend Summary")
    print("=" * 60)
    print(f"  Total runs:    {total}")
    print(
        f"  Passed:        {passed} ({passed / total * 100:.0f}%)"
        if total
        else "  Passed: 0"
    )
    print(f"  Failed:        {failed}")
    print(f"  Total issues:  {total_issues}")
    print()
    print("  By gate:")
    for gate, stats in sorted(by_gate.items()):
        rate = stats["pass"] / stats["total"] * 100 if stats["total"] else 0
        print(
            f"    {gate:15s}  {stats['pass']}/{stats['total']} pass "
            f"({rate:.0f}%)  {stats['issues']} issues"
        )
    print()
    print("  By eval:")
    for ev, stats in sorted(by_eval.items()):
        rate = stats["pass"] / stats["total"] * 100 if stats["total"] else 0
        print(f"    {ev:20s}  {stats['pass']}/{stats['total']} pass ({rate:.0f}%)")
    print()
    print(f"  Recent fail rate (last {len(recent)} runs): {recent_fail_rate:.0f}%")
    print()
    if entries:
        print(f"  First run: {entries[0].get('timestamp', '?')[:19]}")
        print(f"  Last run:  {entries[-1].get('timestamp', '?')[:19]}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI eval trend logger")
    sub = parser.add_subparsers(dest="command")

    log_p = sub.add_parser("log", help="Log an eval run")
    log_p.add_argument("--gate", required=True, help="pre-commit, ci, manual")
    log_p.add_argument("--eval", required=True, help="safety_gate, fake_certainty")
    log_p.add_argument("--status", required=True, help="pass, fail")
    log_p.add_argument("--findings", default="{}", help="JSON findings dict")
    log_p.add_argument("--files", default="", help="Comma-separated file list")
    log_p.add_argument("--branch", default="")
    log_p.add_argument("--commit-sha", default="")

    sum_p = sub.add_parser("summary", help="Show eval trend summary")

    args = parser.parse_args()
    if args.command == "log":
        log_eval(args)
    elif args.command == "summary":
        summary(args)
    else:
        parser.print_help()
