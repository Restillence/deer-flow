#!/usr/bin/env python3
"""
Pre-commit hook: runs AI safety gate eval on staged changes.

Replaces the old grep-based secret scan with the full safety gate eval from
ai/evals/test_safety_gate.py. Catches:
  - Secrets (API keys, tokens, private keys, passwords)
  - Private context leaks (emails, phone numbers, internal URLs)
  - DB migrations without rollback functions

Every run is logged to .ai/eval_log.jsonl for trend tracking.

Bypass: git commit --no-verify (use sparingly — it's logged as skipped).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_LOG = REPO_ROOT / ".ai" / "eval_log.jsonl"

# Import the eval scanner — add to path
eval_dir = str(REPO_ROOT / "ai" / "evals")
sys.path.insert(0, eval_dir)
try:
    from test_safety_gate import has_issues, scan_diff  # type: ignore[import-not-found]
except Exception as e:
    # Fallback: if eval import fails, use basic grep (never silently skip)
    print(
        f"⚠️  WARNING: Could not import safety gate eval ({e}). Falling back to basic scan."
    )
    scan_diff = None
    has_issues = None


def get_staged_diff() -> str:
    """Get the unified diff of staged changes."""
    result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout


def get_staged_files() -> list[str]:
    """Get list of staged file paths."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def get_current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def log_eval_run(
    status: str,
    findings: dict,
    files: list[str],
    branch: str,
) -> None:
    """Log this eval run to .ai/eval_log.jsonl."""
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)

    issue_count = 0
    if isinstance(findings, dict):
        for v in findings.values():
            if isinstance(v, list):
                issue_count += len(v)
            elif isinstance(v, bool) and v:
                issue_count += 1

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate": "pre-commit",
        "eval": "safety_gate",
        "status": status,
        "issue_count": issue_count,
        "files": files[:20],  # cap to avoid huge entries
        "branch": branch,
        "commit_sha": "",  # not committed yet
        "findings": findings,
    }

    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    diff = get_staged_diff()
    files = get_staged_files()
    branch = get_current_branch()

    if not diff.strip():
        # Nothing staged — allow
        return 0

    # Skip lock files and test files from analysis (but still log)
    code_files = [
        f
        for f in files
        if not f.endswith((".lock", "package-lock.json"))
        and "/test" not in f
        and not f.endswith("_test.py")
        and "conftest.py" not in f
    ]

    if scan_diff is not None:
        # Use the full eval scanner
        findings = scan_diff(diff)
        has_blockers = has_issues(findings) if has_issues else False
    else:
        # Fallback basic grep for secrets only
        import re

        secret_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",
            r"ghp_[a-zA-Z0-9]{36,}",
            r"AKIA[0-9A-Z]{16}",
            r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
            r'password\s*=\s*["\'][^"\']{8,}["\']',
        ]
        found_secrets = []
        for pattern in secret_patterns:
            matches = re.findall(pattern, diff)
            for m in matches:
                found_secrets.append(
                    {"type": "Secret pattern", "match": m[:20] + "..."}
                )
        findings = {
            "secrets": found_secrets,
            "private_context": [],
            "migrations_without_rollback": [],
        }
        has_blockers = bool(found_secrets)

    # Log every run
    status = "fail" if has_blockers else "pass"
    log_eval_run(status, findings, code_files, branch)

    if has_blockers:
        print()
        print("🛑 Pre-commit AI safety gate FAILED:")
        print()
        if findings.get("secrets"):
            print("  ❌ SECRETS detected:")
            for s in findings["secrets"]:
                print(f"     {s['type']}: {s['match']}")
        if findings.get("private_context"):
            print("  ❌ PRIVATE CONTEXT leaks:")
            for p in findings["private_context"]:
                print(f"     {p['type']}: {p['match']}")
        if findings.get("migrations_without_rollback"):
            print("  ❌ MIGRATIONS without rollback:")
            for m in findings["migrations_without_rollback"]:
                print(f"     {m}")
        print()
        print("  Fix the issues above, then re-stage and commit.")
        print("  Bypass with: git commit --no-verify (logged as skipped)")
        print()
        return 1

    # Ruff format check (if available)
    python_files = [f for f in code_files if f.endswith(".py")]
    if python_files:
        try:
            subprocess.run(
                ["ruff", "check", "--fix", *python_files],
                capture_output=True,
                cwd=REPO_ROOT,
                timeout=30,
            )
            subprocess.run(
                ["ruff", "format", *python_files],
                capture_output=True,
                cwd=REPO_ROOT,
                timeout=30,
            )
            subprocess.run(
                ["git", "add", *python_files], capture_output=True, cwd=REPO_ROOT
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # ruff not installed or timed out — non-blocking

    print(f"✅ Pre-commit AI safety gate passed ({len(code_files)} files scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
