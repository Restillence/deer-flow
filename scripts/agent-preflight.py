#!/usr/bin/env python3
"""
Preflight check — run at the START of every AI coding session.

Verifies the AI engineering stack is loaded and ready before any code work.
Agents should run this before writing code. It does NOT modify anything —
pure read-only checks.

Usage:
    python3 scripts/agent-preflight.sh          # full check
    python3 scripts/agent-preflight.sh --json   # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_file(rel_path: str, label: str, required: bool = True) -> dict:
    path = REPO_ROOT / rel_path
    exists = path.exists()
    return {
        "item": label,
        "path": rel_path,
        "exists": exists,
        "required": required,
        "status": "pass" if exists else ("fail" if required else "warn"),
    }


def main() -> int:
    checks = [
        # Core stack files
        check_file("ai/router.yml", "Task router"),
        check_file("ai/contracts/code_change.yml", "Code change contract"),
        check_file("ai/tool-registry.yml", "Tool registry"),
        check_file("AGENTS.md", "Agent instructions"),
        # Evals
        check_file("ai/evals/test_safety_gate.py", "Safety gate eval"),
        check_file("ai/evals/test_fake_certainty.py", "Fake certainty eval"),
        # Scripts
        check_file("scripts/eval-log.py", "Eval logger"),
        check_file("scripts/audit-log.py", "Audit logger"),
        check_file("scripts/check-ai-stack.py", "Stack structure checker"),
        # PR template
        check_file(".github/pull_request_template.md", "PR template"),
    ]

    # Also check if .env exists (not tracked, but needed for runtime)
    checks.append(check_file(".env", "Environment config", required=False))

    # Check git hooks are installed
    hook = REPO_ROOT / ".git" / "hooks" / "pre-commit"
    checks.append(
        {
            "item": "Pre-commit hook",
            "path": ".git/hooks/pre-commit",
            "exists": hook.exists(),
            "required": False,
            "status": "pass" if hook.exists() else "warn",
        }
    )

    required_failures = [c for c in checks if c["status"] == "fail"]
    warnings = [c for c in checks if c["status"] == "warn"]

    if args.json:
        print(
            json.dumps(
                {
                    "all_passed": len(required_failures) == 0,
                    "failures": required_failures,
                    "warnings": warnings,
                    "total_checks": len(checks),
                },
                indent=2,
            )
        )
    else:
        print()
        print("=" * 55)
        print("  🤖 AI Stack Preflight Check")
        print("=" * 55)
        print()
        for c in checks:
            icon = {"pass": "✅", "fail": "❌", "warn": "⚠️ "}[c["status"]]
            req = "" if c["required"] else " (optional)"
            print(f"  {icon} {c['item']:<30s} {c['path']}{req}")
        print()
        if required_failures:
            print(
                f"  ❌ {len(required_failures)} REQUIRED item(s) missing — fix before coding."
            )
            print("     The AI stack is incomplete. Read AGENTS.md for setup.")
        elif warnings:
            print(f"  ⚠️  {len(warnings)} optional item(s) missing — non-blocking.")
            print("     All required stack components present. ✅")
        else:
            print("  ✅ All checks passed. Stack is ready.")
        print()
        print("=" * 55)

    return 1 if required_failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI session preflight check")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    sys.exit(main())
