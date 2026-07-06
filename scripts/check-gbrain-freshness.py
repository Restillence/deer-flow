#!/usr/bin/env python3
"""
CI-friendly gbrain / repo-local vector freshness gate.

Stdlib only. No network. Does NOT require live gbrain in CI by default,
because CI runners have no VM brain state. Use --strict-live to additionally
require gbrain on PATH (including ~/.bun/bin) and run `gbrain --version`.

Checks:
1. Repo-local vector index freshness via build-ai-vector-index.py --check
2. ai/tool-registry.yml contains gbrain, openclaw_rag, repo_vector_db
3. ai/router.yml contains openclaw_rag and memory_search/gbrain wording
4. (--strict-live only) gbrain present on PATH (including ~/.bun/bin) and `gbrain --version` runs

Usage:
    python scripts/check-gbrain-freshness.py
    python scripts/check-gbrain-freshness.py --strict-live
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FAILURES: list[str] = []


def check(cond: bool, message: str) -> None:
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {message}")
    if not cond:
        FAILURES.append(message)


def read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(errors="replace")


def exists(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


def check_vector_freshness() -> None:
    print("\n=== 1. Repo-local vector index freshness ===")
    builder = REPO_ROOT / "scripts" / "build-ai-vector-index.py"
    if not builder.exists():
        check(False, "scripts/build-ai-vector-index.py exists")
        return
    result = subprocess.run(
        [sys.executable, str(builder), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip())
    check(result.returncode == 0, "build-ai-vector-index.py --check passes")


def check_tool_registry() -> None:
    print("\n=== 2. Tool Registry knowledge tools ===")
    path = "ai/tool-registry.yml"
    if not exists(path):
        check(False, f"{path} exists")
        return
    text = read(path)
    for tool in ("gbrain", "openclaw_rag", "repo_vector_db"):
        check(f"{tool}:" in text, f"tool-registry.yml declares '{tool}'")


def check_router() -> None:
    print("\n=== 3. Router knowledge-retrieval wiring ===")
    path = "ai/router.yml"
    if not exists(path):
        check(False, f"{path} exists")
        return
    text = read(path)
    check("openclaw_rag" in text, "router.yml references openclaw_rag")
    check(
        "memory_search" in text or "gbrain" in text,
        "router.yml references memory_search / gbrain",
    )


def check_live_gbrain() -> None:
    print("\n=== 4. Live gbrain (--strict-live) ===")
    env = os.environ.copy()
    bun_bin = str(Path.home() / ".bun" / "bin")
    env["PATH"] = f"{bun_bin}:{env.get('PATH', '')}"
    gbrain = shutil.which("gbrain", path=env["PATH"])
    check(gbrain is not None, "gbrain found on PATH including ~/.bun/bin")
    if gbrain is None:
        return
    result = subprocess.run(
        [gbrain, "--version"],
        capture_output=True,
        text=True,
        env=env,
    )
    check(result.returncode == 0, "gbrain --version exits 0")
    if result.stdout.strip():
        print(f"  gbrain version: {result.stdout.strip().splitlines()[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI-friendly gbrain / repo-local vector freshness gate"
    )
    parser.add_argument(
        "--strict-live",
        action="store_true",
        help="Require live gbrain on PATH and run gbrain --version",
    )
    args = parser.parse_args()

    print("gbrain / Vector Freshness Gate")
    print(f"Repo: {REPO_ROOT}")
    print(f"Mode: {'strict-live' if args.strict_live else 'ci (live gbrain optional)'}")

    check_vector_freshness()
    check_tool_registry()
    check_router()
    if args.strict_live:
        check_live_gbrain()
    else:
        print("\n=== 4. Live gbrain check skipped (CI has no VM brain state) ===")

    print("\n" + "=" * 50)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED")
        for f in FAILURES:
            print(f"  x {f}")
        sys.exit(1)
    else:
        print("RESULT: All checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
