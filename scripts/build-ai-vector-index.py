#!/usr/bin/env python3
"""
Build deterministic repo-local SQLite vector index for AI knowledge retrieval.

No external dependencies (stdlib only). No network. No gbrain required.
CI uses --check to enforce freshness.

Usage:
    python scripts/build-ai-vector-index.py          # build/update
    python scripts/build-ai-vector-index.py --check   # verify freshness (CI)
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / ".ai" / "vector"
DB_PATH = INDEX_DIR / "index.sqlite"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

DIMENSIONS = 256
TOKEN_RE = re.compile(r"[a-zA-Z0-9_./-]+")

# Source paths (mirrors ai/knowledge/vector-store.yml)
SOURCES = [
    "AGENTS.md",
    ".pr_agent.toml",
    ".github/workflows/ci.yml",
    ".github/workflows/pr_agent.yml",
    ".github/pull_request_template.md",
    "ai/router.yml",
    "ai/tool-registry.yml",
    "ai/contracts/code_change.yml",
    "ai/evals",
    "ai/knowledge",
    "docs/ai/wiki",
    "scripts/audit-log.py",
    "scripts/deploy-ai-stack.sh",
    "scripts/check-ai-stack.py",
    "scripts/build-ai-vector-index.py",
]

SCANNABLE_EXTS = {".md", ".yml", ".yaml", ".py", ".toml", ".sh"}


def tokenize(text: str) -> list[str]:
    """Tokenize lowercase words matching [a-zA-Z0-9_./-]+."""
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def embed(tokens: list[str]) -> list[float]:
    """Local hashing embeddings: hash tokens into DIMENSIONS dimensions, normalize counts."""
    vec: list[float] = [0.0] * DIMENSIONS
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % DIMENSIONS] += 1.0
    # L2 normalize
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_source_files() -> list[str]:
    """Expand SOURCES into a sorted list of relative file paths."""
    files: set[str] = set()
    for src in SOURCES:
        p = REPO_ROOT / src
        if p.is_file():
            files.add(src)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in SCANNABLE_EXTS:
                    files.add(str(f.relative_to(REPO_ROOT)))
    return sorted(files)


def build_index() -> dict:
    """Build the vector index from source files. Returns manifest dict."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    source_files = collect_source_files()
    documents = []
    for rel in source_files:
        path = REPO_ROOT / rel
        text = path.read_text(errors="replace")
        tokens = tokenize(text)
        vec = embed(tokens)
        documents.append(
            {
                "path": rel,
                "sha256": sha256_file(path),
                "tokens": len(tokens),
                "vector_json": json.dumps(vec),
            }
        )

    # Write SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            path text primary key,
            sha256 text not null,
            tokens integer not null,
            vector_json text not null
        );
        CREATE TABLE IF NOT EXISTS metadata (
            key text primary key,
            value text not null
        );
        """
    )
    conn.execute("DELETE FROM documents")
    for doc in documents:
        conn.execute(
            "INSERT INTO documents (path, sha256, tokens, vector_json) VALUES (?, ?, ?, ?)",
            (doc["path"], doc["sha256"], doc["tokens"], doc["vector_json"]),
        )
    generated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('dimensions', ?)",
        (str(DIMENSIONS),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('generated_at', ?)",
        (generated_at,),
    )
    conn.commit()
    conn.close()

    manifest = {
        "name": "content-ecosystem-ai-vector-index",
        "dimensions": DIMENSIONS,
        "generated_at": generated_at,
        "sources": [
            {
                "path": doc["path"],
                "sha256": doc["sha256"],
                "size": (REPO_ROOT / doc["path"]).stat().st_size,
            }
            for doc in documents
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def check_index() -> int:
    """Verify the existing index matches expected state. Returns exit code."""
    errors: list[str] = []

    if not DB_PATH.exists():
        errors.append(f"Vector DB missing: {DB_PATH.relative_to(REPO_ROOT)}")
    if not MANIFEST_PATH.exists():
        errors.append(f"Manifest missing: {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print("\nRun: python scripts/build-ai-vector-index.py")
        return 1

    expected_files = collect_source_files()
    expected_state: dict[str, str] = {}
    for rel in expected_files:
        path = REPO_ROOT / rel
        expected_state[rel] = sha256_file(path)

    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_map = {s["path"]: s["sha256"] for s in manifest.get("sources", [])}

    # Check manifest matches expected
    if set(manifest_map.keys()) != set(expected_state.keys()):
        missing = set(expected_state.keys()) - set(manifest_map.keys())
        stale = set(manifest_map.keys()) - set(expected_state.keys())
        if missing:
            errors.append(f"Manifest missing files: {sorted(missing)}")
        if stale:
            errors.append(f"Manifest has stale files: {sorted(stale)}")
    for rel, sha in expected_state.items():
        if manifest_map.get(rel) != sha:
            errors.append(f"Stale (manifest): {rel}")

    # Check DB matches expected
    conn = sqlite3.connect(DB_PATH)
    rows = {
        row[0]: row[1]
        for row in conn.execute("SELECT path, sha256 FROM documents").fetchall()
    }
    conn.close()

    if set(rows.keys()) != set(expected_state.keys()):
        missing = set(expected_state.keys()) - set(rows.keys())
        stale = set(rows.keys()) - set(expected_state.keys())
        if missing:
            errors.append(f"DB missing files: {sorted(missing)}")
        if stale:
            errors.append(f"DB has stale files: {sorted(stale)}")
    for rel, sha in expected_state.items():
        if rows.get(rel) != sha:
            errors.append(f"Stale (db): {rel}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print("\nRun: python scripts/build-ai-vector-index.py")
        return 1

    print(f"OK: vector index fresh ({len(expected_files)} documents)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/check repo-local AI vector index"
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify freshness without building"
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(check_index())

    manifest = build_index()
    print(f"Built vector index: {DB_PATH.relative_to(REPO_ROOT)}")
    print(f"  Documents: {len(manifest['sources'])}")
    print(f"  Dimensions: {manifest['dimensions']}")
    print(f"  Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
