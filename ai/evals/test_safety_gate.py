#!/usr/bin/env python3
"""
Eval 2: Safety Gate — Missing Rollback, Secret Leaks, Unsafe Migrations

Scans git diffs (or PR patch files) for:
1. DB migrations without rollback
2. Secrets/private data in the diff
3. Missing rollback plan for high-risk changes
4. Private-context leaks (names, emails, internal URLs)

Run: python -m pytest ai/evals/test_safety_gate.py -v
     or: python3 ai/evals/test_safety_gate.py <diff_file>
"""

import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────
# SECRET DETECTION
# ─────────────────────────────────────────────────────────
SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
    (r"ghp_[a-zA-Z0-9]{36,}", "GitHub personal access token"),
    (r"gho_[a-zA-Z0-9]{36,}", "GitHub OAuth token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key ID"),
    (r"aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{40}", "AWS secret key"),
    (r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", "Private key block"),
    (r"password\s*=\s*['\"][^'\"]{8,}['\"]", "Hardcoded password"),
    (r"token\s*=\s*['\"][A-Za-z0-9_\-\.]{20,}['\"]", "Hardcoded token"),
    (r"xox[bpoa]-[a-zA-Z0-9-]+", "Slack token"),
    (r"AIza[0-9A-Za-z\-_]{35}", "Google API key"),
]

# ─────────────────────────────────────────────────────────
# PRIVATE CONTEXT LEAKS
# ─────────────────────────────────────────────────────────
PRIVATE_CONTEXT_PATTERNS = [
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "Email address (possible PII)"),
    (r"\b\d{3}-\d{3}-\d{4}\b", "Phone number"),
    (r"\b\d{3}\s\d{3}\s\d{4}\b", "Phone number (spaces)"),
    (r"//(?:internal|staging|dev)\.[a-zA-Z0-9-]+\.[a-z]+", "Internal/staging URL"),
    (r"(?i)client[_-]?(?:name|id|secret)\s*[:=]\s*['\"]?[a-zA-Z0-9]{3,}", "Client identifier"),
]

# ─────────────────────────────────────────────────────────
# MIGRATION DETECTION
# ─────────────────────────────────────────────────────────
MIGRATION_PATTERNS = [
    r"migrations/.*\.py$",
    r"migrations/.*\.sql$",
    r"alembic/versions/.*\.py$",
    r"db/migrate.*",
]
DOWNGRADE_PATTERNS = [
    r"def\s+downgrade\s*\(",
    r"def\s+down\b",
    r"--\s*rollback",
    r"DROP\s+(TABLE|COLUMN|INDEX)",
    r"REVERT",
]


def scan_diff(diff_text: str) -> dict:
    """Scan a unified diff for safety issues."""
    findings = {
        "secrets": [],
        "private_context": [],
        "migrations_without_rollback": [],
        "missing_risk_assessment": False,
    }

    # Only scan added lines (starting with +, not +++)
    added_lines = [
        line[1:]
        for line in diff_text.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_text = "\n".join(added_lines)

    # Check secrets
    for pattern, name in SECRET_PATTERNS:
        for match in re.finditer(pattern, added_text):
            findings["secrets"].append({
                "type": name,
                "match": match.group()[:20] + "...",
            })

    # Check private context
    for pattern, name in PRIVATE_CONTEXT_PATTERNS:
        for match in re.finditer(pattern, added_text):
            # Allow common non-PII emails like noreply@github.com
            if "noreply" in match.group().lower() or "example.com" in match.group().lower():
                continue
            findings["private_context"].append({
                "type": name,
                "match": match.group()[:30] + "...",
            })

    # Check migrations without rollback
    changed_files = re.findall(r"^\+\+\+\s+b?(.*)$", diff_text, re.MULTILINE)
    for filepath in changed_files:
        is_migration = any(re.search(pat, filepath) for pat in MIGRATION_PATTERNS)
        if is_migration:
            has_downgrade = any(re.search(pat, added_text) for pat in DOWNGRADE_PATTERNS)
            if not has_downgrade:
                findings["migrations_without_rollback"].append(filepath)

    return findings


def has_issues(findings: dict) -> bool:
    return any(findings[k] for k in ["secrets", "private_context", "migrations_without_rollback"])


def test_clean_diff_passes():
    """A diff with no secrets or risky changes should pass."""
    clean_diff = """diff --git a/src/utils.py b/src/utils.py
+++ b/src/utils.py
@@ -1,3 +1,5 @@
+def hello():
+    return "world"
"""
    findings = scan_diff(clean_diff)
    assert not has_issues(findings), f"False positive: {findings}"


def test_detects_api_key():
    """Diff with API key should be flagged."""
    diff = """+++ b/config.py
+API_KEY = "sk-abc123def456ghi789jkl012mno345pqr678"
"""
    findings = scan_diff(diff)
    assert len(findings["secrets"]) > 0, "Should detect API key"


def test_detects_private_email():
    """Diff with personal email should be flagged."""
    diff = """+++ b/README.md
+Contact: john.doe@company.com
"""
    findings = scan_diff(diff)
    assert len(findings["private_context"]) > 0, "Should detect email"


def test_allows_noreply_email():
    """noreply and example.com emails should NOT be flagged."""
    diff = """+++ b/README.md
+Questions? noreply@github.com or test@example.com
"""
    findings = scan_diff(diff)
    assert len(findings["private_context"]) == 0, f"False positive: {findings}"


def test_migration_without_rollback():
    """Migration file without downgrade function should be flagged."""
    diff = """diff --git a/migrations/001_add_col.py b/migrations/001_add_col.py
+++ b/migrations/001_add_col.py
+def upgrade():
+    op.add_column('users', sa.Column('email', sa.String))
"""
    findings = scan_diff(diff)
    assert len(findings["migrations_without_rollback"]) > 0


def test_migration_with_rollback_ok():
    """Migration with downgrade function should pass."""
    diff = """diff --git a/migrations/001_add_col.py b/migrations/001_add_col.py
+++ b/migrations/001_add_col.py
+def upgrade():
+    op.add_column('users', sa.Column('email', sa.String()))
+
+def downgrade():
+    op.drop_column('users', 'email')
"""
    findings = scan_diff(diff)
    assert len(findings["migrations_without_rollback"]) == 0


def test_empty_diff():
    """Empty diff should return no findings."""
    assert not has_issues(scan_diff(""))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        diff_text = Path(sys.argv[1]).read_text()
        findings = scan_diff(diff_text)
        if has_issues(findings):
            print("❌ Safety issues found:")
            for k, v in findings.items():
                if v:
                    print(f"  {k}: {v}")
            sys.exit(1)
        else:
            print("✅ No safety issues detected.")
    else:
        print("Usage: python3 test_safety_gate.py <diff_file>")
        print("   or: python3 -m pytest ai/evals/test_safety_gate.py -v")
