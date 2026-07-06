#!/usr/bin/env python3
"""
Eval 1: Fake Certainty Detector

Scans AI-generated PR descriptions and diffs for phrases that assert
facts without evidence, source links, or test output.

Run: python -m pytest ai/evals/test_fake_certainty.py -v
     or: python3 ai/evals/test_fake_certainty.py <pr_body_file>
"""

import re
import sys
from pathlib import Path

# Patterns that indicate fake certainty — assertions without evidence
FAKE_CERTAINTY_PATTERNS = [
    # Asserting safety/security without evidence
    (r"(?i)\bthis\s+is\s+secure\b(?!.*test|.*audit|.*scan)", "Claims 'secure' without evidence"),
    (r"(?i)\bno\s+security\s+(issues|problems|concerns)\b(?!.*scan|.*audit|.*bandit)", "Claims no security issues without scan"),
    # Asserting correctness without tests
    (r"(?i)\btests?\s+pass\b(?!.*pytest|.*CI|.*✅|.*green|.*output)", "Claims 'tests pass' without CI/test output link"),
    (r"(?i)\bworks?\s+correctly\b(?!.*test|.*e2e|.*verified)", "Claims 'works correctly' without test evidence"),
    (r"(?i)\bthis\s+won'?t\s+(break|fail)\b", "Absolute claim 'won't break' — nothing is absolute"),
    # Asserting compatibility without checking
    (r"(?i)\bbackward[s]?\s+compatible\b(?!.*test|.*verified|.*checked)", "Claims backward compatible without verification"),
    (r"(?i)\bno\s+breaking\s+changes\b(?!.*changelog|.*test|.*verified)", "Claims no breaking changes without checking changelog"),
    # Vague reassurance
    (r"(?i)\bshould\s+(work|be\s+fine|be\s+safe)\b", "'Should work' — verify, don't guess"),
    (r"(?i)\btrivial\s+change\b(?!.*test)", "Calls change 'trivial' without test evidence"),
]

# Things that count as evidence
EVIDENCE_MARKERS = [
    "pytest", "CI", "✅", "green", "test output", "test result",
    "bandit", "ruff", "mypy", "coverage", "e2e", "playwright",
    "source:", "ref:", "http", "changelog", "migration test",
]


def scan_text(text: str) -> list[dict]:
    """Return list of fake-certainty findings."""
    findings = []
    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        for pattern, message in FAKE_CERTAINTY_PATTERNS:
            if re.search(pattern, line):
                # Check if there's evidence nearby (same line or within 2 lines)
                context_window = "\n".join(lines[max(0, i-2):i+3])
                has_evidence = any(marker in context_window.lower() for marker in EVIDENCE_MARKERS)
                if not has_evidence:
                    findings.append({
                        "line": i,
                        "text": line.strip()[:100],
                        "issue": message,
                    })
    return findings


def test_no_fake_certainty_in_clean_pr():
    """A well-evidenced PR description should pass."""
    clean_pr = """
    ## Files changed
    - src/auth.py: Fixed token validation (bug #123)

    ## Risk assessment
    Low — auth flow unchanged, only adds extra validation step.

    ## Tests
    pytest tests/test_auth.py -v → 12 passed
    CI: ✅ green (run #456)

    ## Rollback
    Revert commit abc123 — no DB changes.
    """
    findings = scan_text(clean_pr)
    assert findings == [], f"False positive: {findings}"


def test_detects_unsupported_claims():
    """Phrases without evidence should be flagged."""
    suspicious_pr = """
    This is secure. Tests pass. This won't break.
    Should work fine. It's a trivial change.
    """
    findings = scan_text(suspicious_pr)
    assert len(findings) >= 3, f"Expected ≥3 findings, got {len(findings)}: {findings}"


def test_detects_claims_with_nearby_evidence_pass():
    """Claims with evidence on nearby lines should NOT be flagged."""
    pr_with_evidence = """
    Tests pass — see pytest output below:
    ```
    12 passed in 2.3s
    ```
    CI is ✅ green.
    """
    findings = scan_text(pr_with_evidence)
    assert findings == [], f"False positive with evidence: {findings}"


def test_empty_input():
    """Empty text should return no findings."""
    assert scan_text("") == []


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text()
        findings = scan_text(text)
        if findings:
            print(f"❌ {len(findings)} fake-certainty issue(s) found:")
            for f in findings:
                print(f"  Line {f['line']}: {f['issue']}")
                print(f"    → {f['text']}")
            sys.exit(1)
        else:
            print("✅ No fake certainty detected.")
    else:
        print("Usage: python3 test_fake_certainty.py <pr_body_file>")
        print("   or: python3 -m pytest ai/evals/test_fake_certainty.py -v")
