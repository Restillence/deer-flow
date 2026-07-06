#!/usr/bin/env python3
"""
Validate AI engineering stack structure for content-ecosystem.

Checks:
1. PR template headings
2. pr_agent.yml synchronize trigger
3. router.yml knowledge retrieval mandate (all four task classes)
4. tool-registry.yml entries (repo_vector_db, knowledge_graph, no stale wording,
   and all full-loop gstack skills: spec/plan/review/QA/health/ship/canary/docs)
5. knowledge files exist (vector-store.yml, graph.yml)
6. router.yml references full-loop gstack capabilities (spec/plan-eng/health/ship)

No external dependencies (stdlib only).

Usage:
    python scripts/check-ai-stack.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PR_HEADINGS = [
    "## 🤖 AI Assistance Disclosure",
    "## 📋 Files Changed",
    "## 🧪 Tests",
    "## ✅ CI",
    "## ⚠️ Risk Assessment",
    "## 🔄 Rollback Plan",
    "## 🔍 Review",
]

TASK_CLASSES = ["code_change", "dependency_bump", "ui_change", "docs_or_blog"]

EXPECTED_TOOL_ORDER: dict[str, list[str]] = {
    # Keep this aligned with the Saeloun-style loop: knowledge first, then
    # Claude Code for long-context repo work, Codex for tight execution/review,
    # gstack for repeatable workflow gates, and OpenCode as a fallback lane.
    "code_change": ["openclaw_rag", "claude_code", "codex"],
    "dependency_bump": ["openclaw_rag", "claude_code", "codex"],
    "ui_change": ["openclaw_rag", "claude_code", "codex"],
    "docs_or_blog": ["openclaw_rag", "web_search", "web_fetch", "claude_code", "codex"],
}

# Full-loop gstack skills that MUST be registered as tools.
REQUIRED_GSTACK_TOOLS = [
    "gstack_review",
    "gstack_qa",
    "gstack_ship",
    "gstack_spec",
    "gstack_plan_eng_review",
    "gstack_plan_ceo_review",
    "gstack_investigate",
    "gstack_health",
    "gstack_land_and_deploy",
    "gstack_canary",
    "gstack_document_release",
    "gstack_autoplan",
]

# Full-loop gstack capabilities that MUST be referenced by the router.
REQUIRED_GSTACK_ROUTER_STRINGS = [
    "gstack_spec",
    "gstack_plan_eng_review",
    "gstack_health",
    "gstack_ship",
]

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


def check_pr_template() -> None:
    print("\n=== 1. PR Template Headings ===")
    path = ".github/pull_request_template.md"
    if not exists(path):
        check(False, f"{path} exists")
        return
    text = read(path)
    for heading in REQUIRED_PR_HEADINGS:
        check(heading in text, f"PR template contains '{heading}'")


def check_pr_agent() -> None:
    print("\n=== 2. PR-Agent synchronize trigger ===")
    path = ".github/workflows/pr_agent.yml"
    if not exists(path):
        check(False, f"{path} exists")
        return
    text = read(path)
    pr_section = text.split("jobs:")[0] if "jobs:" in text else text
    check(
        "synchronize" in pr_section,
        "synchronize in pull_request trigger types",
    )
    check(
        '"synchronize"' in text,
        "synchronize in github_action_config.pr_actions",
    )


def _task_class_section(text: str, tc: str) -> str:
    """Extract the text block for a single task class."""
    marker = f"  {tc}:"
    idx = text.find(marker)
    if idx == -1:
        return ""
    # Section ends at the next task class marker or the routing section
    candidates: list[int] = []
    for n in TASK_CLASSES:
        if n == tc:
            continue
        ni = text.find(f"  {n}:", idx + len(marker))
        if ni != -1:
            candidates.append(ni)
    ri = text.find("routing:", idx + len(marker))
    if ri != -1:
        candidates.append(ri)
    ci = text.find("# ───", idx + len(marker))
    if ci != -1:
        candidates.append(ci)
    end = min(candidates) if candidates else len(text)
    return text[idx:end]


def _tools_in_section(section: str) -> list[str]:
    tools: list[str] = []
    in_tools = False
    for line in section.splitlines():
        stripped = line.strip()
        if stripped == "tools:":
            in_tools = True
            continue
        if in_tools and stripped and not stripped.startswith("-"):
            break
        if in_tools and stripped.startswith("-"):
            tools.append(stripped[1:].split("#", 1)[0].strip())
    return tools


def check_router() -> None:
    print("\n=== 3. Router knowledge retrieval mandate ===")
    path = "ai/router.yml"
    if not exists(path):
        check(False, f"{path} exists")
        return
    text = read(path)
    for tc in TASK_CLASSES:
        section = _task_class_section(text, tc)
        check(section != "", f"router.yml has task class '{tc}'")
        if not section:
            continue
        check("openclaw_rag" in section, f"{tc}: openclaw_rag in tools")
        check(
            "ai/knowledge/vector-store.yml" in section,
            f"{tc}: ai/knowledge/vector-store.yml in context",
        )
        check(
            "ai/knowledge/graph.yml" in section,
            f"{tc}: ai/knowledge/graph.yml in context",
        )
        check(
            "memory_search" in section,
            f"{tc}: knowledge context exit criterion (memory_search/gbrain)",
        )
        tools = _tools_in_section(section)
        expected_prefix = EXPECTED_TOOL_ORDER[tc]
        check(
            tools[: len(expected_prefix)] == expected_prefix,
            f"{tc}: tool order starts with {expected_prefix}",
        )
        check(tools[-1:] == ["opencode"], f"{tc}: opencode is final fallback tool")
        first_gstack_index = next(
            (i for i, tool in enumerate(tools) if tool.startswith("gstack_")), None
        )
        opencode_index = tools.index("opencode") if "opencode" in tools else -1
        check(
            first_gstack_index is not None and first_gstack_index < opencode_index,
            f"{tc}: gstack gates run before opencode fallback",
        )

    # Full-loop gstack capabilities must be referenced somewhere in the router.
    for s in REQUIRED_GSTACK_ROUTER_STRINGS:
        check(s in text, f"router.yml references '{s}'")


def check_tool_registry() -> None:
    print("\n=== 4. Tool Registry ===")
    path = "ai/tool-registry.yml"
    if not exists(path):
        check(False, f"{path} exists")
        return
    text = read(path)
    check("repo_vector_db" in text, "tool-registry.yml contains repo_vector_db")
    check("knowledge_graph" in text, "tool-registry.yml contains knowledge_graph")
    check(
        "Uses GLM-4.5-Flash (free)" not in text,
        "no stale 'Uses GLM-4.5-Flash (free)' wording",
    )
    # Full-loop gstack skills must all be registered as present tools.
    for name in REQUIRED_GSTACK_TOOLS:
        check(
            f"  {name}:" in text,
            f"tool-registry.yml contains gstack entry '{name}'",
        )


def check_knowledge_files() -> None:
    print("\n=== 5. Knowledge Files ===")
    check(
        exists("ai/knowledge/vector-store.yml"), "ai/knowledge/vector-store.yml exists"
    )
    check(exists("ai/knowledge/graph.yml"), "ai/knowledge/graph.yml exists")


def main() -> None:
    print("AI Stack Structure Check")
    print(f"Repo: {REPO_ROOT}")

    check_pr_template()
    check_pr_agent()
    check_router()
    check_tool_registry()
    check_knowledge_files()

    print("\n" + "=" * 50)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED")
        for f in FAILURES:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("RESULT: All checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
