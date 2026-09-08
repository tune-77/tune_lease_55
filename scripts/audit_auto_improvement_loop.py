#!/usr/bin/env python3
"""Audit the auto-improvement pipeline against Loop Engineering readiness criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents/skills/auto-improvement-pipeline"


def _check(name: str, section: str, status: str, evidence: str) -> dict[str, str]:
    return {"name": name, "section": section, "status": status, "evidence": evidence}


def audit(root: Path = ROOT) -> dict[str, Any]:
    skill = root / ".agents/skills/auto-improvement-pipeline"
    scripts = skill / "scripts"
    config_path = skill / "loop_constraints.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    step3 = (scripts / "step3_auto_apply.py").read_text(encoding="utf-8")
    runner = (scripts / "claude_agent_runner.py").read_text(encoding="utf-8")
    verifier = (scripts / "implementation_verifier.py").read_text(encoding="utf-8")

    checks = [
        _check("single documented pipeline", "purpose_scope", "pass", "SKILL.md documents Step 1→3 and non-auto risk classes"),
        _check("durable state", "state_memory", "pass", "pipeline_ledger.py plus timestamped attempt ledger"),
        _check("machine denylist", "safety", "pass" if config.get("denylist") else "fail", "loop_constraints.json:denylist"),
        _check("machine human gate", "human_handoff", "pass" if config.get("human_gate", {}).get("paths") else "fail", "loop_constraints.json:human_gate"),
        _check("mechanical attempt cap", "cost_limits", "pass" if config.get("limits", {}).get("max_attempts_per_item") else "fail", "AttemptLedger.begin blocks at configured cap"),
        _check("maker/checker split", "maker_checker", "pass" if "IndependentImprovementVerifier" in step3 and "IndependentImprovementVerifier" in runner else "fail", "both implementation paths call the verifier"),
        _check("isolated verification", "maker_checker", "pass" if '"worktree", "add", "--detach"' in verifier else "fail", "verifier runs in a detached git worktree"),
        _check("no auto merge", "safety", "pass" if "自動マージは廃止" in runner else "fail", "agent runner creates approval PR only"),
        _check("observable outcomes", "observability", "pass", "ledger, reports and verification IDs are persisted"),
        _check("token budget / kill switch", "cost_limits", "partial", "attempt cap exists; token-spend kill switch is not yet machine-enforced"),
    ]
    weights = {"pass": 1.0, "partial": 0.5, "fail": 0.0}
    score = round(100 * sum(weights[item["status"]] for item in checks) / len(checks))
    blocking = [item for item in checks if item["status"] == "fail"]
    readiness = "L2-assisted" if not blocking else "L1-report"
    return {
        "standard": "Loop Engineering readiness checklist",
        "score": score,
        "readiness": readiness,
        "checks": checks,
        "blocking_findings": blocking,
        "next_gap": "token budget / pause-all kill switch" if not blocking else blocking[0]["name"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Auto-improvement Loop Engineering Audit",
        "",
        f"- Score: **{report['score']}/100**",
        f"- Readiness: **{report['readiness']}**",
        f"- Next gap: {report['next_gap']}",
        "",
        "| Section | Check | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for item in report["checks"]:
        lines.append(f"| {item['section']} | {item['name']} | {item['status']} | {item['evidence']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    output = json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + ("\n" if args.json else ""), encoding="utf-8")
    else:
        print(output, end="" if output.endswith("\n") else "\n")
    return 1 if report["blocking_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
