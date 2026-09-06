from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".agents/skills/auto-improvement-pipeline/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import loop_constraints as constraints  # noqa: E402
from implementation_verifier import IndependentImprovementVerifier  # noqa: E402


def test_denylist_and_human_gate_share_one_machine_policy():
    denied = constraints.evaluate_execution_constraints(
        {"title": "設定更新"}, ["data/private.json"]
    )
    secret = constraints.evaluate_execution_constraints(
        {"title": "設定更新"}, [".streamlit/secrets.toml"]
    )
    gated = constraints.evaluate_execution_constraints(
        {"title": "ログイン文言を修正"}, ["auth/login.py"]
    )
    safe = constraints.evaluate_execution_constraints(
        {"title": "FAQ文言を修正"}, ["frontend/src/app/faq/page.tsx"]
    )
    assert denied["decision"] == "denylist" and denied["allowed"] is False
    assert secret["decision"] == "denylist" and secret["allowed"] is False
    assert gated["decision"] == "human_gate" and gated["allowed"] is False
    assert safe["decision"] == "auto" and safe["allowed"] is True


def test_attempt_ledger_blocks_fourth_attempt(tmp_path):
    ledger = constraints.AttemptLedger(tmp_path)
    assert ledger.begin("REV-1")["count"] == 1
    assert ledger.begin("REV-1")["count"] == 2
    assert ledger.begin("REV-1")["count"] == 3
    blocked = ledger.begin("REV-1")
    assert blocked == {"allowed": False, "count": 3, "max_attempts": 3}


def test_roles_are_distinct_and_verifier_rejects_destructive_candidate():
    verifier = IndependentImprovementVerifier(ROOT)
    roles = constraints.load_loop_constraints()["roles"]
    assert roles["implementer"] != roles["verifier"] == verifier.role
    passed, reason = verifier.sanity_check(
        "def a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n",
        "def a():\n    pass\n",
    )
    assert passed is False
    assert "行数激減" in reason or "関数消失" in reason


def test_verifier_uses_isolated_worktree_and_preserves_maker_file():
    verifier = IndependentImprovementVerifier(ROOT)
    target = SCRIPTS / "loop_constraints.py"
    original = target.read_text(encoding="utf-8")
    result = verifier.verify(target, original, original + "\n", allow_syntax_only=True)
    assert result.passed is True, result.summary
    assert result.isolation == "git-worktree"
    assert target.read_text(encoding="utf-8") == original


def test_audit_reports_l2_without_blocking_findings():
    script = ROOT / "scripts/audit_auto_improvement_loop.py"
    spec = importlib.util.spec_from_file_location("loop_audit", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit(ROOT)
    assert report["readiness"] == "L2-assisted"
    assert report["blocking_findings"] == []
    assert report["score"] >= 90


def test_both_implementation_paths_require_verifier_before_apply():
    step3 = (SCRIPTS / "step3_auto_apply.py").read_text(encoding="utf-8")
    agent = (SCRIPTS / "claude_agent_runner.py").read_text(encoding="utf-8")
    assert step3.index("self.verifier.verify(") < step3.index("self._pending_patches.append(")
    assert agent.index("IndependentImprovementVerifier(workspace).verify(") < agent.index(
        "target_file.write_text(new_code"
    )
