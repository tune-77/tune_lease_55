"""デモ用 quick_ui 種のポリシー分類テスト（ループ実演の決定論化を保証）。

自己改善ループのライブ実演では、auto-fix キューに確実に1件乗る「quick_ui 種」が要る。
本テストは auto_fix_policy（LLM非依存の純ロジック）で、
  - FAQページの文言誤字の種 → auto_fix_allowed / quick_ui / 対象=faq/page.tsx
  - リース期間（審査ロジック）→ DENY（安全ガードレール）
を保証する。demo_chat_logs.json に種が入っていることも確認する。

docs/hackathon_pitch_plan.md の「② ループ実演設計」を支えるテスト。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_POLICY_DIR = _REPO / ".agents" / "skills" / "auto-improvement-pipeline" / "scripts"


@pytest.fixture(scope="module")
def policy():
    sys.path.insert(0, str(_POLICY_DIR))
    import auto_fix_policy  # noqa: E402

    return auto_fix_policy


def test_faq_typo_seed_classifies_as_quick_ui(policy):
    seed = {
        "title": "FAQページの文言の誤字",
        "description": "faqページのよくある質問の見出しに誤字があり、表示の文言を修正したい。",
        "reason": "ユーザーがFAQページの文言の誤字を指摘",
    }
    result = policy.evaluate_auto_fix_policy(seed, str(_REPO))
    assert result["auto_fix_allowed"] is True
    assert result["risk"] == "low"
    # 単一・安全な対象ファイルに推定される。
    assert result.get("inferred_target_module") == "frontend/src/app/faq/page.tsx"

    cls = policy.classify_quick_fix(seed, str(_REPO))
    assert cls["is_quick_fix"] is True
    assert cls["target_module"] == "frontend/src/app/faq/page.tsx"


def test_lease_term_candidate_is_denied(policy):
    # 審査ロジックに関わる候補は自動修正されない（安全ガードレール）。
    bad = {
        "title": "リース期間の初期値がおかしい",
        "description": "リース期間5年の標準がおかしいので直したい",
    }
    result = policy.evaluate_auto_fix_policy(bad, str(_REPO))
    assert result["auto_fix_allowed"] is False
    assert "リース期間" in result["reason"]


def test_scoring_candidate_is_denied(policy):
    bad = {"title": "スコアリング係数を調整したい", "description": "モデルの閾値を変える"}
    result = policy.evaluate_auto_fix_policy(bad, str(_REPO))
    assert result["auto_fix_allowed"] is False


def test_demo_chat_logs_contains_quick_ui_seed():
    path = _REPO / "scripts" / "demo" / "demo_chat_logs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = {x.get("id") for x in data if isinstance(x, dict)}
    assert "chat-006" in ids
    seed = next(x for x in data if x.get("id") == "chat-006")
    # 種の本文に対象推定キー（faq）と許可語（文言/誤字）が含まれる。
    blob = json.dumps(seed, ensure_ascii=False).lower()
    assert "faq" in blob
    assert "文言" in blob and "誤字" in blob
