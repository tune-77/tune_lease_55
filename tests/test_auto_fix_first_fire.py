"""自動修正パイプラインの「初発火」に関する回帰テスト。

背景: auto_fix_policy が禁止/許可キーワードを走査するテキストに識別ハッシュ
(canonical_key, 例: misc_a92f18c9bdb3) を含めていたため、ハッシュ中の偶然の
部分文字列（"db" 等）に禁止語が誤マッチし、正当な quick_ui 候補まで DENY され、
ランクキューが常に空 → 自動修正が一度も発火しなかった。
本テストはハッシュ誤マッチを防ぎつつ、本物の禁止語は従来どおり弾くことを保証する。
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_auto_fix_policy_does_not_scan_canonical_key_hash():
    from scripts import recursive_self_improvement as _rsi  # noqa: F401 (パス設定の副作用)
    import auto_fix_policy as policy

    candidate = {
        "title": "ホーム画面のラベル文言を修正",
        "description": "表示名の誤字を直す",
        "target_module": "frontend/src/app/home/page.tsx",
        "canonical_key": "misc_a92f18c9bdb3",  # ハッシュ中に 'db' を含む
    }
    result = policy.evaluate_auto_fix_policy(candidate, _REPO_ROOT)
    assert result["auto_fix_allowed"] is True, result
    # キーワード走査テキストに識別ハッシュが混入していないこと
    assert "misc_" not in policy._text(candidate)
    assert "misc_" not in policy._body_text(candidate)


def test_auto_fix_policy_still_denies_real_db_keyword():
    from scripts import recursive_self_improvement as _rsi  # noqa: F401
    import auto_fix_policy as policy

    candidate = {
        "title": "DB接続エラーの修正",
        "description": "sqlite の db 接続でエラーが出る",
        "target_module": "api/main.py",
    }
    result = policy.evaluate_auto_fix_policy(candidate, _REPO_ROOT)
    assert result["auto_fix_allowed"] is False


def test_target_inference_resolves_expanded_pages():
    from scripts import recursive_self_improvement as _rsi  # noqa: F401
    import auto_fix_policy as policy

    cases = {
        "FAQページの文言のタイポを修正": "frontend/src/app/faq/page.tsx",
        "案件一覧のラベル表示名を修正": "frontend/src/app/cases/page.tsx",
        "改善ログ画面の説明文の誤字を直す": "frontend/src/app/improvement-log/page.tsx",
    }
    for text, expected in cases.items():
        item = {"title": text, "description": text}
        assert policy.infer_target_module(item, _REPO_ROOT) == expected, text


def test_classify_quick_fix_accepts_genuine_quick_fix():
    from scripts import recursive_self_improvement as _rsi  # noqa: F401
    import auto_fix_policy as policy

    verdict = policy.classify_quick_fix(
        {"title": "FAQページのボタン文言のタイポを修正", "description": "表示名の誤字を直す"},
        _REPO_ROOT,
    )
    assert verdict["is_quick_fix"] is True
    assert verdict["target_module"] == "frontend/src/app/faq/page.tsx"
    assert verdict["risk"] == "low"
    # 返却された candidate はそのまま発火可能な形になっている
    assert verdict["candidate"]["implementation"]["category"] == "quick_ui"
    assert policy.evaluate_auto_fix_policy(verdict["candidate"], _REPO_ROOT)["auto_fix_allowed"] is True


def test_classify_quick_fix_rejects_abstract_and_risky():
    from scripts import recursive_self_improvement as _rsi  # noqa: F401
    import auto_fix_policy as policy

    # 対象ファイル不明の抽象要望
    abstract = policy.classify_quick_fix(
        {"title": "紫苑の記憶参照システムに根本的欠陥がある", "description": "改善してほしい"},
        _REPO_ROOT,
    )
    assert abstract["is_quick_fix"] is False
    assert abstract["candidate"] is None

    # スコアリング等のリスク領域
    risky = policy.classify_quick_fix(
        {"title": "スコアリングの閾値を変更", "description": "承認ラインを調整", "target_module": "scoring_core.py"},
        _REPO_ROOT,
    )
    assert risky["is_quick_fix"] is False


def test_quick_ui_candidate_reaches_ranked_queue(tmp_path, monkeypatch):
    """空の台帳のもと、quick_ui 候補が suppressed されず ranked_queue に載る（＝発火可能）。"""
    from scripts import recursive_self_improvement as rsi
    import pipeline_ledger

    monkeypatch.setattr(pipeline_ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")

    report = {
        "needs_review": [
            {
                "id": "REV-QUICK",
                "title": "ホーム画面のボタン文言のタイポ修正",
                "description": "ホーム画面のボタン文言に誤字があるため表示名を修正する",
                "target_module": "frontend/src/app/home/page.tsx",
            }
        ],
        "applied": [],
    }
    bundle = rsi.build_recursive_self_improvement(report, workspace_root=_REPO_ROOT)

    assert bundle["ranked_queue_count"] == 1
    assert bundle["suppressed_count"] == 0
    queued = bundle["ranked_queue"][0]
    assert queued["auto_fix_policy"]["auto_fix_allowed"] is True
    assert queued["auto_fix_policy"]["risk"] == "low"


def test_inferred_target_is_attached_to_ranked_queue(tmp_path, monkeypatch):
    """対象ファイル未指定の自然文候補でも、推定された対象ファイルと quick_ui 分類が
    ranked_queue エントリに載る（classify_quick_fix の配線）。"""
    from scripts import recursive_self_improvement as rsi
    import pipeline_ledger

    monkeypatch.setattr(pipeline_ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")

    report = {
        "needs_review": [
            {
                "id": "REV-INFER",
                "title": "FAQページのボタン文言のタイポを修正",
                "description": "表示名の誤字を直す",
                # target_module は与えない（推定に依存）
            }
        ],
        "applied": [],
    }
    bundle = rsi.build_recursive_self_improvement(report, workspace_root=_REPO_ROOT)

    assert bundle["ranked_queue_count"] == 1
    queued = bundle["ranked_queue"][0]
    assert queued["target_module"] == "frontend/src/app/faq/page.tsx"
    assert queued["category"] == "quick_ui"
    assert queued["auto_fix_policy"]["auto_fix_allowed"] is True
