"""チャット由来 quick_fix 起票（propose_quick_fix）→ パイプライン取り込みの配線テスト。

紫苑がチャットで受けた小規模修正要望を propose_quick_fix で判定・起票し、
recursive_self_improvement.load_chat_quick_fix_intake が候補源へ取り込んで
ranked_queue に載る（＝自動修正が実行可能になる）ことを検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_propose_quick_fix_accepts_and_writes_intake(tmp_path, monkeypatch):
    import lease_intelligence_tools as tools

    intake = tmp_path / "chat_quick_fix_intake.jsonl"
    monkeypatch.setattr(
        tools, "get_data_path",
        lambda name: str(intake) if "chat_quick_fix_intake" in name else name,
    )

    result = tools.propose_quick_fix("FAQページのボタン文言のタイポを直して")

    assert result["accepted"] is True
    assert result["target_module"] == "frontend/src/app/faq/page.tsx"
    assert intake.exists()
    record = json.loads(intake.read_text(encoding="utf-8").splitlines()[0])
    assert record["target_module"] == "frontend/src/app/faq/page.tsx"
    assert record["category"] == "quick_ui"
    assert record["source"] == "chat"


def test_propose_quick_fix_rejects_risky_and_abstract(tmp_path, monkeypatch):
    import lease_intelligence_tools as tools

    intake = tmp_path / "chat_quick_fix_intake.jsonl"
    monkeypatch.setattr(
        tools, "get_data_path",
        lambda name: str(intake) if "chat_quick_fix_intake" in name else name,
    )

    risky = tools.propose_quick_fix("スコアリングの承認閾値を70に下げて")
    assert risky["accepted"] is False

    abstract = tools.propose_quick_fix("紫苑の記憶システムを根本的に作り直して")
    assert abstract["accepted"] is False

    # 却下時は起票されない
    assert not intake.exists()


def test_execute_tool_dispatches_propose_quick_fix(tmp_path, monkeypatch):
    import lease_intelligence_tools as tools

    intake = tmp_path / "chat_quick_fix_intake.jsonl"
    monkeypatch.setattr(
        tools, "get_data_path",
        lambda name: str(intake) if "chat_quick_fix_intake" in name else name,
    )
    result = tools.execute_tool("propose_quick_fix", {"request": "ホーム画面のボタンの表示名のタイポを直して"})
    assert result["accepted"] is True


def test_tool_declarations_include_propose_quick_fix():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    assert "propose_quick_fix" in {item["name"] for item in TOOL_DECLARATIONS}


def test_intake_feeds_recursive_ranker(tmp_path, monkeypatch):
    from scripts import recursive_self_improvement as rsi
    import pipeline_ledger

    monkeypatch.setattr(pipeline_ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")

    intake = tmp_path / "chat_quick_fix_intake.jsonl"
    intake.write_text(
        json.dumps({
            "id": "chat_abc",
            "title": "FAQページのボタン文言のタイポを直して",
            "description": "表示名の誤字を直す",
            "target_module": "frontend/src/app/faq/page.tsx",
        }) + "\n",
        encoding="utf-8",
    )

    items = rsi.load_chat_quick_fix_intake(intake)
    assert len(items) == 1

    report = {"needs_review": items, "applied": []}
    bundle = rsi.build_recursive_self_improvement(report, workspace_root=_REPO_ROOT)

    assert bundle["ranked_queue_count"] == 1
    assert bundle["ranked_queue"][0]["target_module"] == "frontend/src/app/faq/page.tsx"


def test_load_intake_preserves_original_source(tmp_path):
    """shion_promise 等の元 source を保持し、改善ログUIで出所を辿れるようにする。"""
    from scripts import recursive_self_improvement as rsi

    intake = tmp_path / "chat_quick_fix_intake.jsonl"
    intake.write_text(
        json.dumps({"id": "promise_x", "title": "残価の根拠を調べる", "source": "shion_promise"})
        + "\n"
        + json.dumps({"id": "chat_y", "title": "タイポ修正"})  # source 未指定
        + "\n",
        encoding="utf-8",
    )

    items = {i["id"]: i for i in rsi.load_chat_quick_fix_intake(intake)}
    assert items["promise_x"]["source"] == "shion_promise"
    assert items["chat_y"]["source"] == "chat_quick_fix"  # 未指定は従来どおり
