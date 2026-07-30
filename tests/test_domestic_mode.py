from __future__ import annotations

import json


def test_domestic_mode_empty_memo_uses_rule_fallback():
    import api.domestic_mode as domestic

    result = domestic.evaluate_domestic_mode_memo("")

    assert result["generated"] is False
    assert result["status"] == "hold"
    assert result["decision_layer"] == "文脈不足"
    assert "意図" in result["missing_inputs"]


def test_domestic_mode_normalizes_gemini_result(monkeypatch):
    import api.domestic_mode as domestic

    monkeypatch.setattr(
        domestic,
        "call_gemini_json",
        lambda *_args, **_kwargs: {
            "status": "adopt",
            "label": "採用候補",
            "reason": "意図、残す理由、成功指標が揃っているため採用できます。",
            "next_action": "小さく導線を追加し、30日後に利用率を見ます。",
            "decision_layer": "再配置候補",
            "missing_inputs": [],
            "success_metric": "対象画面への遷移率",
            "review_timing": "30日後",
        },
    )

    result = domestic.evaluate_domestic_mode_memo(
        "内政メモ:\n対象: FAQ\n意図: 必要な場面で見つけやすくする\n残す理由: 問い合わせを減らす\n成功指標: 遷移率"
    )

    assert result["generated"] is True
    assert result["model"] == "gemini"
    assert result["status"] == "adopt"
    assert result["decision_layer"] == "再配置候補"
    assert result["success_metric"] == "対象画面への遷移率"


def test_domestic_mode_invalid_status_becomes_hold():
    import api.domestic_mode as domestic

    result = domestic.normalize_domestic_mode_result(
        {
            "status": "execute",
            "reason": "実行はできません。",
            "next_action": "人間確認に戻します。",
            "missing_inputs": "bad",
        }
    )

    assert result["status"] == "hold"
    assert result["label"] == "保留"
    assert result["missing_inputs"] == []


def test_domestic_mode_saves_self_proposal_logs(tmp_path, monkeypatch):
    import api.domestic_mode as domestic

    proposals_path = tmp_path / "domestic_mode_proposals.jsonl"
    improvement_log_path = tmp_path / "cloudrun_improvement_log.jsonl"
    monkeypatch.setattr(domestic, "_PROPOSALS_PATH", proposals_path)
    monkeypatch.setattr(domestic, "_IMPROVEMENT_LOG_PATH", improvement_log_path)

    result = {
        "generated": True,
        "status": "observe",
        "label": "観測継続",
        "reason": "測定条件はあるが採用根拠がまだ薄い。",
        "next_action": "30日だけ導線ログを見ます。",
        "decision_layer": "観測継続",
        "missing_inputs": ["残す理由"],
        "success_metric": "問い合わせ件数",
        "review_timing": "30日後",
        "model": "gemini",
    }
    saved = domestic.save_domestic_mode_proposal(
        "内政メモ:\n対象: FAQ\n意図: 自己解決を増やす\n成功指標: 問い合わせ件数",
        {"label": "観測継続"},
        result,
    )

    proposal = json.loads(proposals_path.read_text(encoding="utf-8").splitlines()[0])
    improvement = json.loads(improvement_log_path.read_text(encoding="utf-8").splitlines()[0])

    assert saved["saved"] is True
    assert saved["title"] == proposal["title"]
    assert proposal["source"] == "domestic_mode"
    assert proposal["proposal_schema"] == "shion_self_hypothesis_v1"
    assert proposal["human_decision_status"] == "needs_human_review"
    assert improvement["surface"] == "shion_self_proposal"
    assert improvement["proposed_by"] == "shion"
    assert improvement["event_type"] == "domestic_mode_self_proposal"


def test_domestic_memory_context_summarizes_latest_status(tmp_path, monkeypatch):
    import api.domestic_mode as domestic

    proposals_path = tmp_path / "domestic_mode_proposals.jsonl"
    improvement_log_path = tmp_path / "cloudrun_improvement_log.jsonl"
    monkeypatch.setattr(domestic, "_PROPOSALS_PATH", proposals_path)
    monkeypatch.setattr(domestic, "_IMPROVEMENT_LOG_PATH", improvement_log_path)

    proposal = {
        "title": "内政再判定: FAQを観測継続で扱う",
        "canonical_key": "proposal:test",
        "decision_layer": "観測継続",
        "success_metric": "問い合わせ件数",
        "risk": "触らない線: FAQ原文は消さない",
        "status": "needs_human_review",
    }
    proposals_path.write_text(json.dumps(proposal, ensure_ascii=False) + "\n", encoding="utf-8")
    improvement_log_path.write_text(
        json.dumps({"canonical_key": "proposal:test", "status": "approved"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    context = domestic.build_domestic_memory_context()

    assert "内政モード判断メモリ" in context
    assert "status=approved" in context
    assert "問い合わせ件数" in context
    assert "触らない線" in context


def test_self_proposal_auto_connects_to_domestic_mode(tmp_path, monkeypatch):
    import api.domestic_mode as domestic

    proposals_path = tmp_path / "domestic_mode_proposals.jsonl"
    monkeypatch.setattr(domestic, "_PROPOSALS_PATH", proposals_path)
    monkeypatch.setattr(domestic, "_IMPROVEMENT_LOG_PATH", tmp_path / "cloudrun_improvement_log.jsonl")

    proposal = {
        "title": "FAQ導線を見直す",
        "target_page": "/faq",
        "hypothesis": "必要な場面で見つかれば問い合わせが減る",
        "evidence": "/faq の低利用",
        "proposed_change": "審査画面からFAQへ導線を出す",
        "success_metric": "問い合わせ件数",
        "verification_plan": "30日後に比較",
        "risk": "画面が混む",
        "priority": "medium",
        "generated_at": "2026-07-31T09:00:00",
    }
    result = domestic.connect_self_proposal_to_domestic_mode(
        proposal,
        source="usage_loop",
        kind="画面利用",
    )
    duplicate = domestic.connect_self_proposal_to_domestic_mode(
        proposal,
        source="usage_loop",
        kind="画面利用",
    )

    rows = [json.loads(line) for line in proposals_path.read_text(encoding="utf-8").splitlines()]

    assert result["saved"] is True
    assert duplicate["saved"] is False
    assert len(rows) == 1
    assert rows[0]["source"] == "domestic_mode_auto_connect"
    assert rows[0]["source_proposal_source"] == "usage_loop"
    assert rows[0]["domestic_status"] == "observe"
    assert rows[0]["success_metric"] == "問い合わせ件数"
