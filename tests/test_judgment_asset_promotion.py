"""判断資産レビュー・昇格（/api/judgment-assets/promotion-candidates）の回帰テスト。

2026-08-20: 「昇格ボタンを押しても候補が消えず、正規判断資産の件数も増えない」
不具合の再発防止用。原因は2つあった:
  1. Cloud Run実行時のみ永続化をスキップするガード（K_SERVICE分岐）
  2. demo-renewal-asset-candidate が保存済みstateを無視して毎回
     promotion_status="not_promoted" で固定表示されるハードコード
"""
import json

import api.routers.feedback_loop as feedback_loop


def _patch_paths(monkeypatch, tmp_path):
    candidates_jsonl = tmp_path / "autoresearch_judgment_asset_candidates.jsonl"
    state_json = tmp_path / "autoresearch_judgment_asset_candidate_state.json"
    canonical_json = tmp_path / "canonical_judgment_rules.json"
    monkeypatch.setattr(feedback_loop, "_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL", candidates_jsonl)
    monkeypatch.setattr(feedback_loop, "_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON", state_json)
    monkeypatch.setattr(feedback_loop, "_CANONICAL_JUDGMENT_RULES_JSON", canonical_json)
    return candidates_jsonl, state_json, canonical_json


def _write_candidate(path, **overrides):
    row = {
        "id": "cand-1",
        "claim": "更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べて確認する。",
        "edited_claim": "",
        "candidate_type": "application_rule",
        "research_topic": "manual_screening",
        "edit_count": 1,
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "promotion_status": "not_promoted",
        "verified_status": "unverified",
        "source_section": "manual_input",
    }
    row.update(overrides)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return row


def test_promote_adds_canonical_rule_and_removes_candidate_from_list(tmp_path, monkeypatch):
    candidates_jsonl, _state_json, _canonical_json = _patch_paths(monkeypatch, tmp_path)
    _write_candidate(candidates_jsonl)

    before = feedback_loop._load_judgment_asset_promotion_candidates(limit=30)
    assert any(item["id"] == "cand-1" for item in before)
    assert feedback_loop._load_canonical_judgment_asset_candidates(limit=100) == []

    result = feedback_loop._promote_judgment_asset_candidate_to_canonical("cand-1")
    assert result["status"] == "promoted"
    assert result["active_rules"] == 1

    after = feedback_loop._load_judgment_asset_promotion_candidates(limit=30)
    assert all(item["id"] != "cand-1" for item in after)

    active = feedback_loop._load_canonical_judgment_asset_candidates(limit=100)
    assert len(active) == 1


def test_promote_duplicate_statement_updates_existing_rule_instead_of_duplicating(tmp_path, monkeypatch):
    candidates_jsonl, _state_json, canonical_json = _patch_paths(monkeypatch, tmp_path)
    statement = "更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べて確認する。"
    _write_candidate(candidates_jsonl, id="cand-2", edit_count=2, claim=statement)
    canonical_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rules": [
                    {
                        "id": "existing-rule",
                        "status": "active",
                        "domain": "lease_screening",
                        "canonical_statement": statement,
                        "evidence_count": 1,
                        "user_evidence_count": 0,
                        "confidence": 0.8,
                        "private": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = feedback_loop._promote_judgment_asset_candidate_to_canonical("cand-2")
    assert result["status"] == "updated"
    assert result["active_rules"] == 1

    after = feedback_loop._load_judgment_asset_promotion_candidates(limit=30)
    assert all(item["id"] != "cand-2" for item in after)


def test_review_reject_removes_candidate_from_list(tmp_path, monkeypatch):
    candidates_jsonl, _state_json, _canonical_json = _patch_paths(monkeypatch, tmp_path)
    _write_candidate(candidates_jsonl, id="cand-3")

    req = feedback_loop.JudgmentAssetPromotionReviewRequest(action="reject", comment="対象外")
    result = feedback_loop._review_judgment_asset_promotion_candidate("cand-3", req)
    assert result["candidate"]["promotion_status"] == "rejected"

    after = feedback_loop._load_judgment_asset_promotion_candidates(limit=30)
    assert all(item["id"] != "cand-3" for item in after)


def test_no_hardcoded_demo_candidate_survives_state_reload(tmp_path, monkeypatch):
    """demo-renewal-asset-candidate は削除済み。読み込み結果に混入しないことを確認する。"""
    candidates_jsonl, _state_json, _canonical_json = _patch_paths(monkeypatch, tmp_path)
    _write_candidate(candidates_jsonl, id="cand-4")

    rows = feedback_loop._load_autoresearch_judgment_asset_candidates(limit=100)
    assert all(row.get("id") != "demo-renewal-asset-candidate" for row in rows)


def test_rule_flagged_candidate_without_usage_reaches_review_list_after_evidence(tmp_path, monkeypatch):
    candidates_jsonl, state_json, _canonical_json = _patch_paths(monkeypatch, tmp_path)
    base = {
        "edited_claim": "", "edit_count": 0, "use_count": 0, "useful_count": 0, "rejected_count": 0,
        "verified_status": "unverified", "source_section": "担当者が確認する質問", "research_topic": "topic",
    }
    rows = [
        {**base, "id": "used", "claim": "更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べて確認する。",
         "candidate_type": "application_rule", "promotion_status": "not_promoted", "asset_quality": "actionable",
         "use_count": 2, "useful_count": 1},
        {**base, "id": "flag-rule", "claim": "黒字だけで資金繰りが健全と過信しない運用ルールの候補です。",
         "candidate_type": "application_rule", "promotion_status": "needs_review_quality",
         "asset_quality": "textbook_general", "research_date": "2026-09-30"},
        {**base, "id": "flag-caution", "claim": "物件受領証があっても納入済みとは限らないと考える戒めの候補です。",
         "candidate_type": "caution", "asset_quality": "textbook_general", "research_date": "2026-09-01"},
        {**base, "id": "plain-unused", "claim": "使用実績のない通常候補は従来どおり一覧に出さないことを確認する。",
         "candidate_type": "caution", "promotion_status": "not_promoted", "asset_quality": "actionable"},
    ]
    candidates_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # 旧版が状態ファイルに残した黙殺ステータスでも要確認として出る
    state_json.write_text(json.dumps({"flag-caution": {"promotion_status": "not_promoted_textbook_general"}}), encoding="utf-8")

    listed = feedback_loop._load_judgment_asset_promotion_candidates(limit=30)
    ids = [item["id"] for item in listed]
    assert ids == ["used", "flag-caution", "flag-rule"]
    assert all(item["promotion_status"] == "needs_review_quality" for item in listed if item["rule_review"])


def _write_chat_candidates(path, count):
    rows = [
        {
            "id": f"chat-{n:03d}",
            "claim": f"取引先{n}番の見積書は、銀行の融資実行日と日付が一致しているかを確認する。",
            "candidate_type": "caution",
            "research_topic": "chat_judgment_teaching",
            "research_date": "2026-10-02",
            "source_section": "manual_input",
            "promotion_status": "needs_review_quality",
            "edit_count": 1,
        }
        for n in range(count)
    ]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def test_list_reports_total_beyond_page_and_promotes_past_rank_100(tmp_path, monkeypatch):
    """2026-10-02: 一覧は上限30件で切った件数を「昇格候補」に出していたため、押しても件数が
    動かず「消えない」ように見えた。昇格も上限100件の中しか探さず101位以下は404だった。"""
    candidates_jsonl, _state, _canonical = _patch_paths(monkeypatch, tmp_path)
    rows = _write_chat_candidates(candidates_jsonl, 120)

    listed = feedback_loop.get_judgment_asset_promotion_candidates(limit=30)
    assert listed["count"] == 30 and listed["total_count"] == 120

    ranked_ids = [item["id"] for item in feedback_loop._rank_judgment_asset_promotion_candidates()]
    last = ranked_ids[-1]
    assert feedback_loop._promote_judgment_asset_candidate_to_canonical(last)["status"] == "promoted"
    assert feedback_loop.get_judgment_asset_promotion_candidates(limit=30)["total_count"] == len(rows) - 1


def test_reviewed_chat_candidates_stay_out_after_daily_recompute(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import build_autoresearch_judgment_asset_candidates as builder

    candidates_jsonl, state_json, _canonical = _patch_paths(monkeypatch, tmp_path)
    _write_chat_candidates(candidates_jsonl, 3)
    feedback_loop._promote_judgment_asset_candidate_to_canonical("chat-000")
    feedback_loop._review_judgment_asset_promotion_candidate(
        "chat-001", SimpleNamespace(action="reject", comment="")
    )

    # 日次再計算（main と同じ順: write_state → preserve_reviewed_candidates → write_jsonl）
    builder.write_state(state_json, [], builder.load_state(state_json))
    kept = builder.preserve_reviewed_candidates([], existing_jsonl=candidates_jsonl, state_path=state_json)
    builder.write_jsonl(candidates_jsonl, kept)

    listed = [item["id"] for item in feedback_loop._rank_judgment_asset_promotion_candidates()]
    assert listed == ["chat-002"]
    state = json.loads(state_json.read_text(encoding="utf-8"))
    assert state["chat-000"]["promotion_status"] == "promoted"
    assert state["chat-001"]["promotion_status"] == "rejected"
