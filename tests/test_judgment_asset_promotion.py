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
