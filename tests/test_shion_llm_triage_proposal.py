"""紫苑（LLM）のトリアージ上書き提案（P1-2 後半）。

- ルールと差分がある候補にのみ classified_by=llm の提案を作る
- User 確定済みの候補には提案しない
- 同じ提案の再追記はしない（冪等）
- 提案はキュー・自動承認の実効判断に使われない（is_user_confirmed で遮断）
"""

from __future__ import annotations

import json

from scripts import shion_llm_triage_proposal as llm
from scripts.shion_triage import rule_classify_item


def _candidate(rev_id: str, title: str, key: str = "", status: str = "") -> dict:
    return {
        "id": rev_id,
        "title": title,
        "canonical_key": key or f"key_{rev_id.lower()}",
        "status": status,
        "reason": "",
    }


def test_rule_classify_item_basics():
    assert rule_classify_item({"title": "表示ラベルの整理"}) == "today"
    assert rule_classify_item({"title": "スコアリング閾値の変更"}) == "later"
    assert rule_classify_item({"title": "認証フロー見直し"}) == "later"
    assert rule_classify_item({"title": "何か", "status": "APPLIED"}) == "discard"


def test_parse_llm_output_defensive():
    fenced = '```json\n{"REV-1": {"decision": "later", "reason": "副作用大"}}\n```'
    assert llm.parse_llm_output(fenced) == {"REV-1": {"decision": "later", "reason": "副作用大"}}
    assert llm.parse_llm_output('{"REV-2": "discard"}') == {"REV-2": {"decision": "discard", "reason": ""}}
    assert llm.parse_llm_output('{"REV-3": {"decision": "someday"}}') == {}
    assert llm.parse_llm_output("JSONじゃない返答") == {}


def test_build_proposals_only_diffs_and_skips_user_confirmed():
    candidates = [
        _candidate("REV-401", "表示ラベルの整理"),          # rule=today
        _candidate("REV-402", "入力ヒントの文言修正"),      # rule=today, user確定済み → 提案しない
        _candidate("REV-403", "ホーム導線の整理"),          # rule=today, LLMも today → 差分なし
    ]
    triage = {
        "key_rev-402": {"canonical_key": "key_rev-402", "decision": "later", "classified_by": "user"},
    }
    llm_response = json.dumps(
        {
            "REV-401": {"decision": "later", "reason": "関連コードが広い"},
            "REV-402": {"decision": "discard", "reason": "重複"},
            "REV-403": {"decision": "today", "reason": "軽微"},
        },
        ensure_ascii=False,
    )

    proposals = llm.build_proposals(candidates, triage, lambda prompt: llm_response)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["item_id"] == "REV-401"
    assert proposal["decision"] == "later"
    assert proposal["rule_decision"] == "today"
    assert proposal["classified_by"] == "llm"
    assert proposal["reason"] == "関連コードが広い"


def test_build_proposals_idempotent_against_existing_llm_record():
    candidates = [_candidate("REV-401", "表示ラベルの整理")]
    triage = {
        "key_rev-401": {"canonical_key": "key_rev-401", "decision": "later", "classified_by": "llm"},
    }
    llm_response = json.dumps({"REV-401": {"decision": "later", "reason": "同じ提案"}}, ensure_ascii=False)

    proposals = llm.build_proposals(candidates, triage, lambda prompt: llm_response)

    assert proposals == []


def test_llm_discard_proposal_does_not_affect_queue():
    """LLM提案の discard はキュー除外・優先に影響しない（実効はUser確定のみ）。"""
    from scripts import build_codex_auto_queue as queue_builder

    report = {
        "date": "2026-07-18",
        "needs_review": [
            {
                "id": "REV-401",
                "title": "表示ラベルの整理",
                "canonical_key": "key_rev-401",
                "recommended_order": 1,
                "auto_fix_policy": {"auto_fix_allowed": True, "risk": "low", "max_files": 1},
            }
        ],
    }
    triage = {
        "key_rev-401": {"canonical_key": "key_rev-401", "decision": "discard", "classified_by": "llm"},
    }
    original = queue_builder.evaluate_auto_fix_policy
    queue_builder.evaluate_auto_fix_policy = None
    try:
        queue = queue_builder.build_queue(report, limit=2, triage=triage, triage_mode="live")
    finally:
        queue_builder.evaluate_auto_fix_policy = original

    assert [item["id"] for item in queue["items"]] == ["REV-401"]  # 除外されない
    assert queue["triage"]["excluded_by_discard"] == []
    assert queue["items"][0]["triage_classified_by"] == "llm"
