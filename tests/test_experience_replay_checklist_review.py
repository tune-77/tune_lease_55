from __future__ import annotations

import json

from scripts import review_experience_replay_checklist as review


def _candidate_payload() -> dict:
    return {
        "mode": "experience_replay_response_checklist_candidates",
        "global_checklist_candidates": [
            {
                "id": "global_missing_01",
                "trigger": "lease_practical_answer",
                "checklist_item": "回答前に `所有者 / 使用者` が今回の判断に関係するか確認する。",
                "evidence_count": 3,
            }
        ],
        "case_checklist_candidates": [
            {
                "id": "exp_replay_check_01",
                "source_case_id": "case_food",
                "query": "飲食業をやりたいんだけどリース使う？",
                "topic": "飲食業リース",
                "priority": 92,
                "checklist_items": [
                    "飲食業は初期投資メリットだけで終えず、撤退時の物件回収・中古市場・原状回復も見る。"
                ],
            },
            {
                "id": "exp_replay_check_02",
                "source_case_id": "case_car",
                "query": "レンタカーってリースできる？",
                "topic": "自動車周辺リスク",
                "priority": 90,
                "checklist_items": ["自動車系は車検・登録・所有者/使用者を分けて確認する。"],
            },
        ],
    }


def test_active_checklist_uses_only_accepted_or_revised_reviews(tmp_path):
    payload = _candidate_payload()
    reviews_path = tmp_path / "reviews.jsonl"

    review.record_review(
        candidate_id="global_missing_01",
        decision="accepted",
        candidates_payload=payload,
        reviews_path=reviews_path,
    )
    review.record_review(
        candidate_id="exp_replay_check_01",
        decision="revised",
        candidates_payload=payload,
        edited_items=["飲食業では撤退時の物件回収・中古市場・原状回復を必ず見る。"],
        reviews_path=reviews_path,
    )
    review.record_review(
        candidate_id="exp_replay_check_02",
        decision="rejected",
        candidates_payload=payload,
        reviews_path=reviews_path,
    )

    active = review.build_active_checklist(payload, reviews_path=reviews_path)

    assert active["summary"]["active_count"] == 2
    assert active["summary"]["review_counts"]["rejected"] == 1
    active_ids = {item["id"] for item in active["active_items"]}
    assert active_ids == {"global_missing_01", "exp_replay_check_01"}
    revised = next(item for item in active["active_items"] if item["id"] == "exp_replay_check_01")
    assert revised["checklist_items"] == ["飲食業では撤退時の物件回収・中古市場・原状回復を必ず見る。"]


def test_write_outputs_persists_active_json_and_review_report(tmp_path):
    payload = _candidate_payload()
    reviews_path = tmp_path / "reviews.jsonl"
    active_json = tmp_path / "active.json"
    report_md = tmp_path / "review.md"
    review.record_review(
        candidate_id="global_missing_01",
        decision="accepted",
        candidates_payload=payload,
        reviews_path=reviews_path,
    )

    reviewed = review.overlay_reviews(review.normalize_candidates(payload), reviews_path=reviews_path)
    active = review.build_active_checklist(payload, reviews_path=reviews_path)
    review.write_outputs(active, reviewed, active_json=active_json, report_md=report_md)

    saved = json.loads(active_json.read_text(encoding="utf-8"))
    assert saved["mode"] == "experience_replay_accepted_reflection_gate_checklist"
    assert saved["active_items"][0]["id"] == "global_missing_01"
    assert "# Experience Replay Checklist Review" in report_md.read_text(encoding="utf-8")
