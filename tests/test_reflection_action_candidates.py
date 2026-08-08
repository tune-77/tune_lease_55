from __future__ import annotations

import json
from pathlib import Path

from scripts import build_reflection_action_candidates as mod


def _delta() -> dict:
    return {
        "target_date": "2026-08-09",
        "sources": {
            "today_reflection": {"path": "/vault/Private Reflection/2026-08-09.md"},
            "today_memory": {"path": "/repo/memory/2026-08-09.md"},
        },
        "metrics": {"reflection_similarity_to_yesterday": 0.42},
        "quality": {"status": "pass"},
        "operational_handoff": {
            "shion_next_actions": [
                "次回は内省を、補助金案件の確認項目と判断資産レビューへ戻す。",
            ],
            "user_requests": [
                "内省アクション候補が現場で使える文面か、採用・修正・却下で確認してもらう。",
            ],
        },
        "judgment_change_log": {
            "次回から変える確認事項": "過去案件の学びとして、条件付き承認時の追加確認を先に出す。",
            "判断資産候補": "標準承認でも返済原資と設備用途を一文で残す。",
        },
    }


def test_build_candidates_turns_reflection_into_reviewable_actions():
    candidates = mod.build_candidates(_delta(), generated_at="2026-08-09T00:00:00")

    assert candidates
    first = candidates[0]
    assert first["proposal_schema"] == "shion_self_hypothesis_v1"
    assert first["action_schema"] == "shion_reflection_action_candidate_v1"
    assert first["human_decision_status"] == "needs_human_review"
    assert first["review_policy"] == "human_decision_required"
    assert first["canonical_key"].startswith("reflection_action:")
    assert "採用(修正)/保留/却下" in first["effect_tracking"]
    assert "User判断なしにプロンプト・RAG・判断資産へ昇格しない" in first["risk"]


def test_main_appends_only_new_candidates(tmp_path: Path):
    delta_path = tmp_path / "delta.json"
    output_path = tmp_path / "reflection_action_candidates.jsonl"
    report_path = tmp_path / "reflection_action_candidates.md"
    delta_path.write_text(json.dumps(_delta(), ensure_ascii=False), encoding="utf-8")

    assert mod.main.__name__ == "main"
    payload = mod._load_json(delta_path)
    candidates = mod.build_candidates(payload, generated_at="2026-08-09T00:00:00")
    mod._append_jsonl(output_path, candidates)
    existing = mod._existing_keys(output_path)
    new_candidates = [item for item in candidates if item["canonical_key"] not in existing]
    mod._append_jsonl(output_path, new_candidates)
    mod.write_report(candidates, report_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(candidates)
    assert "human_decision_required" in report_path.read_text(encoding="utf-8")
