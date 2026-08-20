import json
from types import SimpleNamespace

from api.chat_judgment_asset_capture import (
    chat_judgment_asset_candidate_type,
    create_manual_judgment_asset_candidate,
    extract_chat_judgment_asset_claim,
    load_autoresearch_judgment_asset_candidates,
)


def test_load_autoresearch_judgment_asset_candidates_overlays_state(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    state = tmp_path / "state.json"
    candidates.write_text(
        json.dumps(
            {
                "id": "c1",
                "claim": "審査では、受注根拠を見る。",
                "edited_claim": "審査では、受注根拠を見る。",
                "candidate_type": "application_rule",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    state.write_text(json.dumps({"c1": {"use_count": 3, "verified_status": "verified"}}), encoding="utf-8")

    rows = load_autoresearch_judgment_asset_candidates(
        candidates_jsonl=candidates,
        candidate_state_json=state,
        limit=10,
    )

    row = next(item for item in rows if item["id"] == "c1")
    assert row["use_count"] == 3
    assert row["verified_status"] == "verified"
    assert all(item["id"] != "demo-renewal-asset-candidate" for item in rows)


def test_create_manual_judgment_asset_candidate_writes_jsonl_and_state(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    state = tmp_path / "state.json"
    req = SimpleNamespace(
        claim="判断基準として、増車案件は荷主契約と運転手確保をセットで確認する。",
        candidate_type="confirmation_question",
        research_topic="chat_judgment_teaching",
        case_id="chat:u1",
        review_id=0,
    )

    row = create_manual_judgment_asset_candidate(
        req,
        candidates_jsonl=candidates,
        candidate_state_json=state,
    )

    written = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    state_payload = json.loads(state.read_text(encoding="utf-8"))
    assert written == [row]
    assert row["candidate_type"] == "confirmation_question"
    assert state_payload[row["id"]]["verification_note"] == "manual_candidate_created"


def test_extract_chat_judgment_asset_claim_and_candidate_type():
    assert extract_chat_judgment_asset_claim("この案件はどう判断すればいい？") == ""
    claim = extract_chat_judgment_asset_claim(
        "審査では、設備更新案件は既存機の稼働実績と受注根拠をセットで確認する。"
    )

    assert "設備更新案件" in claim
    assert chat_judgment_asset_candidate_type(claim) == "confirmation_question"
    assert chat_judgment_asset_candidate_type("条件付き承認なら保証人追加を見る") == "condition_signal"
    assert chat_judgment_asset_candidate_type("粉飾兆候に注意する") == "caution"
