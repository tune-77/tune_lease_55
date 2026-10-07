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


def _capture_kwargs(created):
    return dict(
        user_id="u1",
        surface="test",
        candidates_loader=lambda limit=1000: [],
        candidate_creator=lambda req: created.append(req) or {"id": "n1", "claim": req.claim},
        request_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        cloudrun_event_recorder=lambda **_kwargs: {"status": "skipped"},
    )


def test_capture_saves_by_rule_even_if_jev_shadow_fails():
    from api.chat_judgment_asset_capture import capture_chat_judgment_asset_if_needed

    created = []

    def broken_shadow(_claim):
        raise RuntimeError("jev down")

    result = capture_chat_judgment_asset_if_needed(
        "歯科医院の開業前リースは、銀行の融資実行日と見積書の日付が一致しているか必ず確認する。",
        jev_shadow=broken_shadow,
        **_capture_kwargs(created),
    )

    assert result["captured"] is True
    assert len(created) == 1


def test_record_chat_teaching_jev_shadow_logs_probability_without_text(tmp_path, monkeypatch):
    from api.chat_judgment_asset_capture import record_chat_teaching_jev_shadow

    log = tmp_path / "jev.jsonl"
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", str(log))
    claim = "運送業は燃料費の変動で返済原資が揺れるので資金繰り表を確認する。"

    record_chat_teaching_jev_shadow(claim, judge=lambda _c: 0.123456)
    record_chat_teaching_jev_shadow(claim, judge=lambda _c: None)

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["guard"] == "chat_teaching_capture"
    assert rows[0]["mode"] == "shadow"
    assert rows[0]["probability"] == 0.123
    assert claim not in log.read_text(encoding="utf-8")


def test_mask_for_jev_hides_company_and_person_and_blocks_pii():
    from api.chat_judgment_asset_capture import mask_for_jev

    masked = mask_for_jev("株式会社ヤマダ運輸の山田社長は資金繰りに詳しい")
    assert "ヤマダ" not in masked and "山田" not in masked
    assert mask_for_jev("連絡先は 03-1234-5678 です") == ""


def test_capture_records_content_source_user_or_shion(tmp_path):
    # REV-498: ユーザーの教示は user、「保存して」と頼まれた紫苑の返答は shion として登録する
    from api.chat_judgment_asset_capture import capture_chat_judgment_asset_if_needed, create_manual_judgment_asset_candidate

    created = []
    capture_chat_judgment_asset_if_needed(
        "歯科医院の開業前リースは、銀行の融資実行日と見積書の日付が一致しているか必ず確認する。",
        jev_shadow=lambda _claim: None,
        **_capture_kwargs(created),
    )
    capture_chat_judgment_asset_if_needed(
        "承知いたしました。### 稟議コメントテンプレート 返済原資と保全を分けて書く。",
        jev_shadow=lambda _claim: None,
        user_requested=True,
        **_capture_kwargs(created),
    )

    assert [req.content_source for req in created] == ["user", "shion"]

    row = create_manual_judgment_asset_candidate(
        SimpleNamespace(
            claim=created[1].claim, candidate_type="application_rule", research_topic="chat_judgment_teaching",
            case_id="chat:u1", review_id=None, content_source="shion",
        ),
        candidates_jsonl=tmp_path / "c.jsonl",
        candidate_state_json=tmp_path / "s.json",
    )
    assert row["content_source"] == "shion"
