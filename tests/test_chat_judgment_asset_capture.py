import json

import api.main as main


def test_extract_chat_judgment_asset_claim_requires_teaching_signal():
    assert main._extract_chat_judgment_asset_claim("この案件はどう判断すればいい？") == ""

    claim = main._extract_chat_judgment_asset_claim(
        "審査では、新店舗案件は初期赤字だけで止めず、既存店実績と撤退条件をセットで見る。"
    )

    assert "新店舗案件" in claim
    assert "撤退条件" in claim


def test_capture_chat_judgment_asset_creates_manual_candidate(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.jsonl"
    state = tmp_path / "state.json"
    monkeypatch.setattr(main, "_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL", candidates)
    monkeypatch.setattr(main, "_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON", state)
    monkeypatch.setattr(
        main,
        "record_cloudrun_input_event",
        lambda **kwargs: {"ok": True, "event_type": kwargs["event_type"]},
    )

    result = main._capture_chat_judgment_asset_if_needed(
        "判断方法として、運送業の増車は荷主契約期間とドライバー確保をセットで確認する。",
        user_id="u1",
        surface="test_chat",
        response_mode="shion",
    )

    assert result["captured"] is True
    assert result["candidate"]["research_topic"] == "chat_judgment_teaching"
    rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["candidate_type"] == "confirmation_question"
    assert "荷主契約期間" in rows[0]["claim"]


def test_capture_chat_judgment_asset_deduplicates_same_claim(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.jsonl"
    state = tmp_path / "state.json"
    monkeypatch.setattr(main, "_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL", candidates)
    monkeypatch.setattr(main, "_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON", state)
    monkeypatch.setattr(main, "record_cloudrun_input_event", lambda **kwargs: {"ok": True})

    message = "判断基準として、工作機械更新は受注根拠と既存機の稼働状況をセットで見る。"
    first = main._capture_chat_judgment_asset_if_needed(
        message,
        user_id="u1",
        surface="test_chat",
        response_mode="shion",
    )
    second = main._capture_chat_judgment_asset_if_needed(
        message,
        user_id="u1",
        surface="test_chat",
        response_mode="shion",
    )

    rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    assert first["captured"] is True
    assert second["captured"] is True
    assert second["duplicate"] is True
    assert len(rows) == 1
