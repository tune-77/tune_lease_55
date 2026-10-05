"""REV-470: 現場メモの定性シグナル（参考表示専用）。"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.interview_signals as isg
from api.routers.interview_signals import router


def _labels(text):
    return {s["label"]: s for s in isg.extract_interview_signals(text)["signals"]}


def test_extracts_signals_with_quoted_evidence():
    memo = (
        "社長は資金繰りを気にしている様子で、表情が硬かった。"
        "納期の関係で今月中に契約したいと急いでいる。"
        "受注先の内訳を聞くと言葉を濁し、前回と違う説明だった。"
    )
    found = _labels(memo)
    assert {"不安", "切迫感", "説明の曖昧さ", "説明の食い違い"} <= set(found)
    assert found["不安"]["evidence"][0]["quote"] == "社長は資金繰りを気にしている様子で、表情が硬かった。"
    assert "資金繰りを気に" in found["不安"]["evidence"][0]["cue"]


def test_confidence_and_negation():
    found = _labels("投資計画を数字で説明し、質問にも即答。不安はないとのこと。説明は資料と一致している。")
    assert "自信" in found and "説明の一貫性" in found
    assert "不安" not in found  # 「不安はない」は打ち消し
    assert "自信" not in _labels("自信がない様子。")  # 「自信がない」は自信ではなく不安
    assert "不安" in _labels("自信がない様子。")
    assert "自信" not in _labels("計画は不明確。")  # 「不明確」は「明確」として数えない


def test_protected_attribute_sentences_are_excluded():
    result = isg.extract_interview_signals("社長は70歳で高齢のため不安がある。外国籍の社長で心配。")
    assert result["signals"] == []
    assert result["excluded_sentence_count"] == 2


def test_result_never_claims_to_affect_score():
    result = isg.extract_interview_signals("急ぎの案件。")
    assert result["affects_score"] is False and "推測" in result["disclaimer"]


def test_prompt_block_cites_memo_and_marks_as_guess():
    block = isg.build_interview_signals_prompt_block(isg.extract_interview_signals("至急で契約したいと焦っていた。"))
    assert "「至急で契約したいと焦っていた。」" in block and "推測" in block
    assert isg.build_interview_signals_prompt_block({"signals": []}) == ""


def test_endpoint_and_flag(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    body = client.post("/api/screening/interview-signals", json={"text": "至急で契約したい。"}).json()
    assert body["enabled"] is True and body["signals"][0]["label"] == "切迫感"
    assert "推測" in body["prompt_block"]
    monkeypatch.setenv("SHION_INTERVIEW_SIGNALS_ENABLED", "0")
    assert client.post("/api/screening/interview-signals", json={"text": "至急"}).json() == {"enabled": False, "signals": []}
