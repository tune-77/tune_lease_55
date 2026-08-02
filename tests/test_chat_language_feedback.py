import json

from api.chat_language_feedback import (
    capture_language_judgment_material,
    record_response_impact_prediction,
    score_response_impact,
)


def _redact(text, *, limit):
    return str(text)[:limit].replace("secret", "[REDACTED]")


def _record(**kwargs):
    return {"ok": True, "event_type": kwargs["event_type"], "surface": kwargs["surface"]}


def test_capture_language_judgment_material_writes_jsonl(tmp_path):
    path = tmp_path / "language_materials.jsonl"

    result = capture_language_judgment_material(
        "審査ではこの言い回しを残したい secret",
        user_id="u1",
        surface="chat",
        response_mode="shion",
        category="test",
        language_materials_jsonl=path,
        redactor=_redact,
        cloudrun_event_recorder=_record,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert result["captured"] is True
    assert result["material"]["asset_stage"] == "raw_language_material"
    assert rows[0]["text"].endswith("[REDACTED]")
    assert result["writeback"]["event_type"] == "language_judgment_material_preserved"


def test_capture_language_judgment_material_handles_empty_after_redaction(tmp_path):
    result = capture_language_judgment_material(
        "secret",
        user_id="u1",
        surface="chat",
        language_materials_jsonl=tmp_path / "materials.jsonl",
        redactor=lambda text, *, limit: "",
        cloudrun_event_recorder=_record,
    )

    assert result == {"captured": False, "reason": "empty_after_redaction"}


def test_score_response_impact_flags_strong_short_reply_to_sensitive_message():
    score = score_response_impact("それは怖い", "絶対ダメ")

    assert score["reaction_risk"] == "high"
    assert "断定が強く" in " ".join(score["risk_factors"])
    assert score["shadow_mode"] is True


def test_record_response_impact_prediction_writes_shadow_entry(tmp_path):
    path = tmp_path / "response_impact.jsonl"

    result = record_response_impact_prediction(
        user_message="不安です",
        assistant_reply="ただし、前提を確認して次に進めます。",
        user_id="u1",
        surface="chat",
        response_mode="shion",
        category="test",
        response_impact_predictions_jsonl=path,
        redactor=_redact,
        cloudrun_event_recorder=_record,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert result["captured"] is True
    assert result["prediction"]["shadow_mode"] is True
    assert rows[0]["event_type"] == "response_impact_prediction"
    assert rows[0]["status"] == "shadow_only"
