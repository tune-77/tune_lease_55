from __future__ import annotations

import typesafe_research_verify_guard as guard

_NOTE = """## 結論
- 補助金の交付決定前の検収は避ける。

## 判断に使える確認済み事実
- 2026年度のものづくり補助金は交付決定前の発注を補助対象外とする。
- 対象設備の法定耐用年数は7年である。
- 短い

## 担当者が確認する質問
- 交付決定通知の日付を確認できますか。
"""


def _choice(value: str, confidence: float) -> dict[str, object]:
    return {"type": "choice", "choice": value, "confidence": confidence}


def test_extract_claims_reads_only_the_confirmed_facts_section():
    claims = guard.extract_claims(_NOTE)
    assert claims == [
        "2026年度のものづくり補助金は交付決定前の発注を補助対象外とする。",
        "対象設備の法定耐用年数は7年である。",
    ], "推論や確認質問の節を混ぜると、原文と一致しないのが正常な文を誤検知する"


def test_mode_defaults_to_off():
    assert guard.verify_mode({}) == "off"
    assert guard.verify_mode({"TYPESAFE_RESEARCH_VERIFY_MODE": "shadow"}) == "shadow"


def test_unsupported_claim_is_flagged_and_rendered():
    def fake_request(payload):
        assert len(payload["state"]["claims"]) == 2
        return {
            "answers": {
                "c0_verdict": _choice("verified", 0.92),
                "c1_verdict": _choice("unsupported", 0.88),
            },
            "model": "jev-latest",
        }

    result = guard.verify_note_claims(_NOTE, "補助金は交付決定前の発注を対象外とする。", request_fn=fake_request)

    assert result["counts"] == {"verified": 1, "contradicted": 0, "unsupported": 1, "needs_review": 0}
    assert len(result["flagged"]) == 1
    section = guard.render_verification_section(result)
    assert "出典突き合わせ" in section
    assert "原文に記載なし" in section
    assert "法定耐用年数は7年" in section


def test_low_confidence_unsupported_goes_to_needs_review_not_flagged():
    def fake_request(payload):
        return {
            "answers": {
                "c0_verdict": _choice("unsupported", 0.40),
                "c1_verdict": _choice("verified", 0.95),
            },
            "model": "jev-latest",
        }

    result = guard.verify_note_claims(_NOTE, "根拠テキスト", request_fn=fake_request)
    assert result["flagged"] == [], "確信の薄い判定を「裏が取れない」と断定しない"
    # REV-424: ただし黙って通すと未検証の数値が判断資産へ流れるので要確認に出す。
    assert [item["reason"] for item in result["needs_review"]] == ["low_confidence_unsupported"]
    section = guard.render_verification_section(result)
    assert "要確認" in section and "確信低" in section
    assert "原文に記載なし" not in section


def test_note_without_claims_skips_the_request():
    def fail_request(payload):  # pragma: no cover
        raise AssertionError("主張が無いのに課金した")

    result = guard.verify_note_claims("## 結論\n本文なし\n", "根拠", request_fn=fail_request)
    assert result["status"] == "skipped"


def test_missing_and_invalid_answers_are_not_silently_dropped():
    def fake_request(payload):
        # c0 は回答欠落、c1 は選択肢外（壊れた回答）
        return {"answers": {"c1_verdict": _choice("maybe", 0.9)}, "model": "jev-latest"}

    result = guard.verify_note_claims(_NOTE, "根拠", request_fn=fake_request)

    assert result["judged_count"] == 0
    assert [(item["index"], item["reason"]) for item in result["needs_review"]] == [
        (0, "missing_answer"),
        (1, "invalid_answer"),
    ]
    assert result["counts"]["needs_review"] == 2
    section = guard.render_verification_section(result)
    assert "判定なし" in section and "判定不正" in section


def test_non_numeric_confidence_is_invalid_answer():
    def fake_request(payload):
        return {
            "answers": {
                "c0_verdict": {"type": "choice", "choice": "verified", "confidence": "high"},
                "c1_verdict": _choice("verified", 0.9),
            }
        }

    result = guard.verify_note_claims(_NOTE, "根拠", request_fn=fake_request)
    assert [item["reason"] for item in result["needs_review"]] == ["invalid_answer"]
    assert guard.judgment_route(result["results"][0]) == "needs_review"
    assert guard.judgment_route(result["results"][1]) == "verified"


def test_confident_verified_claims_render_nothing():
    def fake_request(payload):
        return {"answers": {"c0_verdict": _choice("verified", 0.9), "c1_verdict": _choice("verified", 0.3)}}

    result = guard.verify_note_claims(_NOTE, "根拠", request_fn=fake_request)
    assert result["needs_review"] == []
    assert guard.render_verification_section(result) == ""
