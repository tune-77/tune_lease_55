from __future__ import annotations

import lease_intelligence_mind as mind


def test_typesafe_recipe_request_is_one_choice():
    payload = mind.build_typesafe_recipe_classification_request(
        "改善タイトル: 表示ラベル修正\n変更対象ファイル: frontend/page.tsx",
        model="jev-test",
    )

    assert payload["model"] == "jev-test"
    question = payload["questions"]["recommendation"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"auto", "discuss", "review"}


def test_high_confidence_typesafe_recipe_result_is_used():
    result = mind._classify_recipe_with_typesafe(
        "改善タイトル: 表示ラベル修正\n変更対象ファイル: frontend/page.tsx",
        request_fn=lambda _payload: {
            "model": "jev-test",
            "answers": {
                "recommendation": {
                    "type": "choice",
                    "choice": "auto",
                    "confidence": 0.94,
                }
            },
            "usage": {"input_tokens": 30},
        },
    )

    assert result == {
        "recommendation": "auto",
        "reason": "小さく可逆的な表示変更として自動対応可能",
        "confidence": 0.94,
        "provider": "typesafe",
    }


def test_low_confidence_recipe_result_falls_back():
    result = mind._classify_recipe_with_typesafe(
        "改善タイトル: 表示ラベル修正",
        request_fn=lambda _payload: {
            "answers": {
                "recommendation": {
                    "type": "choice",
                    "choice": "auto",
                    "confidence": 0.50,
                }
            }
        },
    )

    assert result is None


def test_sensitive_recipe_is_not_sent():
    result = mind._classify_recipe_with_typesafe(
        "A社の案件番号 ABC-1234 を修正",
        request_fn=lambda _payload: (_ for _ in ()).throw(AssertionError("must not send")),
    )

    assert result is None


def test_recipe_typesafe_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TYPESAFE_RECIPE_CLASSIFY_ENABLED", "0")

    result = mind._classify_recipe_with_typesafe(
        "改善タイトル: 表示ラベル修正",
        request_fn=lambda _payload: (_ for _ in ()).throw(AssertionError("must not send")),
    )

    assert result is None
