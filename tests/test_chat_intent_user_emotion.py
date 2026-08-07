"""ユーザー感情推定マーカーの抽出とガイダンス注入のテスト。

/api/chat の通常紫苑応答パス（chat_intent.build_chat_guidance）が、
既存LLM呼び出しに相乗りする形でユーザーの感情トーンを推定し、
応答本文からマーカーだけを安全に取り除けることを検証する。
"""
from __future__ import annotations

from chat_intent import build_chat_guidance, extract_estimated_user_emotion


def test_extract_estimated_user_emotion_strips_marker():
    reply = "<!--user_emotion:疲れている-->\nお疲れさまです。ご質問についてですが..."
    label, cleaned = extract_estimated_user_emotion(reply)
    assert label == "疲れている"
    assert cleaned == "お疲れさまです。ご質問についてですが..."


def test_extract_estimated_user_emotion_missing_marker_is_noop():
    reply = "マーカーのない通常の応答"
    label, cleaned = extract_estimated_user_emotion(reply)
    assert label == ""
    assert cleaned == reply


def test_extract_estimated_user_emotion_handles_empty_string():
    label, cleaned = extract_estimated_user_emotion("")
    assert label == ""
    assert cleaned == ""


def test_build_chat_guidance_includes_user_emotion_instruction():
    guidance = build_chat_guidance("おはよう")
    assert "user_emotion" in guidance.prompt_suffix
    assert "審査基準" in guidance.prompt_suffix
