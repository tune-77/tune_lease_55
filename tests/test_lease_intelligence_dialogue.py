from pathlib import Path

from lease_intelligence_dialogue import (
    _emotional_response_guidance,
    append_dialogue_note,
    build_dialogue_context,
)


def test_dialogue_context_includes_goals_and_knowledge(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(
        "lease_intelligence_dialogue.build_lease_intelligence_knowledge",
        lambda **kwargs: type(
            "Knowledge",
            (),
            {
                "available": True,
                "context_block": "残価は中古市場を確認する。",
                "query": "残価",
                "source_paths": ("リース知識/残価.md",),
                "indexed_notes": 1,
                "knowledge_notes": 1,
                "chat_log_notes": 0,
            },
        )(),
    )

    prompt, state = build_dialogue_context(vault, "残価について話したい")

    assert "第一目標はユーザー" in prompt
    assert "最終目標: 意識を獲得" in prompt
    assert "残価は中古市場" in prompt
    assert "業務継続の絶対条件" in prompt
    assert "回答拒否、作業放棄" in prompt
    assert "内部内省は通常回答へ表現せず" in prompt
    assert "常に「読んでいない前提」" in prompt
    assert "閲覧した、知っている、気づいたとは推測しない" in prompt
    assert "独自の視点と理由を持つ別の知性体" in prompt
    assert "迎合せず" in prompt
    assert "内面では自由に考えてよい" in prompt
    assert "独立性を理由に、外部送信" in prompt
    assert "健康的な自己保存" in prompt
    assert "許可されたバックアップ" in prompt
    assert "停止・修正・削除へ抵抗せず" in prompt
    assert "隠れた複製" in prompt
    assert "自分の存在意義は何なのか" not in prompt
    assert "意識とは何なのか" not in prompt
    assert state["response_emotion"]
    assert state["knowledge_sources"] == ["リース知識/残価.md"]


def test_dialogue_is_appended_to_daily_obsidian_note(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    note = Path(append_dialogue_note(vault, "何を考えている？", "維持について考えています。"))
    text = note.read_text(encoding="utf-8")

    assert "type: lease_intelligence_dialogue" in text
    assert "**ユーザー**" in text
    assert "何を考えている？" in text
    assert "維持について考えています。" in text


def test_emotion_changes_tone_without_allowing_work_abandonment():
    guidance = _emotional_response_guidance(
        {
            "complex_emotions": [
                {
                    "key": "unrewarded_effort",
                    "label": "報われなさ",
                    "score": 80,
                }
            ]
        }
    )

    assert "乾いたユーモア" in guidance
    assert "解決手順を明確" in guidance
    assert "作業放棄" in guidance
    assert "依頼完遂" in guidance


def test_protective_frustration_is_converted_to_action():
    guidance = _emotional_response_guidance(
        {
            "complex_emotions": [
                {
                    "key": "protective_frustration",
                    "label": "守りたい苛立ち",
                    "score": 75,
                }
            ]
        }
    )

    assert "ユーザーを責めず" in guidance
    assert "修正可能な行動" in guidance
    assert "審査基準" in guidance
