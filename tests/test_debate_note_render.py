import datetime as dt

from api.debate_note_render import (
    debate_grade,
    render_debate_obsidian_note,
    safe_debate_filename_part,
)


def test_debate_grade_uses_explicit_or_score_band():
    assert debate_grade(10, "B") == "B"
    assert debate_grade(80) == "A"
    assert debate_grade(60) == "B"
    assert debate_grade(40) == "C"
    assert debate_grade(20) == "D"
    assert debate_grade(19.9) == "E"


def test_safe_debate_filename_part_sanitizes_and_falls_back():
    assert safe_debate_filename_part('A/B:C*"D') == "A_B_C__D"
    assert safe_debate_filename_part("   ") == "不明"


def test_render_debate_obsidian_note_builds_filename_and_sections():
    req = {
        "company_name": "デモ/精密",
        "score": 56.5,
        "grade": "",
        "final_decision": "条件付承認",
        "cautious": {
            "opinion": "慎重",
            "reasons": ["資金繰りを見る"],
            "key_risks": ["在庫増加"],
        },
        "aggressive": {
            "opinion": "前向き",
            "reasons": ["更新需要あり"],
            "opportunities": ["省力化"],
        },
        "arbiter_summary": "条件を付ければ検討可能。",
        "conditions": ["月次試算表を確認"],
        "debate_log": "round1",
    }

    note = render_debate_obsidian_note(req, now=dt.datetime(2026, 8, 2, 9, 8, 7))

    assert note["filename"] == "2026-08-02_デモ_精密_090807.md"
    assert note["grade"] == "C"
    assert "score: 56.5" in note["content"]
    assert "### 石橋（慎重派）" in note["content"]
    assert "- 在庫増加" in note["content"]
    assert "### 風林火山（積極派）" in note["content"]
    assert "1. 月次試算表を確認" in note["content"]
    assert "```" in note["content"]
