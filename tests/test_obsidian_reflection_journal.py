import datetime as dt
from pathlib import Path

from scripts.obsidian_reflection_journal import build_journal_entry


def _fake_generate(prompt, **kwargs):
    return {
        "used": True,
        "status": "ok",
        "text": (
            "PROBLEM: 情報が多すぎて整理できていない。\n"
            "INSIGHT: 個人の学習ログが実は一次情報として価値がある。\n"
            "SOLUTION: 週次で学習ログを1本の記事にまとめる。"
        ),
        "error": "",
    }


REPORT = {
    "selected_theme": "機械学習",
    "proposed_angle": {"angle": "機械学習×システム改善の切り口", "status": "ok"},
    "buzz_checks": [{"theme": "機械学習", "sources": [{"title": "A", "uri": "http://a"}], "summary": "話題です"}],
}


def test_writes_note_under_system_improvement_reflection_not_private_reflection(tmp_path):
    vault = tmp_path

    result = build_journal_entry(vault, REPORT, generate_fn=_fake_generate)

    assert result["status"] == "written"
    note_path = Path(result["note_path"])
    assert note_path.exists()
    assert "System Improvement Reflection" in note_path.parts
    assert "Private Reflection" not in note_path.parts
    text = note_path.read_text(encoding="utf-8")
    assert "問題（困っていること）" in text
    assert "気づき（新しい見方）" in text
    assert "解決（具体的にどうすればいいか）" in text


def test_skips_when_no_selected_theme(tmp_path):
    result = build_journal_entry(tmp_path, {"selected_theme": "", "proposed_angle": {}}, generate_fn=_fake_generate)
    assert result["status"] == "skipped_no_selected_theme"
    assert result["written"] is False


def test_skips_when_report_missing(tmp_path):
    result = build_journal_entry(tmp_path, None, generate_fn=_fake_generate)
    assert result["status"] == "skipped_no_theme_radar_report"


def test_does_not_overwrite_existing_entry_same_day(tmp_path):
    vault = tmp_path
    today = dt.date(2026, 8, 19)

    first = build_journal_entry(vault, REPORT, today=today, generate_fn=_fake_generate)
    assert first["status"] == "written"

    second = build_journal_entry(vault, REPORT, today=today, generate_fn=_fake_generate)
    assert second["status"] == "skipped_already_exists"


def test_force_overwrites_existing_entry(tmp_path):
    vault = tmp_path
    today = dt.date(2026, 8, 19)

    build_journal_entry(vault, REPORT, today=today, generate_fn=_fake_generate)
    second = build_journal_entry(vault, REPORT, today=today, generate_fn=_fake_generate, force=True)
    assert second["status"] == "written"
