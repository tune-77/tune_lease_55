from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "lease_news_digest.py"
_SPEC = importlib.util.spec_from_file_location("lease_news_digest", _MODULE_PATH)
assert _SPEC and _SPEC.loader
digest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = digest
_SPEC.loader.exec_module(digest)


def test_write_lease_news_focus_note_creates_project_note_and_daily_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "novelist_agent.generate_daily_lease_grumble",
        lambda **_: [
            "今日も稟議書を開いた。",
            "数字は正直だが、営業の説明は長い。",
            "プリンの代わりに追加資料が届いた。",
            "明日も返済予定表と向き合う。",
        ],
    )
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "リースニュース"
    news_dir.mkdir(parents=True)

    note = news_dir / "2026-06-11_リースニュース_建設_AI導入.md"
    note.write_text(
        """---
date: 2026-06-11
week: 2026-W24
month: 2026-06
tags: ["建設/不動産", "製造/DX"]
region: 国内
source: Example News
importance: 中
---
# 建設会社がAI導入で事務作業を効率化

## 3行要約
- 建設会社がAIを導入し、事務作業を削減する。
- 省力化投資の効果が見えやすくなる。
- 補助金の適用余地も検討されている。

## 活用メモ
審査では省力化投資と現場稼働への影響を確認する。
""",
        encoding="utf-8",
    )

    focus = digest.get_latest_lease_news_focus(vault)
    assert focus.available
    assert focus.theme_summary == "国内 / 中"
    assert any("リース期間" in line for line in focus.focus_lines)

    result = digest.write_lease_news_focus_note(date_str="2026-06-11", vault=vault, focus=focus)
    assert result is not None
    assert Path(result.note_path).exists()

    focus_note = vault / "Projects" / "tune_lease_55" / "News" / "2026-06-11_lease-news-focus.md"
    assert focus_note.exists()
    focus_text = focus_note.read_text(encoding="utf-8")
    assert "## 注目論点" in focus_text
    assert "リース期間・中古価値・再リース余地を確認する。" in focus_text

    daily_note = vault / "Daily" / "2026-06-11.md"
    assert daily_note.exists()
    daily_text = daily_note.read_text(encoding="utf-8")
    assert "##" in daily_text
    assert "リースニュースの注目論点" in daily_text

    reflection = digest.write_lease_news_reflection_note(date_str="2026-06-11", vault=vault, focus=focus)
    assert reflection is not None
    reflection_note = vault / "Projects" / "tune_lease_55" / "News" / "2026-06-11_lease-news-reflection.md"
    assert reflection_note.exists()
    reflection_text = reflection_note.read_text(encoding="utf-8")
    assert "## 今日の考え" in reflection_text
    assert "明日見ること" in reflection_text

    parsed = digest.get_latest_lease_news_reflection(vault)
    assert parsed.available
    assert parsed.note_date == "2026-06-11"
    assert parsed.note_path == "Projects/tune_lease_55/News/2026-06-11_lease-news-reflection.md"
    assert parsed.thought_lines
    assert 3 <= len(parsed.thought_lines) <= 4
    assert parsed.tomorrow_lines
    assert parsed.illustration_url == "/lease-grumble/2026-06-11.webp"
