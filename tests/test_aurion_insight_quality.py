"""@AI_Insight_Evolved の中身の修正（REV-492）。

- 鍵: launchd の plist には鍵が無いため、環境変数だけを見ると毎回フォールバック定型文になっていた
- Web 抜粋: 生HTMLの断片（<!DOCTYPE…、Login </a>…）ではなく本文の段落を出す
- 結論が前回と同じ日は「変化なし」と短く書く
"""
from pathlib import Path

import pytest

from scripts import aurion_core_daily as acd

_HTML = """<!DOCTYPE html><html><head><title>t</title><script>var x = "risk";</script>
<style>.risk{color:red}</style></head><body>
<nav><a href="/login">Login</a> <p>Menu credit risk link that should be ignored entirely here ok</p></nav>
<p>Short risk.</p>
<p>Equipment finance demand is supported by replacement investment, while policy uncertainty and credit risk remain material for lessors.</p>
<p>Second paragraph about SME borrowing costs that remain high relative to pre-pandemic levels in many markets.</p>
<footer><p>Contact us about credit risk and leasing at any time of the day please.</p></footer>
</body></html>"""


# ── 鍵 ──────────────────────────────────────────────────────────────
def test_key_prefers_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert acd._aurion_gemini_key() == "env-key"


def test_key_falls_back_to_secret_manager(monkeypatch):
    import secret_manager

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(secret_manager, "get_gemini_api_key", lambda: " toml-key ")

    assert acd._aurion_gemini_key() == "toml-key"


def test_missing_key_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(acd, "_aurion_gemini_key", lambda: "")

    result = acd._generate_conclusions_with_llm({}, {}, {}, False, {}, [])

    assert result is None
    assert "APIキーが見つからない" in capsys.readouterr().out


def test_reasoning_records_source(monkeypatch):
    monkeypatch.setenv("AURION_HOLD_SECONDS", "0")
    monkeypatch.setattr(acd, "_generate_conclusions_with_llm", lambda **_: ["a", "b", "c"])
    assert acd.cross_reasoning_loop({}, {}, {"findings": []})["conclusions_source"] == "gemini"

    monkeypatch.setattr(acd, "_generate_conclusions_with_llm", lambda **_: None)
    assert acd.cross_reasoning_loop({}, {}, {"findings": []})["conclusions_source"] == "fallback"


# ── Web 抜粋 ─────────────────────────────────────────────────────────
def test_html_to_text_keeps_body_paragraphs_only():
    paragraphs = acd._html_to_text(_HTML)

    joined = " ".join(paragraphs)
    assert "Equipment finance demand" in joined
    assert "<" not in joined and "Login" not in joined and "var x" not in joined
    assert "Menu credit" not in joined and "Contact us" not in joined
    assert "Short risk." not in joined


def test_excerpt_starts_at_matching_paragraph_and_respects_limit():
    excerpt = acd._excerpt_from_paragraphs(acd._html_to_text(_HTML), ["SME", "risk"], limit=200)

    assert excerpt.startswith("Second paragraph about SME")
    assert len(excerpt) <= 200


def test_excerpt_empty_when_no_keyword():
    assert acd._excerpt_from_paragraphs(["nothing relevant in this long paragraph at all, really none"], ["SME"]) == ""


# ── 変化なし ─────────────────────────────────────────────────────────
def _write_note(directory: Path, date: str, items: list[str]) -> None:
    body = "\n".join(f"- {i}" for i in items)
    (directory / f"@AI_Insight_Evolved_{date}.md").write_text(
        f"# x\n\n## 3. 結論\n\n- 生成元: fallback\n{body}\n\n## 4. 設計レビュー\n", encoding="utf-8"
    )


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setattr(acd, "LEASE_VAULT", tmp_path)
    monkeypatch.setattr(acd, "date_str", lambda: "2026-10-08")
    return tmp_path


def _write_today(conclusions: list[str], source: str = "fallback") -> str:
    reasoning = {"conclusions": conclusions, "conclusions_source": source, "steps": []}
    path = acd.write_evolved_insight({}, {}, {}, {"findings": []}, reasoning, Path("@AI_Daily_Report_2026-10-08_0600.md"))
    return path.read_text(encoding="utf-8")


def test_same_conclusions_as_previous_note_are_shortened(vault):
    _write_note(vault, "2026-10-07", ["結論A", "結論B"])

    text = _write_today(["結論A", "結論B"])

    assert "- 変化なし（前回の結論と同じ）" in text
    assert "- 結論A" not in text
    assert "- 生成元: fallback" in text


def test_new_conclusions_are_written_in_full(vault):
    _write_note(vault, "2026-10-07", ["結論A", "結論B"])

    text = _write_today(["新しい結論X", "新しい結論Y", "新しい結論Z"], source="gemini")

    assert "- 新しい結論X" in text and "変化なし" not in text
    assert "- 生成元: gemini" in text


def test_comparison_skips_previous_unchanged_day(vault):
    _write_note(vault, "2026-10-06", ["結論A", "結論B"])
    (vault / "@AI_Insight_Evolved_2026-10-07.md").write_text(
        "## 3. 結論\n\n- 生成元: fallback\n- 変化なし（前回の結論と同じ）\n", encoding="utf-8"
    )

    assert "- 変化なし（前回の結論と同じ）" in _write_today(["結論A", "結論B"])
