from __future__ import annotations

import importlib
from pathlib import Path


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    return vault


def test_collect_obsidian_context_includes_recent_notes(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    note = vault / "Projects" / "tune_lease_55" / "AI Chat" / "2026-05-16.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# memo\n\n条件付き承認は、追加資料・期間短縮・前受金で進める。",
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    hits = obsidian_bridge.collect_obsidian_context("条件付き承認", limit=3)
    assert hits, "Obsidianの保存メモを拾えていない"
    assert any("条件付き承認" in h["snippet"] for h in hits)


def test_build_prompt_mentions_condition_playbook(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    note = vault / "Projects" / "tune_lease_55" / "AI Chat" / "2026-05-16.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "## memo\n\n承認条件は追加資料と期間短縮。",
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge, chat_assistant
    importlib.reload(obsidian_bridge)
    importlib.reload(chat_assistant)

    prompt = chat_assistant._build_prompt(
        "条件付き承認の具体的な方法を教えて",
        [{"role": "user", "content": "前の相談"}],
        {"score": 66, "judgment": "条件付"},
        obsidian_bridge.collect_obsidian_context("条件付き承認", limit=2),
        [],
        humor_style="standard",
    )
    assert "条件付き承認の説明方針" in prompt
    assert "Obsidianの過去メモ" in prompt
    assert "1. 追加資料 2. 期間短縮" in prompt


def test_web_context_parses_search_results(monkeypatch):
    html = """
    <html><body>
      <a class="result__a" href="https://example.com/alpha">Alpha Result</a>
      <a class="result__snippet">Alpha snippet text.</a>
      <a class="result__a" href="https://example.com/bravo">Bravo Result</a>
      <a class="result__snippet">Bravo snippet text.</a>
    </body></html>
    """

    class Resp:
        status_code = 200
        text = html
        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Resp()

    from mobile_app import web_bridge
    monkeypatch.setattr(web_bridge.requests, "get", fake_get)

    hits = web_bridge.collect_web_context("最新情報", limit=2)
    assert len(hits) == 2
    assert hits[0]["title"] == "Alpha Result"
    assert hits[0]["url"] == "https://example.com/alpha"
    assert "snippet" in hits[0]


def test_build_prompt_mentions_web_section(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge, chat_assistant
    importlib.reload(obsidian_bridge)
    importlib.reload(chat_assistant)

    prompt = chat_assistant._build_prompt(
        "Gemini の最新モデルを教えて",
        [],
        None,
        [],
        [{"title": "Gemini", "url": "https://example.com", "snippet": "info", "domain": "example.com"}],
        humor_style="standard",
    )
    assert "Web参照の方針" in prompt
    assert "Web検索結果" in prompt


def test_append_improvement_note_writes_daily_log(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    result = obsidian_bridge.append_improvement_note(
        "改善候補",
        "## 抽出された改善候補\n\n- **入力導線** [high]\n  - ユーザー要望: Enterで誤送信しないようにしたい\n  - 改善案: チャット欄のみEnter送信にする\n  - 根拠: 送信と審査が競合しやすい",
    )
    assert result["status"] == "saved"
    assert "Improvement Log" in result["path"]
    saved = (vault / result["path"]).read_text(encoding="utf-8")
    assert "入力導線" in saved


def test_append_web_note_writes_daily_log(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    result = obsidian_bridge.append_web_note(
        "Web参照メモ",
        "## Web要点\n\n- Gemini 2.5 Flash は公式ブログで更新\n- 公式情報を優先する",
    )
    assert result["status"] == "saved"
    assert "Web Research" in result["path"]
    saved = (vault / result["path"]).read_text(encoding="utf-8")
    assert "Gemini 2.5 Flash" in saved


def test_build_obsidian_digest_combines_multiple_notes(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    for idx, text in enumerate([
        "## memo\n\n条件付き承認は追加資料と期間短縮が基本。",
        "## memo\n\n条件付き承認は前受金も候補。",
    ], start=1):
        note = vault / "Projects" / "tune_lease_55" / "AI Chat" / f"2026-05-16-{idx}.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(text, encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    hits = obsidian_bridge.collect_obsidian_context("条件付き承認", limit=4)
    digest = obsidian_bridge.build_obsidian_digest("条件付き承認", hits)
    assert "Obsidian統合要約" in digest["digest"]
    assert "関連ノート数" in digest["digest"]
    assert "追加資料" in digest["digest"] or "前受金" in digest["digest"]
    assert "[[" in digest["digest"]


def test_append_wiki_note_writes_hub(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    result = obsidian_bridge.append_wiki_note(
        "条件付き承認",
        "## 要点\n\n- 追加資料\n- 期間短縮\n- 前受金",
        related_paths=[
            "Projects/tune_lease_55/AI Chat/2026-05-16.md",
            "Projects/tune_lease_55/2026-05-12_ファイナンスリース_autoresearch.md",
        ],
    )
    assert result["status"] == "saved"
    saved = (vault / result["path"]).read_text(encoding="utf-8")
    assert "関連ノート" in saved
    assert "[[Projects/tune_lease_55/AI Chat/2026-05-16|2026-05-16]]" in saved
    assert "条件付き承認" in saved
