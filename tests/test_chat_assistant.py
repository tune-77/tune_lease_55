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
        humor_style="standard",
    )
    assert "条件付き承認の説明方針" in prompt
    assert "Obsidianの過去メモ" in prompt
    assert "1. 追加資料 2. 期間短縮" in prompt


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
