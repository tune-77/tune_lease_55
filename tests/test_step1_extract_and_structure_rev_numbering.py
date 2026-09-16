from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = ROOT / ".agents" / "skills" / "auto-improvement-pipeline" / "scripts" / "step1_extract_and_structure.py"
_spec = importlib.util.spec_from_file_location("step1_extract_and_structure", _SCRIPT)
step1_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(step1_mod)


def test_max_ledger_rev_number_reads_existing_ledger(tmp_path, monkeypatch):
    (tmp_path / "CLAUDE.md").write_text("", encoding="utf-8")
    ledger_dir = tmp_path / "scripts"
    ledger_dir.mkdir()
    (ledger_dir / "improvement_ledger.jsonl").write_text(
        '{"rev_id": "REV-233", "status": "applied"}\n'
        '{"rev_id": "REV-237", "status": "applied"}\n'
        '{"key": "misc_abc", "status": "needs_review"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(step1_mod, "_find_repo_root", lambda: tmp_path)

    assert step1_mod._max_ledger_rev_number() == 237


def test_max_ledger_rev_number_zero_when_ledger_missing(tmp_path, monkeypatch):
    (tmp_path / "CLAUDE.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(step1_mod, "_find_repo_root", lambda: tmp_path)

    assert step1_mod._max_ledger_rev_number() == 0


def test_extract_improvements_numbers_after_ledger_max(monkeypatch):
    """REV-019 二重採番の回帰防止: 抽出ID は毎回1からではなく、
    台帳の最大REV番号の続きから振られる。"""
    monkeypatch.setattr(step1_mod, "_max_ledger_rev_number", lambda: 237)
    monkeypatch.setattr(step1_mod, "_existing_canonical_keys", lambda: set())

    chat_log = "[改善] ダミーの改善内容がここに入ります。十分な長さのテキスト。"
    improvements = step1_mod.extract_improvements_from_chat_log(chat_log)

    assert len(improvements) == 1
    assert improvements[0]["id"] == "REV-238"


def test_extract_improvements_skips_idea_already_in_ledger(monkeypatch):
    """同一の改善アイデアが台帳に既に存在する場合、新規REVを発行しない
    （scripts/README_ledger.md 記載の REV-230/237・REV-292 重複発行の再発防止）。"""
    monkeypatch.setattr(step1_mod, "_max_ledger_rev_number", lambda: 237)
    text = "回答が途中で切れる問題を修正する。ユーザーからの指摘が複数件ある。"
    existing_key = step1_mod._canonical_key_for(text[:50], text)
    monkeypatch.setattr(step1_mod, "_existing_canonical_keys", lambda: {existing_key})

    chat_log = f"[改善] {text}"
    improvements = step1_mod.extract_improvements_from_chat_log(chat_log)

    assert improvements == []


def test_extract_improvements_dedupes_same_idea_within_one_chat_log(monkeypatch):
    """同一チャットログ内で同じアイデアが複数回言及されても1件しか採番しない。"""
    monkeypatch.setattr(step1_mod, "_max_ledger_rev_number", lambda: 237)
    monkeypatch.setattr(step1_mod, "_existing_canonical_keys", lambda: set())

    text = "回答が途中で切れる問題を修正する。ユーザーからの指摘が複数件ある。"
    chat_log = f"[改善] {text}\n\n[改善] {text}"
    improvements = step1_mod.extract_improvements_from_chat_log(chat_log)

    assert len(improvements) == 1
