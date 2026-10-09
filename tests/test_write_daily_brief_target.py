"""REV-585: DAILY-BRIEF はメインVaultのルート1か所だけに書く（lease-wiki-vault には書かない）。"""
from __future__ import annotations

import scripts.write_daily_brief as wdb


def test_daily_brief_written_only_to_vault_root(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    wiki = root / "lease-wiki-vault"
    wiki.mkdir(parents=True)
    old_copy = wiki / "DAILY-BRIEF.md"
    old_copy.write_text("old", encoding="utf-8")
    monkeypatch.setattr(wdb, "VAULT_PATH", root)
    monkeypatch.setattr(wdb, "ICLOUD_MAIN_VAULT_PATH", root)
    monkeypatch.setattr(wdb, "ICLOUD_VAULT_PATH", wiki)
    monkeypatch.setattr(wdb, "OUTPUT_PATH", root / "DAILY-BRIEF.md")
    monkeypatch.setattr(wdb, "LATEST_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(wdb, "MACRO_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(wdb, "SIDECAR_BRIEF_MD", tmp_path / "missing.md")

    wdb.main()

    assert (root / "DAILY-BRIEF.md").read_text(encoding="utf-8").startswith("# DAILY-BRIEF")
    assert old_copy.read_text(encoding="utf-8") == "old"  # 既存の複製は消さず、上書きもしない
