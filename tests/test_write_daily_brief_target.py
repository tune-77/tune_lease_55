"""REV-585: DAILY-BRIEF はルートだけ更新し、旧コピーはRAGから除外する。"""
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
    retired = old_copy.read_text(encoding="utf-8")
    assert retired.endswith("old")  # 既存の内容は消さない
    assert "rag_exclude: true" in retired


def test_retire_legacy_daily_brief_is_idempotent(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    wiki = root / "lease-wiki-vault"
    wiki.mkdir(parents=True)
    old_copy = wiki / "DAILY-BRIEF.md"
    old_copy.write_text("---\ntags: [daily]\n---\n# old\n", encoding="utf-8")
    monkeypatch.setattr(wdb, "ICLOUD_VAULT_PATH", wiki)
    monkeypatch.setattr(wdb, "OUTPUT_PATH", root / "DAILY-BRIEF.md")

    assert wdb.retire_legacy_daily_brief() is True
    once = old_copy.read_text(encoding="utf-8")
    assert wdb.retire_legacy_daily_brief() is False
    assert old_copy.read_text(encoding="utf-8") == once
    assert once.count("rag_exclude: true") == 1


def test_retire_legacy_daily_brief_ignores_body_marker(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    wiki = root / "lease-wiki-vault"
    wiki.mkdir(parents=True)
    old_copy = wiki / "DAILY-BRIEF.md"
    old_copy.write_text(
        "---\ntags: [daily]\n---\n# old\nrag_exclude: true\n", encoding="utf-8"
    )
    monkeypatch.setattr(wdb, "ICLOUD_VAULT_PATH", wiki)
    monkeypatch.setattr(wdb, "OUTPUT_PATH", root / "DAILY-BRIEF.md")

    assert wdb.retire_legacy_daily_brief() is True
    frontmatter = old_copy.read_text(encoding="utf-8").split("\n---", 1)[0]
    assert "rag_exclude: true" in frontmatter


def test_failed_root_write_does_not_retire_legacy_copy(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    wiki = root / "lease-wiki-vault"
    wiki.mkdir(parents=True)
    output = root / "DAILY-BRIEF.md"
    old_copy = wiki / "DAILY-BRIEF.md"
    old_copy.write_text("old", encoding="utf-8")
    original_write_text = wdb.Path.write_text

    def fail_root_write(path, *args, **kwargs):
        if path == output:
            raise OSError("busy")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(wdb, "VAULT_PATH", root)
    monkeypatch.setattr(wdb, "ICLOUD_MAIN_VAULT_PATH", root)
    monkeypatch.setattr(wdb, "ICLOUD_VAULT_PATH", wiki)
    monkeypatch.setattr(wdb, "OUTPUT_PATH", output)
    monkeypatch.setattr(wdb, "LATEST_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(wdb, "MACRO_JSON", tmp_path / "missing.json")
    monkeypatch.setattr(wdb, "SIDECAR_BRIEF_MD", tmp_path / "missing.md")
    monkeypatch.setattr(wdb.Path, "write_text", fail_root_write)
    monkeypatch.setattr(wdb, "report_pipeline_failure", lambda *args, **kwargs: None)

    wdb.main()

    assert old_copy.read_text(encoding="utf-8") == "old"
