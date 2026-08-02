from api.chat_identity_memory import (
    build_chat_identity_memory_prompt_block,
    extract_markdown_section,
    invalidate_chat_identity_memory_cache,
    load_chat_identity_memory_payload,
    read_chat_memory_file,
)


def _memory_dir(vault):
    path = vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Public" / "Chat Memory"
    path.mkdir(parents=True)
    return path


def test_read_chat_memory_file_clips_long_text(tmp_path):
    path = tmp_path / "memory.md"
    path.write_text("abcdef", encoding="utf-8")

    assert read_chat_memory_file(path, limit=3) == "abc\n..."
    assert read_chat_memory_file(tmp_path / "missing.md") == ""


def test_extract_markdown_section_reads_single_h2_section():
    markdown = "# Title\n\n## 長期記憶\nalpha\n\n## 継続中の方針\nbeta"

    assert extract_markdown_section(markdown, "長期記憶") == "alpha"
    assert extract_markdown_section(markdown, "missing") == ""


def test_load_chat_identity_memory_payload_reads_layer_files(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_VAULT_LOCAL_DIR", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    invalidate_chat_identity_memory_cache()
    memory_dir = _memory_dir(tmp_path)
    (memory_dir / "identity.md").write_text("identity text", encoding="utf-8")
    (memory_dir / "judgment-principles.md").write_text("judgment text", encoding="utf-8")
    (memory_dir / "recent-continuity.md").write_text("recent text", encoding="utf-8")

    payload = load_chat_identity_memory_payload(str(tmp_path))

    assert "【紫苑同一性メモリ】" in payload["block"]
    assert "### Core Identity Memory" in payload["block"]
    assert payload["layers"] == {"identity": True, "judgment": True, "recent": True}
    assert len(payload["refs"]) == 3


def test_load_chat_identity_memory_payload_falls_back_to_latest_pack(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_VAULT_LOCAL_DIR", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    invalidate_chat_identity_memory_cache()
    memory_dir = _memory_dir(tmp_path)
    (memory_dir / "latest_cloud_chat_memory_pack.md").write_text(
        "## 長期記憶\nidentity from pack\n\n## 継続中の方針\nrecent from pack",
        encoding="utf-8",
    )

    payload = load_chat_identity_memory_payload(str(tmp_path))

    assert "identity from pack" in payload["block"]
    assert "recent from pack" in payload["block"]
    assert payload["layers"]["identity"] is True
    assert payload["layers"]["recent"] is True


def test_build_chat_identity_memory_prompt_block_returns_prefixed_block(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_VAULT_LOCAL_DIR", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    invalidate_chat_identity_memory_cache()
    memory_dir = _memory_dir(tmp_path)
    (memory_dir / "identity.md").write_text("identity text", encoding="utf-8")

    block, payload = build_chat_identity_memory_prompt_block(str(tmp_path))

    assert block.startswith("\n\n【紫苑同一性メモリ】")
    assert payload["layers"]["identity"] is True
