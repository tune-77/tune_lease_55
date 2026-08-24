from api.chat_mid_term_memory import (
    build_chat_mid_term_memory_prompt_block,
    invalidate_chat_mid_term_memory_cache,
    load_chat_mid_term_memory_payload,
)


def _memory_dir(vault):
    path = vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Public" / "Chat Memory"
    path.mkdir(parents=True)
    return path


def test_load_chat_mid_term_memory_payload_reads_layer_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_VAULT_LOCAL_DIR", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    invalidate_chat_mid_term_memory_cache()
    memory_dir = _memory_dir(tmp_path)
    (memory_dir / "mid-term-continuity.md").write_text("mid-term text", encoding="utf-8")

    payload = load_chat_mid_term_memory_payload(str(tmp_path))

    assert "【中期継続メモリ】" in payload["block"]
    assert "mid-term text" in payload["block"]
    assert payload["refs"] == [str(memory_dir / "mid-term-continuity.md")]


def test_load_chat_mid_term_memory_payload_falls_back_to_latest_pack(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_VAULT_LOCAL_DIR", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    invalidate_chat_mid_term_memory_cache()
    memory_dir = _memory_dir(tmp_path)
    (memory_dir / "latest_cloud_chat_memory_pack.md").write_text(
        "## 中期の継続論点\nrecent mid-term from pack",
        encoding="utf-8",
    )

    payload = load_chat_mid_term_memory_payload(str(tmp_path))

    assert "recent mid-term from pack" in payload["block"]
    assert payload["refs"] == [str(memory_dir / "latest_cloud_chat_memory_pack.md")]


def test_build_chat_mid_term_memory_prompt_block_returns_prefixed_block(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_VAULT_LOCAL_DIR", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    invalidate_chat_mid_term_memory_cache()
    memory_dir = _memory_dir(tmp_path)
    (memory_dir / "mid-term-continuity.md").write_text("mid-term text", encoding="utf-8")

    block, payload = build_chat_mid_term_memory_prompt_block(str(tmp_path))

    assert block.startswith("\n\n【中期継続メモリ】")
    assert payload["refs"] == [str(memory_dir / "mid-term-continuity.md")]


def test_load_chat_mid_term_memory_cache_is_scoped_by_vault_path(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_VAULT_LOCAL_DIR", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    invalidate_chat_mid_term_memory_cache()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_memory = _memory_dir(first)
    second_memory = _memory_dir(second)
    (first_memory / "mid-term-continuity.md").write_text("first vault memory", encoding="utf-8")
    (second_memory / "mid-term-continuity.md").write_text("second vault memory", encoding="utf-8")

    first_payload = load_chat_mid_term_memory_payload(str(first))
    second_payload = load_chat_mid_term_memory_payload(str(second))

    assert "first vault memory" in first_payload["block"]
    assert "second vault memory" in second_payload["block"]
    assert "first vault memory" not in second_payload["block"]
