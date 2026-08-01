from api.chat_user_personal_memory import (
    build_user_personal_memory_prompt_block,
    invalidate_user_personal_memory_cache,
    load_user_personal_memory_payload,
    read_cloudrun_personal_memory_lines,
    read_personal_memory_lines,
)


def test_read_personal_memory_lines_filters_keywords(tmp_path):
    path = tmp_path / "memory.md"
    path.write_text(
        "\n".join([
            "random note",
            "犬の名前はタム",
            "好み: 短く答える",
            "x" * 901,
        ]),
        encoding="utf-8",
    )

    lines, ref = read_personal_memory_lines(path, limit=10)

    assert lines == ["犬の名前はタム", "好み: 短く答える"]
    assert ref == str(path)


def test_read_personal_memory_lines_all_lines_keeps_non_keywords(tmp_path):
    path = tmp_path / "memory.md"
    path.write_text("random note\n犬の名前はタム", encoding="utf-8")

    lines, _ = read_personal_memory_lines(path, limit=10, all_lines=True)

    assert lines == ["random note", "犬の名前はタム"]


def test_read_cloudrun_personal_memory_lines_uses_injected_reader(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "lease-chat")

    def reader(*, days):
        assert days == 45
        return [
            {
                "event_type": "chat_exchange",
                "surface": "chat",
                "ts": "2026-08-01T00:00:00Z",
                "payload": {"user_message": "覚えて。犬の名前はタムです"},
            },
            {
                "event_type": "personal_memory",
                "payload": {"dog_name": "タム", "derived_lines": ["- [confirmed] Dog name: タム"]},
            },
        ]

    lines, ref = read_cloudrun_personal_memory_lines(limit=5, recent_events_reader=reader)

    assert lines[0] == "- [confirmed] Dog name: タム"
    assert len(lines) == len(set(lines))
    assert ref == "gcs://cloudrun-inputs/personal-memory"


def test_load_user_personal_memory_payload_dedupes_and_prioritizes(tmp_path, monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("CLOUDRUN_PENDING_GCS_ENABLED", raising=False)
    invalidate_user_personal_memory_cache()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    memory_path = data_dir / "user_personal_memory.md"
    memory_path.write_text(
        "\n".join([
            "[candidate] 好み: 詳細より結論を先に",
            "[confirmed] Dog name: タム",
            "[confirmed] Dog name: タム",
        ]),
        encoding="utf-8",
    )

    payload = load_user_personal_memory_payload(
        repo_root=tmp_path,
        data_path_resolver=lambda name: str(data_dir / name),
    )

    assert payload["line_count"] == 2
    assert payload["block"].splitlines()[8] == "- [confirmed] Dog name: タム"
    assert payload["refs"] == [str(memory_path)]


def test_build_user_personal_memory_prompt_block_returns_prefixed_block(tmp_path, monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("CLOUDRUN_PENDING_GCS_ENABLED", raising=False)
    invalidate_user_personal_memory_cache()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "user_personal_memory.md").write_text("[confirmed] Dog name: タム", encoding="utf-8")

    block, payload = build_user_personal_memory_prompt_block(
        repo_root=tmp_path,
        data_path_resolver=lambda name: str(data_dir / name),
    )

    assert block.startswith("\n\n【ユーザー個人記憶】")
    assert payload["line_count"] == 1
