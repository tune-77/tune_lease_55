from pathlib import Path

from scripts.obsidian_note_curator import build_report, detect_new_or_changed_files, load_state


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_generate(text: str):
    def _inner(prompt, **kwargs):
        return {"used": True, "status": "ok", "text": text, "error": ""}

    return _inner


def test_apply_writes_tags_and_moves_file(tmp_path):
    vault = tmp_path / "vault"
    backup_dir = tmp_path / "backup"
    _write(vault / "Inbox" / "new_note.md", "人工知能と機械学習の最新動向について書いたメモ。")

    report = build_report(
        vault,
        state_path=tmp_path / "state.json",
        backup_dir=backup_dir,
        apply=True,
        max_files=20,
        generate_fn=_fake_generate("TAGS: AI, テスト\nFOLDER: Research/AI"),
    )

    assert report["candidate_count"] == 1
    entry = report["entries"][0]
    assert entry["applied"] is True
    assert entry["moved"] is True
    new_path = vault / "Research" / "AI" / "new_note.md"
    assert new_path.exists()
    assert "tags:" in new_path.read_text(encoding="utf-8")
    assert backup_dir.exists()


def test_dry_run_does_not_persist_state(tmp_path):
    vault = tmp_path / "vault"
    _write(vault / "note.md", "本文")
    state_path = tmp_path / "state.json"

    build_report(
        vault,
        state_path=state_path,
        backup_dir=tmp_path / "backup",
        apply=False,
        max_files=20,
        generate_fn=_fake_generate("TAGS: x\nFOLDER: y"),
    )

    assert not state_path.exists()
    state = load_state(state_path)
    assert detect_new_or_changed_files(vault, state)


def test_second_run_skips_already_seen_file(tmp_path):
    vault = tmp_path / "vault"
    _write(vault / "note.md", "本文")

    generate_fn = _fake_generate("TAGS: x\nFOLDER: y")
    state_path = tmp_path / "state.json"
    backup_dir = tmp_path / "backup"
    first = build_report(vault, state_path=state_path, backup_dir=backup_dir, apply=True, max_files=20, generate_fn=generate_fn)
    assert first["candidate_count"] == 1

    second = build_report(vault, state_path=state_path, backup_dir=backup_dir, apply=True, max_files=20, generate_fn=generate_fn)
    assert second["candidate_count"] == 0


def test_unparseable_llm_response_is_not_applied(tmp_path):
    vault = tmp_path / "vault"
    _write(vault / "note.md", "本文")

    report = build_report(
        vault,
        state_path=tmp_path / "state.json",
        backup_dir=tmp_path / "backup",
        apply=True,
        max_files=20,
        generate_fn=_fake_generate("すみません、わかりません"),
    )

    entry = report["entries"][0]
    assert entry["applied"] is False
