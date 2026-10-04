from pathlib import Path

from scripts.obsidian_note_curator import _parse_proposal, build_report, detect_new_or_changed_files, load_state


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


def test_unreadable_icloud_file_is_left_untouched(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    note = vault / "00-MOC" / "MOC.md"
    _write(note, "大事な本文")
    original_read_text = Path.read_text

    def _read_text(self, *args, **kwargs):
        if self == note:
            raise OSError(11, "Resource deadlock avoided")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_text)
    calls = []

    def _generate(prompt, **kwargs):
        calls.append(prompt)
        return {"used": True, "status": "ok", "text": "TAGS: x\nFOLDER: moved", "error": ""}

    state_path = tmp_path / "state.json"
    report = build_report(vault, state_path=state_path, backup_dir=tmp_path / "backup", apply=True, max_files=20, generate_fn=_generate)
    monkeypatch.undo()

    assert report["entries"][0]["proposal_status"] == "unreadable"
    assert report["entries"][0]["applied"] is False
    assert note.read_text(encoding="utf-8") == "大事な本文"
    assert not (vault / "moved").exists()
    assert calls == []
    assert "00-MOC/MOC.md" not in load_state(state_path).get("seen_files", {})


def test_empty_file_is_skipped(tmp_path):
    vault = tmp_path / "vault"
    _write(vault / "note.md", "  \n")

    report = build_report(
        vault,
        state_path=tmp_path / "state.json",
        backup_dir=tmp_path / "backup",
        apply=True,
        max_files=20,
        generate_fn=_fake_generate("TAGS: x\nFOLDER: y"),
    )

    assert report["entries"][0]["proposal_status"] == "empty"
    assert (vault / "note.md").exists()
    assert not (vault / "y").exists()


def test_parse_proposal_tags_and_folder_on_one_line():
    proposal = _parse_proposal("TAGS: a, b FOLDER: x")
    assert proposal["tags"] == ["a", "b"]
    assert proposal["folder"] == "x"
