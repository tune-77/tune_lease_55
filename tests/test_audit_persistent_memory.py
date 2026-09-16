import json

from scripts import audit_persistent_memory as audit


def test_extract_bullets_finds_dash_prefixed_items():
    text = "\n".join(
        [
            "# Persistent Memory",
            "",
            "## Rules",
            "",
            "- 永続記憶は頻繁に更新しない。",
            "- 個人情報や一時的な好みはここへ置かない。",
        ]
    )

    bullets = audit.extract_bullets(text)

    assert bullets == [
        "永続記憶は頻繁に更新しない。",
        "個人情報や一時的な好みはここへ置かない。",
    ]


def test_main_fails_when_substantial_file_has_no_bullets(tmp_path, capsys):
    """書式が箇条書き(- )から離れると抽出が0件になる。内容量があるのに0件はドリフト。"""
    persistent = tmp_path / "PERSISTENT_MEMORY.md"
    persistent.write_text(
        "# Persistent Memory\n\n"
        + ("* 箇条書き記号を変えてしまった例。テキストはそれなりの分量がある。\n" * 10),
        encoding="utf-8",
    )
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"

    exit_code = _run_main(audit, persistent, output_json, output_md)

    assert exit_code == 1
    assert "書式が変わった可能性" in capsys.readouterr().err
    assert not output_json.exists()


def test_main_is_ok_when_file_is_short_or_missing(tmp_path, capsys):
    """空/短いファイル（真の意味で対象が無い）は誤検知しない。"""
    persistent = tmp_path / "PERSISTENT_MEMORY.md"
    persistent.write_text("# Persistent Memory\n", encoding="utf-8")
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"

    exit_code = _run_main(audit, persistent, output_json, output_md)

    assert exit_code == 0
    assert output_json.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["summary"]["bullets"] == 0


def _run_main(audit_module, persistent, output_json, output_md):
    import sys

    argv = [
        "audit_persistent_memory.py",
        "--persistent", str(persistent),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        return audit_module.main()
    finally:
        sys.argv = old_argv
