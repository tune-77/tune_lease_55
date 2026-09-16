from scripts import build_shion_memory_index as idx


def test_main_warns_and_exits_nonzero_when_sources_exist_but_index_is_empty(tmp_path, monkeypatch, capsys):
    """MEMORY.md等の記憶ソースは存在するのに、ブレット記法のドリフト等で
    索引レコードが0件になった場合、無条件exit 0のまま無音停止しないことを確認する。"""
    monkeypatch.setattr(idx, "REPO_ROOT", tmp_path)
    # 箇条書きが "- " ではなく "* " に変わり、抽出ロジックが拾えなくなった想定
    (tmp_path / "MEMORY.md").write_text("# Memory\n\n* 長期記憶の項目\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_shion_memory_index.py",
            "--output", str(tmp_path / "out" / "index.json"),
        ],
    )

    exit_code = idx.main()

    assert exit_code == 1
    assert "索引レコードが0件" in capsys.readouterr().err


def test_main_returns_zero_when_no_known_memory_sources_exist(tmp_path, monkeypatch):
    """既知の記憶ソースが1つも無い（真に対象が無い）場合は誤検知せず正常終了する。"""
    monkeypatch.setattr(idx, "REPO_ROOT", tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_shion_memory_index.py",
            "--output", str(tmp_path / "out" / "index.json"),
        ],
    )

    assert idx.main() == 0
