from pathlib import Path

from obsidian_humor import build_humor_prompt_addon


def test_build_humor_prompt_addon_reads_relevant_notes(tmp_path: Path):
    humor_dir = tmp_path / "Humor"
    humor_dir.mkdir()
    (humor_dir / "口調ルール.md").write_text("- ユーモアは最後に1文だけ添える。\n", encoding="utf-8")
    (humor_dir / "NG表現.md").write_text("- 顧客を馬鹿にしない。\n", encoding="utf-8")
    (humor_dir / "リスク別ユーモア.md").write_text(
        "## 中リスク\n\n- 数字は少し背伸びしています。\n",
        encoding="utf-8",
    )
    (humor_dir / "業種別ユーモア.md").write_text(
        "## 建設業\n\n- 財務の基礎工事を確認します。\n",
        encoding="utf-8",
    )
    (humor_dir / "物件別ユーモア.md").write_text(
        "## 建設機械\n\n- 現場では頼れる機械です。\n",
        encoding="utf-8",
    )

    addon = build_humor_prompt_addon(
        {"score": 62, "industry_sub": "D 建設業", "asset_name": "建設機械"},
        vault=tmp_path,
    )

    assert "Obsidianユーモア編集室" in addon
    assert "中リスク" in addon
    assert "基礎工事" in addon
    assert "頼れる機械" in addon


def test_build_humor_prompt_addon_returns_empty_for_missing_vault(tmp_path: Path):
    addon = build_humor_prompt_addon({"score": 80}, vault=tmp_path / "missing")
    assert addon == ""
