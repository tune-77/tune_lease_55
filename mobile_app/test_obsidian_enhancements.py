"""
Obsidian RAG 改善モジュールのユニットテスト
"""

import tempfile
import time
from pathlib import Path
from obsidian_bridge_enhancements import (
    extract_frontmatter,
    extract_metadata,
    _file_hash,
    _needs_update,
    BM25Scorer,
    filter_by_industry,
    filter_by_score_range,
    extract_wikilinks,
)


def test_extract_frontmatter():
    """Frontmatter 抽出テスト。"""
    text = """---
title: Test Note
tags: [tag1, tag2]
industry: c 製造業
---
Body content here.
"""
    result = extract_frontmatter(text)
    assert result.get("title") == "Test Note"
    assert result.get("tags") == ["tag1", "tag2"]
    assert result.get("industry") == "c 製造業"


def test_extract_frontmatter_empty():
    """Frontmatter なしテスト。"""
    text = "Just plain text without frontmatter."
    result = extract_frontmatter(text)
    assert result == {}


def test_extract_metadata():
    """メタデータ抽出テスト。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        note_path = vault / "test_note.md"
        content = """---
title: Test Case
tags: [製造, 高スコア]
industry: c 製造業
score_range: [70, 80]
credit_rating: 4-6
---
Test content with #tag1 and [[wikilink]].
"""
        note_path.write_text(content)

        metadata = extract_metadata(note_path, content)
        assert metadata["title"] == "Test Case"
        assert "製造" in metadata["tags"]
        assert metadata["score_range"] == (70, 80)
        assert metadata["credit_rating"] == "4-6"
        assert metadata["has_wikilinks"] is True


def test_file_hash():
    """ファイルハッシュ計算テスト。"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content")
        path = Path(f.name)

    try:
        hash1 = _file_hash(path)
        hash2 = _file_hash(path)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256
    finally:
        path.unlink()


def test_bm25_scorer():
    """BM25 スコアリングテスト。"""
    documents = [
        "製造業のスコアリングガイド",
        "建設業の信用格付判定",
        "製造業 Q-Risk 分析",
    ]

    scorer = BM25Scorer()
    scorer.fit(documents)

    # "製造業" で検索すると、doc[0] と doc[2] が高スコア
    score0 = scorer.score("製造業", documents[0])
    score1 = scorer.score("製造業", documents[1])
    score2 = scorer.score("製造業", documents[2])

    assert score0 > score1
    assert score2 > score1


def test_filter_by_industry():
    """業種フィルタテスト。"""
    notes = [
        {
            "path": "note1.md",
            "metadata": {"industry": "c 製造業"}
        },
        {
            "path": "note2.md",
            "metadata": {"industry": "d 建設業"}
        },
        {
            "path": "note3.md",
            "metadata": {}
        },
    ]

    # 製造業でフィルタ
    filtered = filter_by_industry(notes, "c")
    assert len(filtered) == 2  # note1 と note3（メタデータなし）
    assert filtered[0]["path"] == "note1.md"


def test_filter_by_score_range():
    """スコア範囲フィルタテスト。"""
    notes = [
        {
            "path": "case1.md",
            "metadata": {"score_range": (60, 70)}
        },
        {
            "path": "case2.md",
            "metadata": {"score_range": (70, 80)}
        },
        {
            "path": "case3.md",
            "metadata": {"score_range": (85, 95)}
        },
    ]

    # スコア 65-75 でフィルタ（オーバーラップ検査）
    filtered = filter_by_score_range(notes, 65.0, 75.0)

    # case1 (60-70) と case2 (70-80) がマッチ
    assert len(filtered) == 2
    assert filtered[0]["path"] in ["case1.md", "case2.md"]


def test_extract_wikilinks():
    """Wikilink 抽出テスト。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)

        # リンク先ファイルを作成
        (vault / "target.md").write_text("target content")
        (vault / "subdir").mkdir()
        (vault / "subdir" / "linked.md").write_text("linked content")

        text = """
Some text with [[target]] and [[subdir/linked]].
Also [[nonexistent]] which doesn't exist.
"""

        links = extract_wikilinks(text, vault)
        link_names = [p.name for p in links]

        assert "target.md" in link_names
        assert "linked.md" in link_names
        assert len(links) == 2


if __name__ == "__main__":
    # 簡易テスト実行
    print("Running tests...")

    test_extract_frontmatter()
    print("✓ test_extract_frontmatter")

    test_extract_frontmatter_empty()
    print("✓ test_extract_frontmatter_empty")

    test_extract_metadata()
    print("✓ test_extract_metadata")

    test_file_hash()
    print("✓ test_file_hash")

    test_bm25_scorer()
    print("✓ test_bm25_scorer")

    test_filter_by_industry()
    print("✓ test_filter_by_industry")

    test_filter_by_score_range()
    print("✓ test_filter_by_score_range")

    test_extract_wikilinks()
    print("✓ test_extract_wikilinks")

    print("\n✅ All tests passed!")
