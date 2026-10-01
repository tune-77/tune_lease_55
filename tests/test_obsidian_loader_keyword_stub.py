from api.knowledge.obsidian_loader import _chunk_by_h2, is_keyword_stub, normalize_chunk_text


def test_is_keyword_stub_detects_space_separated_query_lines():
    assert is_keyword_stub("補助金不採択時の返済余力確認 判断資産 確認質問 条件付き承認 反証 リスク 兆候 再利用")
    assert is_keyword_stub("食品製造業の冷凍冷蔵設備 canonical_topic 重複 不足 更新トリガー 審査論点 体系化")


def test_is_keyword_stub_keeps_sentences_lists_and_short_titles():
    assert not is_keyword_stub("境界スコアを、追加確認と条件設定で承認側へ寄せられるか")
    assert not is_keyword_stub("農業機械リースの季節性と補助金依存リスク")
    assert not is_keyword_stub("- 追加資料\n- 期間短縮\n- 前受金\n- 保証担保")
    assert not is_keyword_stub("スコアリングドリフト兆候: 60-80帯 win_pct(57.9%) < 40-60帯(62.2%)")
    assert not is_keyword_stub("違う　中古車だから　車検は2年　2年　2年の合計6年だよ")


def test_chunk_by_h2_skips_query_section():
    body = (
        "## Query\n補助金不採択時の返済余力確認 判断資産 確認質問 条件付き承認 反証 リスク\n\n"
        "## Answer Summary\n補助金は後払いが多く、短期資金繰りの改善とは限らない。\n"
    )
    chunks = _chunk_by_h2(body, "/v/a.md", "a.md", {}, 0.0)
    assert [chunk.section for chunk in chunks] == ["Answer Summary"]


def test_normalize_chunk_text_ignores_bullet_and_space_variation():
    assert normalize_chunk_text("- 直近の  ニュース\n") == normalize_chunk_text("直近の ニュース")


def test_chunk_by_h2_skips_vertex_note_metadata_sections():
    body = "## Topic\n残価リスク\n\n## Mode\n知識棚卸し (`knowledge_audit`)\n\n## Answer Summary\n中古市場の厚みで残価を見る。\n"
    vertex = _chunk_by_h2(body, "/v/a.md", "a.md", {"source": "vertex_ai_search_workflow"}, 0.0)
    other = _chunk_by_h2(body, "/v/b.md", "b.md", {}, 0.0)
    assert [chunk.section for chunk in vertex] == ["Answer Summary"]
    assert [chunk.section for chunk in other] == ["Topic", "Mode", "Answer Summary"]


def test_normalize_chunk_text_treats_number_only_differences_as_same():
    assert normalize_chunk_text("60-80帯 win_pct(58.5%) < 40-60帯(62.4%)") == normalize_chunk_text(
        "60-80帯 win_pct(58.7%) < 40-60帯(62.7%)"
    )
