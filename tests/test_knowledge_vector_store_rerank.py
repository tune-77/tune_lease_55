from api.knowledge.vector_store import KnowledgeVectorStore


def _hit(path: str, *, text: str, distance: float = 0.2) -> dict:
    return {
        "text": text,
        "file_name": path.rsplit("/", 1)[-1],
        "file_path": path,
        "section": "",
        "ref": "",
        "wikilinks": "",
        "distance": distance,
        "source": "vector",
    }


def test_rerank_prefers_curated_knowledge_over_news_and_sync_copy():
    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rag-test")
    query = "信用リスク分類と要注意先の判定基準を確認したい"
    hits = [
        _hit(
            "05-クリップ_記事/リースニュース/noise.md",
            text="信用リスク分類 要注意先 判定基準",
            distance=0.1,
        ),
        _hit(
            "/tmp/lease-wiki-vault/99_Synced_From_Origin/03-知識_業界/リース審査実務/信用リスク分類.md",
            text="信用リスク分類 要注意先 判定基準",
            distance=0.1,
        ),
        _hit(
            "03-知識_業界/リース審査実務/信用リスク分類.md",
            text="信用リスク分類 要注意先 判定基準",
            distance=0.2,
        ),
    ]

    ranked = store._rerank_hits(query, hits, top_k=3)

    assert ranked[0]["file_path"] == "03-知識_業界/リース審査実務/信用リスク分類.md"


def test_rerank_deduplicates_chunks_from_the_same_note():
    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rag-test")
    path = "リース知識/補助金・税制優遇とリース.md"
    hits = [
        _hit(path, text="ものづくり補助金はリース取引でも対象になる場合がある", distance=0.1),
        _hit(path, text="公募要領でリース可否を確認する", distance=0.2),
        _hit("03-知識_業界/補助金・融資/補助金_対象要件.md", text="補助金 リース 対象要件", distance=0.3),
    ]

    ranked = store._rerank_hits("ものづくり補助金はリースでも使える？", hits, top_k=3)

    assert [hit["file_path"] for hit in ranked].count(path) == 1


def test_ranking_config_can_be_reloaded_without_restarting(tmp_path):
    config_path = tmp_path / "rag_ranking.json"
    config_path.write_text('{"keyword_pool_multiplier": 4}', encoding="utf-8")
    store = KnowledgeVectorStore(ranking_config={"keyword_pool_multiplier": 2})
    store._ranking_config_path = str(config_path)
    store._ranking_config_mtime = 0.0

    store._maybe_reload_ranking_config()

    assert store._ranking_config["keyword_pool_multiplier"] == 4
