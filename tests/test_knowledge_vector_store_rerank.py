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


def test_query_terms_adds_domain_subterms_for_compound_japanese_queries():
    terms = KnowledgeVectorStore._query_terms(
        "中古工作機械の主軸稼働時間、制御装置、搬出費を残価評価したい"
    )

    assert "工作機械" in terms
    assert "主軸" in terms
    assert "稼働時間" in terms
    assert "制御装置" in terms
    assert "搬出費" in terms
    assert "残価" in terms


def test_query_terms_adds_construction_asset_subterms():
    terms = KnowledgeVectorStore._query_terms(
        "油圧ショベルPC200のアワーメーターと油圧系から再販リスクを判断したい"
    )

    assert "油圧ショベル" in terms
    assert "pc200" in terms
    assert "アワーメーター" in terms
    assert "油圧系" in terms


def test_rerank_prefers_matching_construction_asset_note():
    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rag-test")
    query = "油圧ショベルPC200のアワーメーターと油圧系から再販リスクを判断したい"
    hits = [
        _hit(
            "Projects/tune_lease_55/Asset Knowledge/高所作業車/高所作業車 点検記録・安全装置・再販リスク.md",
            text="再販リスク アワーメーター 油圧系",
            distance=None,
        )
        | {"score": 16},
        _hit(
            "Projects/tune_lease_55/Asset Knowledge/建機/コマツ PC200 油圧ショベル 残価・再販リスク.md",
            text="PC200 油圧ショベル アワーメーター 油圧系 再販リスク",
            distance=None,
        )
        | {"score": 16},
    ]

    ranked = store._rerank_hits(query, hits, top_k=2)

    assert ranked[0]["file_path"].startswith("Projects/tune_lease_55/Asset Knowledge/建機/")


def test_rerank_pushes_chat_logs_below_curated_knowledge_for_domain_queries():
    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rag-test")
    query = "競合の低い料率に対して営業が説明すべき条件差と見積比較の観点は？"
    hits = [
        _hit(
            "Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-08.md",
            text="競合 低い料率 営業 説明 条件差 見積比較 観点",
            distance=0.1,
        ),
        _hit(
            "Projects/tune_lease_55/Research/Vertex Distilled/2026-08-08_knowledge_audit_金利・料率・競合条件の組み立て_0cfbafc1d1e41613.md",
            text="競合条件 料率 金利 見積 比較 営業説明",
            distance=0.2,
        ),
    ]

    ranked = store._rerank_hits(query, hits, top_k=2)

    assert ranked[0]["file_path"].startswith("Projects/tune_lease_55/Research/Vertex Distilled/")


def test_source_priority_checks_low_priority_paths_before_generic_project_boost():
    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rag-test")

    chat_priority = store._source_priority(
        _hit("Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-08.md", text="")
    )
    research_priority = store._source_priority(
        _hit("Projects/tune_lease_55/Research/Vertex Distilled/rate.md", text="")
    )

    assert chat_priority < research_priority


def test_keyword_search_uses_short_domain_terms_as_candidate_terms():
    class FakeCollection:
        def count(self):
            return 1

        def get(self, include):
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["rate-note"],
                "documents": ["金利・料率・競合条件の組み立て"],
                "metadatas": [
                    {
                        "file_name": "金利・料率・競合条件の組み立て.md",
                        "file_path": "Projects/tune_lease_55/Research/Vertex Distilled/金利・料率・競合条件の組み立て.md",
                        "section": "概要",
                        "obsidian_ref": "[[金利・料率・競合条件の組み立て#概要]]",
                        "wikilinks": "",
                    }
                ],
            }

    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rag-test")
    store._collection = FakeCollection()

    results = store._keyword_search(
        "競合の低い料率に対して営業が説明すべき条件差と見積比較の観点は？",
        top_k=5,
    )

    assert results
    assert results[0]["doc_id"] == "rate-note"


def test_rerank_drops_negative_low_priority_noise():
    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rag-test")
    query = "売掛金や棚卸資産の増加から粉飾決算を見抜く確認方法は？"
    hits = [
        _hit(
            "Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md",
            text="売掛金 棚卸資産 粉飾決算",
            distance=0.8,
        ),
        _hit(
            "03-知識_業界/リース審査実務/業種別審査チェックリスト_追補.md",
            text="売掛金 棚卸資産 増加 確認",
            distance=0.3,
        ),
        _hit("Projects/tune_lease_55/Research/Auto Research/2026-06-06_cash-flow.md", text="売掛金 確認", distance=0.4),
        _hit("Projects/tune_lease_55/Research/Auto Research/2026-06-20_cash-flow.md", text="棚卸資産 確認", distance=0.4),
        _hit("Projects/tune_lease_55/Research/Auto Research/2026-07-03_industry-risk.md", text="粉飾 決算 確認", distance=0.4),
        _hit("Projects/tune_lease_55/Research/Auto Research/2026-07-06_cash-flow.md", text="資金繰り 確認", distance=0.4),
    ]

    ranked = store._rerank_hits(query, hits, top_k=5)

    assert ranked
    assert all("AI Chat/" not in hit["file_path"] for hit in ranked)
