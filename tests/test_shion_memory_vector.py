import json

from api.shion_memory_recall import recall_memories


def test_vector_scores_rescue_zero_keyword_overlap(tmp_path):
    """語彙一致0件でも埋め込み類似が高ければ想起できる（同義語・言い換え対策）。"""
    index = {
        "records": [
            {
                # 「ユンボ」質問に語彙一致しないショベル記憶
                "id": "mem_excavator",
                "content": "油圧ショベルの満了時はアワーメーターと足回りを確認する。",
                "memory_type": "factual_memory",
                "status": "active",
            },
            {
                "id": "mem_unrelated",
                "content": "医療機器は保守期限を確認する。",
                "memory_type": "factual_memory",
                "status": "active",
            },
        ]
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    without_vector = recall_memories("ユンボの返却時チェックは？", index_path=path)
    assert "mem_excavator" not in without_vector["refs"]
    assert without_vector["vector_used"] is False

    with_vector = recall_memories(
        "ユンボの返却時チェックは？",
        index_path=path,
        vector_scores={"mem_excavator": 0.72, "mem_unrelated": 0.31},
    )
    assert with_vector["vector_used"] is True
    assert with_vector["refs"][0] == "mem_excavator"


def test_low_similarity_does_not_add_noise(tmp_path):
    index = {
        "records": [
            {
                "id": "mem_noise",
                "content": "朝のニュース収集の手順メモ。",
                "memory_type": "technical_memory",
                "status": "active",
            }
        ]
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    recalled = recall_memories(
        "この案件の審査は？", index_path=path, vector_scores={"mem_noise": 0.2}
    )

    # 類似度フロア(0.3)未満はスコアに乗らない
    assert recalled["refs"] == []


def test_vector_module_degrades_gracefully_without_chromadb():
    """chromadb 未導入環境では利用不可と判定され、検索は空を返す。"""
    from api import shion_memory_vector

    assert shion_memory_vector.similarity_scores("テスト質問") == {}
    assert shion_memory_vector.is_available() is False


def test_background_sync_not_started_without_chromadb():
    """chromadb が無い環境では自動構築スレッドを起動しない（クライアント不在で早期return）。"""
    from api import shion_memory_vector

    shion_memory_vector.similarity_scores("テスト質問")

    assert shion_memory_vector._background_sync_started is False


def test_hybrid_disabled_by_default(monkeypatch):
    from api import shion_memory_vector

    monkeypatch.delenv("SHION_MEMORY_HYBRID", raising=False)
    assert shion_memory_vector.hybrid_enabled() is False
    monkeypatch.setenv("SHION_MEMORY_HYBRID", "1")
    assert shion_memory_vector.hybrid_enabled() is True
