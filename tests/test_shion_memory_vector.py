import json

import numpy as np
import pytest

from api.shion_memory_recall import recall_memories
from scripts.revise_shion_memory import apply_revisions


class _FakeCollection:
    """埋め込み同期済みのIDだけを持つ擬似コレクション。"""

    def __init__(self, ids):
        self._ids = list(ids)
        self.added: list[str] = []

    def count(self):
        return len(self._ids)

    def query(self, *, query_embeddings, n_results, include):
        ids = self._ids[:n_results]
        # 距離0.1 -> 類似度 1/(1+0.1)≒0.91（想起側の加点フロア0.3を大きく超える）
        return {"ids": [ids], "distances": [[0.1] * len(ids)]}

    def add(self, *, ids, embeddings, documents, metadatas):
        self.added.extend(str(i) for i in ids)


class _FakeEncoder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 8))


@pytest.fixture
def vector_env(tmp_path, monkeypatch):
    """ベクトル層を chromadb / sentence-transformers 無しで動かすための足場。"""
    from api import shion_memory_vector as vec

    index_path = tmp_path / "shion_memory_index.json"
    monkeypatch.setattr(vec, "_SYNC_STATE_PATH", tmp_path / "sync_state.json")
    monkeypatch.setattr(vec, "_resolve_index_path_safe", lambda: index_path)
    monkeypatch.setattr(vec, "_INDEX_PATH", index_path)
    monkeypatch.setattr(vec, "_background_sync_started", False)
    monkeypatch.setattr(vec, "_last_sync_attempt_fingerprint", "")

    sync_calls: list[int] = []
    monkeypatch.setattr(vec, "_ensure_background_sync", lambda: sync_calls.append(1))

    def _write_index(records):
        index_path.write_text(
            json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8"
        )

    return vec, index_path, _write_index, sync_calls


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


def test_shared_encoder_returns_none_without_local_model():
    """ローカルモデル未配置の環境では共有エンコーダーは None（フォールバック前提）。"""
    from api.knowledge.vector_store import get_shared_encoder

    assert get_shared_encoder() is None


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


def test_sync_records_fingerprint_and_clears_staleness(vector_env, monkeypatch):
    """同期すると索引指紋が残り、以後は「未同期」と判定されない。"""
    vec, index_path, write_index, _ = vector_env
    write_index([{"id": "mem_a", "content": "コンテナの残価を確認する。", "status": "active"}])

    # 同期前は指紋が無いので未同期扱い
    assert vec.index_sync_is_stale(index_path) is True

    collection = _FakeCollection([])
    monkeypatch.setattr(vec, "_get_collection", lambda: collection)
    monkeypatch.setattr(vec, "_get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(vec, "_get_client", lambda: None)

    summary = vec.sync_from_index(index_path)

    assert summary["synced"] == 1
    assert vec.index_sync_is_stale(index_path) is False

    # 索引が更新されれば再び未同期になる
    write_index(
        [
            {"id": "mem_a", "content": "コンテナの残価を確認する。", "status": "active"},
            {"id": "mem_b", "content": "追加された記憶。", "status": "active"},
        ]
    )
    assert vec.index_sync_is_stale(index_path) is True


def test_revision_successor_is_not_replaced_by_stale_vector_boost(vector_env, monkeypatch):
    """改訂後にベクトル未同期なら、旧結論だけを埋め込み加点して返してはいけない。

    記憶の改訂（scripts/revise_shion_memory.apply_revisions）は索引へ後継記憶を
    追記するが、ベクトルコレクションは全量再構築バッチでしか更新されない。
    コレクションには改訂前の旧記憶しか入っていないため、鮮度を見ずに加点すると
    語彙一致0件の言い換え質問（=埋め込み層が存在する理由そのもの）で、
    「改訂済みの古い結論」だけが加点されて想起され、正しい後継記憶は
    スコア0で候補から丸ごと落ちる。
    """
    vec, index_path, write_index, sync_calls = vector_env
    monkeypatch.setenv("SHION_MEMORY_HYBRID", "1")

    # 改訂前の索引をベクトル同期した状態を作る（コレクションには mem_old だけ）
    index = {
        "records": [
            {
                "id": "mem_old",
                "content": "コンテナの法定耐用年数は6年。",
                "memory_type": "factual_memory",
                "status": "active",
                "supersedes": [],
            }
        ]
    }
    write_index(index["records"])
    collection = _FakeCollection(["mem_old"])
    monkeypatch.setattr(vec, "_get_collection", lambda: collection)
    monkeypatch.setattr(vec, "_get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(vec, "_get_client", lambda: None)
    vec.sync_from_index(index_path)
    assert vec.index_sync_is_stale(index_path) is False

    # 制度改定で改訂 → 索引に後継記憶(active)が増え、旧記憶は revised になる
    apply_revisions(
        index,
        [{"old_id": "mem_old", "new_content": "コンテナの法定耐用年数は7年（2026-07改訂）。", "reason": "制度改定"}],
    )
    write_index(index["records"])
    successor_id = next(r["id"] for r in index["records"] if "mem_old" in (r.get("supersedes") or []))

    # ベクトル側は mem_old しか知らない = 索引だけ進んだ状態
    assert vec.index_sync_is_stale(index_path) is True

    # 言い換え質問（"コンテナ"/"法定耐用年数" と語彙一致しない）
    paraphrase = "海上輸送用の箱は何年で償却する？"

    # 未同期を検出したら加点を返さず、再同期を促す
    assert vec.similarity_scores(paraphrase) == {}
    assert sync_calls, "未同期を検出したら再同期を促すべき"

    recalled = recall_memories(paraphrase, index_path=index_path)
    assert recalled["vector_used"] is False
    assert "mem_old" not in recalled["refs"], "改訂前の旧結論を単独で想起してはいけない"

    # 反証: 未同期のまま旧記憶だけに加点すると、改訂前の結論が唯一の想起結果になり、
    # 正しい後継記憶は候補から落ちる（鮮度判定を入れなかった場合の挙動）
    stale_boosted = recall_memories(
        paraphrase,
        index_path=index_path,
        vector_scores={"mem_old": 0.91},
    )
    assert stale_boosted["refs"] == ["mem_old"]
    assert successor_id not in stale_boosted["refs"]

    # 再同期後は後継記憶も埋め込み対象になり、旧結論より上に来る
    collection_after = _FakeCollection([])
    monkeypatch.setattr(vec, "_get_collection", lambda: collection_after)
    vec.sync_from_index(index_path)
    assert successor_id in collection_after.added
    assert vec.index_sync_is_stale(index_path) is False

    resynced = recall_memories(
        paraphrase,
        index_path=index_path,
        vector_scores={"mem_old": 0.91, successor_id: 0.91},
    )
    assert resynced["refs"][0] == successor_id
