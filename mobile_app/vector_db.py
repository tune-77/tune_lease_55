"""
軽量ベクトル DB（Pinecone/Weaviate の代替）

フェーズ2: ベクトル DB 統合
- ローカルベースの高速ベクトル検索
- リアルタイム同期
- スケーラビリティ対応
"""

import os
import json
import logging
import pickle
import hashlib
import math
import re
from typing import Any
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorIndex:
    """ローカルベクトルインデックス"""
    
    def __init__(self, index_name: str = "rag_index", persist_path: str = "mobile_app/.vector_index"):
        """
        初期化
        
        Args:
            index_name: インデックス名
            persist_path: 永続化パス
        """
        self.index_name = index_name
        self.persist_path = persist_path
        self.vectors: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict] = {}
        self.vector_ids = set()
        
        os.makedirs(persist_path, exist_ok=True)
        self._load()
    
    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: dict = None):
        """ベクトルを追加"""
        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata or {}
        self.vector_ids.add(vector_id)
        logger.debug(f"➕ ベクトル追加: {vector_id}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        類似ベクトルを検索（cosine similarity）
        
        Args:
            query_vector: クエリベクトル
            top_k: 返す結果数
            
        Returns:
            [(vector_id, similarity_score), ...] のリスト
        """
        if not self.vectors:
            return []
        
        results = []
        
        for vector_id, vector in self.vectors.items():
            # cosine similarity
            dot_product = np.dot(query_vector, vector)
            norm_query = np.linalg.norm(query_vector)
            norm_vec = np.linalg.norm(vector)
            
            if norm_query == 0 or norm_vec == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_vec)
            
            results.append((vector_id, float(similarity)))
        
        # 相似度でソート（降順）
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def delete_vector(self, vector_id: str):
        """ベクトルを削除"""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            del self.metadata[vector_id]
            self.vector_ids.discard(vector_id)
            logger.debug(f"🗑️  ベクトル削除: {vector_id}")
    
    def _save(self):
        """インデックスを永続化"""
        index_file = os.path.join(self.persist_path, f"{self.index_name}.pkl")
        meta_file = os.path.join(self.persist_path, f"{self.index_name}_meta.json")
        
        try:
            # ベクトル保存
            with open(index_file, 'wb') as f:
                pickle.dump(self.vectors, f)
            
            # メタデータ保存
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ インデックス保存: {self.index_name}")
        except Exception as e:
            logger.error(f"❌ 保存エラー: {e}")
    
    def _load(self):
        """インデックスを読み込む"""
        index_file = os.path.join(self.persist_path, f"{self.index_name}.pkl")
        meta_file = os.path.join(self.persist_path, f"{self.index_name}_meta.json")
        
        try:
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    self.vectors = pickle.load(f)
                self.vector_ids = set(self.vectors.keys())
            
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"✅ インデックス読み込み: {len(self.vectors)} ベクトル")
        except Exception as e:
            logger.warning(f"⚠️  読み込みエラー: {e}")
    
    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            "index_name": self.index_name,
            "vector_count": len(self.vectors),
            "vector_ids": list(self.vector_ids)[:10],  # 最初の10個
            "timestamp": datetime.now().isoformat()
        }


class LocalVectorDB:
    """ローカルベクトル DB（Pinecone/Weaviate 互換）"""
    
    def __init__(
        self,
        db_name: str = "lease_rag",
        embedding_model: Any = None,
        persist_path: str = "mobile_app/.vector_index",
        prefer_chroma: bool = True,
    ):
        """
        初期化
        
        Args:
            db_name: DB 名
            embedding_model: embedding モデル
        """
        self.db_name = db_name
        self.embedding_model = embedding_model
        self.persist_path = persist_path
        self.backend = "local"
        self.index: VectorIndex | None = None
        self._chroma_client = None
        self._chroma_collection = None

        if prefer_chroma and self._init_chroma():
            self.backend = "chroma"
        else:
            self.index = VectorIndex(index_name=db_name, persist_path=persist_path)

    def _init_chroma(self) -> bool:
        """ChromaDB を初期化。失敗時は旧ローカル実装へフォールバック。"""
        try:
            import chromadb

            chroma_path = os.path.join(self.persist_path, "chroma")
            os.makedirs(chroma_path, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=chroma_path)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.db_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("✅ ChromaDB backend initialized: %s", chroma_path)
            return True
        except Exception as e:
            logger.warning("⚠️  ChromaDB backend unavailable, using local fallback: %s", e)
            self._chroma_client = None
            self._chroma_collection = None
            return False

    @staticmethod
    def _clean_metadata(metadata: dict | None) -> dict:
        """ChromaDB が受け取れる scalar metadata に整形。"""
        cleaned: dict[str, str | int | float | bool] = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[str(key)] = value
            else:
                cleaned[str(key)] = json.dumps(value, ensure_ascii=False, default=str)
        return cleaned

    @staticmethod
    def _default_embedding(text: str, dimension: int = 384) -> np.ndarray:
        """embedding_fn 未指定時の決定的な軽量ベクトル化。ランダム検索を避ける。"""
        vec = np.zeros(dimension, dtype=np.float32)
        tokens = re.findall(r"[\w\d]+", (text or "").lower())
        if not tokens:
            tokens = [(text or "")[:64]]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(float(np.dot(vec, vec))) or 1.0
        return vec / norm

    def _embed(self, text: str, embedding_fn=None) -> np.ndarray:
        if embedding_fn:
            embedding = embedding_fn(text)
        elif self.embedding_model is not None:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        else:
            embedding = self._default_embedding(text)
        if embedding is None:
            return self._default_embedding(text)
        return np.asarray(embedding, dtype=np.float32)
    
    def upsert(self, documents: list[dict], embedding_fn=None):
        """
        ドキュメントを DB に追加/更新
        
        Args:
            documents: ドキュメントリスト
            embedding_fn: embedding 生成関数
        """
        logger.info(f"📥 {len(documents)} 個のドキュメントを upsert 中...")

        if self.backend == "chroma" and self._chroma_collection is not None:
            ids: list[str] = []
            contents: list[str] = []
            metadatas: list[dict] = []
            embeddings: list[list[float]] = []

            for doc in documents:
                doc_id = str(doc.get("id") or doc.get("path") or "unknown")
                content = f"{doc.get('title', '')} {doc.get('content', '')}"
                ids.append(doc_id)
                contents.append(content)
                metadatas.append(self._clean_metadata({
                    "title": doc.get("title"),
                    "path": doc.get("path"),
                    "content_preview": content[:200],
                    "timestamp": datetime.now().isoformat(),
                }))
                embeddings.append(self._embed(content, embedding_fn).tolist())

            if ids:
                self._chroma_collection.upsert(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            logger.info(f"✅ ChromaDB upsert 完了: {len(ids)} 件")
            return

        if self.index is None:
            self.index = VectorIndex(index_name=self.db_name, persist_path=self.persist_path)
        
        for doc in documents:
            doc_id = doc.get("id", doc.get("path", "unknown"))
            content = f"{doc.get('title', '')} {doc.get('content', '')}"
            embedding = self._embed(content, embedding_fn)
            
            self.index.add_vector(
                vector_id=doc_id,
                vector=embedding,
                metadata={
                    "title": doc.get("title"),
                    "path": doc.get("path"),
                    "content_preview": content[:200],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        self.index._save()
        logger.info(f"✅ Upsert 完了: {len(documents)} 件")
    
    def query(self, query_text: str, embedding_fn=None, top_k: int = 5) -> list[dict]:
        """
        ベクトル DB をクエリ
        
        Args:
            query_text: クエリテキスト
            embedding_fn: embedding 生成関数
            top_k: 返す結果数
            
        Returns:
            スコア付きドキュメント
        """
        logger.info(f"🔍 クエリ: {query_text}")
        query_embedding = self._embed(query_text, embedding_fn)

        if self.backend == "chroma" and self._chroma_collection is not None:
            count = self._chroma_collection.count()
            if count == 0:
                return []
            results = self._chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )

            response = []
            for vector_id, document, metadata, distance in zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            ):
                similarity = max(0.0, min(1.0, 1.0 - float(distance or 0.0)))
                response.append({
                    "id": vector_id,
                    "similarity_score": similarity,
                    "content_preview": (document or "")[:200],
                    **(metadata or {}),
                })

            logger.info(f"✅ ChromaDB query 完了: {len(response)} 件")
            return response

        if self.index is None:
            self.index = VectorIndex(index_name=self.db_name, persist_path=self.persist_path)
        
        # 検索
        results = self.index.search(query_embedding, top_k=top_k)
        
        # メタデータを追加
        response = []
        for vector_id, similarity in results:
            metadata = self.index.metadata.get(vector_id, {})
            response.append({
                "id": vector_id,
                "similarity_score": similarity,
                **metadata
            })
        
        logger.info(f"✅ {len(response)} 件の結果を取得")
        return response
    
    def delete(self, document_ids: list[str]) -> int:
        """
        PHASE 2: ドキュメントを削除

        Args:
            document_ids: 削除するドキュメント ID のリスト

        Returns:
            実際に削除されたドキュメント数
        """
        logger.info(f"🗑️  {len(document_ids)} 個のドキュメント削除中...")

        if self.backend == "chroma" and self._chroma_collection is not None:
            existing = self._chroma_collection.get(ids=[str(doc_id) for doc_id in document_ids])
            existing_ids = [str(doc_id) for doc_id in existing.get("ids", [])]
            if existing_ids:
                self._chroma_collection.delete(ids=existing_ids)
            logger.info(f"✅ ChromaDB 削除完了: {len(existing_ids)} / {len(document_ids)} 件")
            return len(existing_ids)

        if self.index is None:
            self.index = VectorIndex(index_name=self.db_name, persist_path=self.persist_path)

        deleted_count = 0
        for doc_id in document_ids:
            if doc_id in self.index.vectors:
                self.index.delete_vector(doc_id)
                deleted_count += 1
            else:
                logger.warning(f"⚠️  ドキュメント {doc_id} は存在しません")

        if deleted_count > 0:
            self.index._save()

        logger.info(f"✅ 削除完了: {deleted_count} / {len(document_ids)} 件")
        return deleted_count

    def get_stats(self) -> dict:
        """DB の統計情報を取得"""
        if self.backend == "chroma" and self._chroma_collection is not None:
            return {
                "index_name": self.db_name,
                "backend": "chroma",
                "vector_count": self._chroma_collection.count(),
                "timestamp": datetime.now().isoformat(),
            }
        if self.index is None:
            self.index = VectorIndex(index_name=self.db_name, persist_path=self.persist_path)
        stats = self.index.get_stats()
        stats["backend"] = "local"
        return stats


# テスト用
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 ベクトル DB テスト")
    print("=" * 60)
    
    # DB 初期化
    db = LocalVectorDB(db_name="test_rag")
    
    # テストドキュメント
    test_docs = [
        {"id": "doc1", "title": "Q-Risk", "content": "自己資本比率30%が基準"},
        {"id": "doc2", "title": "業種別リスク", "content": "飲食業は高リスク"},
        {"id": "doc3", "title": "物件評価", "content": "残価評価は重要"},
    ]
    
    # Upsert
    db.upsert(test_docs)
    
    # クエリ
    results = db.query("自己資本比率", top_k=2)
    print("\n📋 検索結果:")
    for r in results:
        print(f"  [{r['similarity_score']:.3f}] {r['title']}")
    
    # 統計
    stats = db.get_stats()
    print(f"\n📊 DB 統計: {stats}")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")
