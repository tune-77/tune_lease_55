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
from typing import Any, Optional
from pathlib import Path
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
    
    def __init__(self, db_name: str = "lease_rag", embedding_model: Any = None):
        """
        初期化
        
        Args:
            db_name: DB 名
            embedding_model: embedding モデル
        """
        self.db_name = db_name
        self.embedding_model = embedding_model
        self.index = VectorIndex(index_name=db_name)
    
    def upsert(self, documents: list[dict], embedding_fn=None):
        """
        ドキュメントを DB に追加/更新
        
        Args:
            documents: ドキュメントリスト
            embedding_fn: embedding 生成関数
        """
        logger.info(f"📥 {len(documents)} 個のドキュメントを upsert 中...")
        
        for doc in documents:
            doc_id = doc.get("id", doc.get("path", "unknown"))
            content = f"{doc.get('title', '')} {doc.get('content', '')}"
            
            # embedding 生成
            if embedding_fn:
                embedding = embedding_fn(content)
            else:
                # ダミー embedding（ランダムベクトル）
                embedding = np.random.rand(384)
            
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
        
        # query embedding 生成
        if embedding_fn:
            query_embedding = embedding_fn(query_text)
        else:
            query_embedding = np.random.rand(384)
        
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
        return self.index.get_stats()


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
