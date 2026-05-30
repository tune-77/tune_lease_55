"""
セマンティック検索エンジン（embedding + cosine similarity）
リース審査 AI RAG システム用

フェーズ1: セマンティック検索
- 各ドキュメントの embedding 生成・キャッシュ
- クエリの embedding 計算
- cosine similarity で関連性スコア計算
"""

import os
import json
import logging
from pathlib import Path
from typing import Any
from functools import lru_cache
import numpy as np
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using fallback mode.")

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """セマンティック検索エンジン"""
    
    def __init__(self, model_name: str = "sentence-transformers/multilingual-MiniLM-L12-v2"):
        """
        初期化
        
        Args:
            model_name: SentenceTransformer モデル名
        """
        self.model_name = model_name
        self.embeddings_cache = {}
        self.embedding_model = None
        self.vault_path = "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
        self.cache_file = "mobile_app/.embeddings_cache.json"
        
        if EMBEDDING_AVAILABLE:
            try:
                logger.info(f"📦 Embedding モデル読み込み中: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                logger.info("✅ Embedding モデル読み込み完了")
            except Exception as e:
                logger.error(f"❌ モデル読み込みエラー: {e}")
                self.embedding_model = None
        
        self._load_cache()
    
    def _load_cache(self):
        """キャッシュをファイルから読み込む"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.embeddings_cache = json.load(f)
                logger.info(f"✅ キャッシュ読み込み: {len(self.embeddings_cache)} エントリ")
            except Exception as e:
                logger.warning(f"⚠️  キャッシュ読み込みエラー: {e}")
    
    def _save_cache(self):
        """キャッシュをファイルに保存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ キャッシュ保存: {len(self.embeddings_cache)} エントリ")
        except Exception as e:
            logger.warning(f"⚠️  キャッシュ保存エラー: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray | None:
        """
        テキストの embedding を生成
        
        Args:
            text: 入力テキスト
            
        Returns:
            embedding ベクトル（又は None）
        """
        if not EMBEDDING_AVAILABLE or self.embedding_model is None:
            return None
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"❌ Embedding 生成エラー: {e}")
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        cosine similarity を計算
        
        Args:
            vec1, vec2: ベクトル
            
        Returns:
            相似度スコア（0-1）
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(max(0, min(1, similarity)))  # 0-1 に正規化
        except Exception as e:
            logger.error(f"❌ Cosine similarity エラー: {e}")
            return 0.0
    
    def search_similar(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        類似ドキュメントを検索
        
        Args:
            query: 検索クエリ
            documents: ドキュメント一覧
            top_k: 返す結果数
            
        Returns:
            スコア付きドキュメント（降順）
        """
        if not EMBEDDING_AVAILABLE:
            logger.warning("⚠️  Embedding 機能が利用できません")
            return documents[:top_k]
        
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return documents[:top_k]
        
        results = []
        for doc in documents:
            doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            doc_embedding = self.generate_embedding(doc_text)
            
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            results.append({
                **doc,
                "similarity_score": similarity
            })
        
        # スコアで降順ソート
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            "model": self.model_name,
            "embedding_available": EMBEDDING_AVAILABLE,
            "cache_entries": len(self.embeddings_cache),
            "timestamp": datetime.now().isoformat()
        }


class SemanticRAGRetriever:
    """セマンティック RAG 検索器"""
    
    def __init__(self):
        self.search_engine = SemanticSearchEngine()
        self.obsidian_documents = self._load_obsidian_documents()
    
    def _load_obsidian_documents(self) -> list[dict]:
        """Obsidian ドキュメントを読み込む"""
        documents = []
        vault_path = Path("/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault")
        
        for md_file in vault_path.rglob("*.md"):
            if any(skip in str(md_file) for skip in [".obsidian", ".claude", ".claudian"]):
                continue
            
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Frontmatter を抽出
                title = md_file.stem
                rel_path = md_file.relative_to(vault_path)
                
                documents.append({
                    "title": title,
                    "path": str(rel_path),
                    "content": content[:1000],  # 最初の1000文字
                    "full_path": str(md_file)
                })
            except Exception as e:
                logger.warning(f"⚠️  ドキュメント読み込みエラー: {md_file} - {e}")
        
        logger.info(f"✅ {len(documents)} 個のドキュメント読み込み完了")
        return documents
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        セマンティック検索で関連ドキュメントを検索
        
        Args:
            query: 検索クエリ
            top_k: 返す結果数
            
        Returns:
            類似度スコア付きドキュメント
        """
        logger.info(f"🔍 セマンティック検索: {query}")
        
        results = self.search_engine.search_similar(query, self.obsidian_documents, top_k=top_k)
        
        logger.info(f"✅ {len(results)} 件の結果を取得")
        return results


# テスト用
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 セマンティック検索エンジン テスト")
    print("=" * 60)
    
    retriever = SemanticRAGRetriever()
    
    # テストクエリ
    test_queries = [
        "飲食業のリース審査",
        "自己資本比率30%",
        "条件付き承認",
    ]
    
    for query in test_queries:
        print(f"\n📌 クエリ: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['similarity_score']:.3f}] {result['title']}")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")
