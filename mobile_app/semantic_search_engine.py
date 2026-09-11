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
import hashlib
import logging
import sys
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import resolve_obsidian_vault  # noqa: E402

DEFAULT_LOCAL_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentence-transformers"
    / "paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_REMOTE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _default_model_name() -> str:
    configured = os.environ.get("OBSIDIAN_RAG_MODEL", "").strip()
    if configured:
        return configured
    if DEFAULT_LOCAL_MODEL_DIR.is_dir():
        return str(DEFAULT_LOCAL_MODEL_DIR)
    return DEFAULT_REMOTE_MODEL_NAME


class SemanticSearchEngine:
    """セマンティック検索エンジン"""
    
    def __init__(self, model_name: str | None = None):
        """
        初期化
        
        Args:
            model_name: SentenceTransformer モデル名
        """
        self.model_name = model_name or _default_model_name()
        self.embeddings_cache = {}
        self._document_embedding_cache: dict[str, np.ndarray] = {}
        self._query_embedding_cache: dict[str, np.ndarray] = {}
        self.embedding_model = None
        # 索引先は runtime_paths の解決結果に従う（env → iCloud）。
        # ここを直書きすると RAG の索引先だけ他モジュールとずれる。
        self.vault_path = str(resolve_obsidian_vault())
        self.cache_file = "mobile_app/.embeddings_cache.json"
        if EMBEDDING_AVAILABLE:
            try:
                logger.info(f"📦 Embedding モデル読み込み中: {self.model_name}")
                if os.path.isdir(self.model_name):
                    self.embedding_model = self._load_local_sentence_transformer(self.model_name)
                elif os.environ.get("OBSIDIAN_RAG_USE_ENCODER", "").strip() == "1":
                    self.embedding_model = SentenceTransformer(self.model_name, device="cpu")
                else:
                    logger.info(
                        "Embedding モデルはローカルに無いためスキップします: %s",
                        self.model_name,
                    )
                logger.info("✅ Embedding モデル読み込み完了")
            except Exception as e:
                logger.error(f"❌ モデル読み込みエラー: {e}")
                self.embedding_model = None
        
        self._load_cache()

    def _load_local_sentence_transformer(self, model_name: str):
        try:
            return SentenceTransformer(
                model_name,
                device="cpu",
                local_files_only=True,
            )
        except TypeError:
            old_hf = os.environ.get("HF_HUB_OFFLINE")
            old_tf = os.environ.get("TRANSFORMERS_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                return SentenceTransformer(model_name, device="cpu")
            finally:
                if old_hf is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_hf
                if old_tf is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = old_tf
    
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
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"❌ Embedding 生成エラー: {e}")
            return None

    def _query_embedding(self, text: str) -> np.ndarray | None:
        cached = self._query_embedding_cache.get(text)
        if cached is not None:
            return cached
        embedding = self.generate_embedding(text)
        if embedding is not None:
            if len(self._query_embedding_cache) >= 256:
                self._query_embedding_cache.pop(next(iter(self._query_embedding_cache)))
            self._query_embedding_cache[text] = embedding
        return embedding
    
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

        query_embedding = self._query_embedding(query)
        if query_embedding is None:
            return documents[:top_k]

        if not documents:
            return []

        scores: dict[int, float] = {}
        embeddings: list[np.ndarray | None] = [None] * len(documents)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        cache_keys: list[str] = []
        for i, doc in enumerate(documents):
            doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            identity = str(doc.get("path") or doc.get("id") or i)
            digest = hashlib.sha1(doc_text.encode("utf-8")).hexdigest()
            cache_key = f"{identity}:{digest}"
            cache_keys.append(cache_key)
            cached = self._document_embedding_cache.get(cache_key)
            if cached is not None:
                embeddings[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(doc_text)

        if missing_texts and self.embedding_model is not None:
            try:
                encoded = self.embedding_model.encode(
                    missing_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for idx, vector in zip(missing_indices, encoded):
                    array = np.asarray(vector, dtype=float)
                    embeddings[idx] = array
                    self._document_embedding_cache[cache_keys[idx]] = array
            except Exception as exc:
                logger.warning(f"⚠️ 文書Embeddingの一括生成に失敗: {exc}")

        valid_indices = [i for i, vector in enumerate(embeddings) if vector is not None]
        if valid_indices:
            matrix = np.vstack([embeddings[i] for i in valid_indices])
            query_norm = np.linalg.norm(query_embedding)
            row_norms = np.linalg.norm(matrix, axis=1)
            denominators = row_norms * query_norm
            similarities = np.divide(
                matrix @ query_embedding,
                denominators,
                out=np.zeros(len(valid_indices), dtype=float),
                where=denominators != 0,
            )
            for idx, similarity in zip(valid_indices, similarities):
                scores[idx] = float(np.clip(similarity, 0.0, 1.0))

        results = [
            {**doc, "similarity_score": scores.get(i, 0.0)}
            for i, doc in enumerate(documents)
        ]

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
        """Obsidian ドキュメントを共通ブリッジ経由で読み込む。"""
        try:
            from mobile_app.obsidian_bridge import iter_indexed_obsidian_documents
        except Exception:
            try:
                from obsidian_bridge import iter_indexed_obsidian_documents
            except Exception as e:
                logger.warning(f"⚠️  Obsidian 共通ブリッジ読み込みエラー: {e}")
                return []

        documents = iter_indexed_obsidian_documents(include_chat_logs=False, max_chars=1000)
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
