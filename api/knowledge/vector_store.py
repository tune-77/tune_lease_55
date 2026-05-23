"""
ChromaDB 永続化ベクトルストア。
paraphrase-multilingual-MiniLM-L12-v2 モデルで日本語テキストをベクトル化する。
"""
from __future__ import annotations

import os
import logging
import threading
from typing import Literal

logger = logging.getLogger(__name__)

_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
_COLLECTION_NAME = "obsidian_knowledge"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class KnowledgeVectorStore:
    """ChromaDB ラッパー。遅延初期化でスタートアップをブロックしない。"""

    def __init__(self, chroma_dir: str = _CHROMA_DIR, model_name: str = _MODEL_NAME):
        self._chroma_dir = chroma_dir
        self._model_name = model_name
        self._client = None
        self._collection = None
        self._encoder = None
        self._init_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """初回アクセス時に ChromaDB と encoder を初期化する。スレッドセーフ。"""
        if self._encoder is not None:
            return

        with self._init_lock:
            # ロック取得後に再チェック（二重初期化防止）
            if self._encoder is not None:
                return

            import chromadb
            from sentence_transformers import SentenceTransformer

            os.makedirs(self._chroma_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._encoder = SentenceTransformer(self._model_name, device="cpu")
            logger.info(f"[KnowledgeVectorStore] initialized: {self._chroma_dir}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        return self._encoder.encode(texts, normalize_embeddings=True).tolist()

    def upsert_chunks(self, chunks: list) -> int:
        """Chunk リストを ChromaDB にアップサート。追加件数を返す。"""
        if not chunks:
            return 0
        self._ensure_initialized()

        ids = [c.doc_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{
            "file_name": c.file_name,
            "file_path": c.file_path,
            "section": c.section,
            "obsidian_ref": c.obsidian_ref,
            "mtime": c.mtime,
            **{k: str(v) for k, v in c.metadata.items() if k not in ("section",)},
        } for c in chunks]
        embeddings = self._embed(texts)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def search(
        self,
        query: str,
        mode: Literal["support", "refute", "both"] = "both",
        top_k: int = 3,
    ) -> list[dict]:
        """
        クエリに近いチャンクを検索する。

        Args:
            query:  検索クエリ（自然言語）
            mode:   "support" → 肯定的エビデンス、"refute" → 否定的エビデンス、
                    "both" → 両方（プレフィックスを付けて検索）
            top_k:  返す最大件数

        Returns:
            [{"text": str, "ref": str, "distance": float, ...}, ...]
        """
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        if mode == "support":
            effective_query = f"成功事例 承認 リスク低い {query}"
        elif mode == "refute":
            effective_query = f"失敗事例 否決 リスク高い 問題 {query}"
        else:
            effective_query = query

        embedding = self._embed([effective_query])[0]
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(1, self._collection.count())),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "ref": meta.get("obsidian_ref", ""),
                "file_name": meta.get("file_name", ""),
                "section": meta.get("section", ""),
                "distance": round(float(dist), 4),
            })
        return hits

    def count(self) -> int:
        """インデックス内のドキュメント数を返す。"""
        try:
            self._ensure_initialized()
            return self._collection.count()
        except Exception:
            return 0


# モジュールレベルシングルトン（APIサーバーで共有）
_store: KnowledgeVectorStore | None = None


def get_store() -> KnowledgeVectorStore:
    global _store
    if _store is None:
        _store = KnowledgeVectorStore()
    return _store
