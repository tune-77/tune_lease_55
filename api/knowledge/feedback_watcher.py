"""
Obsidian Vault の Feedback/ フォルダを監視し、担当者フィードバックを ChromaDB に登録する。

フィードバックファイルの frontmatter 形式:
---
case_id: CASE-001
agent: 石橋
correction: 建設業の単年黒字は季節性で評価すべき
---
（本文は任意）
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from typing import Iterator

logger = logging.getLogger(__name__)

_DEFAULT_VAULT_PATH = (
    "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/"
)
_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", _DEFAULT_VAULT_PATH)
_FEEDBACK_SUBDIR = "Feedback"
_FEEDBACK_COLLECTION = "lease_feedback"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ── ChromaDB フィードバックストア（シングルトン） ─────────────────────────────────

_feedback_store = None
_feedback_lock = threading.Lock()


class FeedbackVectorStore:
    """フィードバック専用の ChromaDB コレクション。"""

    def __init__(self, chroma_dir: str, model_name: str = _MODEL_NAME):
        self._chroma_dir = chroma_dir
        self._model_name = model_name
        self._client = None
        self._collection = None
        self._encoder = None
        self._init_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        if self._encoder is not None:
            return
        with self._init_lock:
            if self._encoder is not None:
                return
            import chromadb
            from sentence_transformers import SentenceTransformer

            os.makedirs(self._chroma_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name=_FEEDBACK_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            self._encoder = SentenceTransformer(self._model_name, device="cpu")
            logger.info(f"[FeedbackStore] initialized: {self._chroma_dir}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        return self._encoder.encode(texts, normalize_embeddings=True).tolist()

    def upsert(self, fb_id: str, text: str, metadata: dict) -> None:
        self._ensure_initialized()
        embedding = self._embed([text])[0]
        self._collection.upsert(
            ids=[fb_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding],
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        self._ensure_initialized()
        if self._collection.count() == 0:
            return []
        embedding = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
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
                "case_id": meta.get("case_id", ""),
                "agent": meta.get("agent", ""),
                "correction": meta.get("correction", ""),
                "file_name": meta.get("file_name", ""),
                "distance": round(float(dist), 4),
            })
        return hits

    def count(self) -> int:
        try:
            self._ensure_initialized()
            return self._collection.count()
        except Exception:
            return 0


def _get_feedback_store() -> FeedbackVectorStore:
    global _feedback_store
    if _feedback_store is None:
        with _feedback_lock:
            if _feedback_store is None:
                chroma_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "chroma_db",
                )
                _feedback_store = FeedbackVectorStore(chroma_dir)
    return _feedback_store


# ── フィードバックファイルパーサー ────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta, text[m.end():]


def _file_id(file_path: str) -> str:
    return "fb_" + hashlib.md5(file_path.encode()).hexdigest()[:16]


def _iter_feedback_files(vault_path: str) -> Iterator[tuple[str, dict, str]]:
    """
    Feedback/ 配下の .md ファイルをスキャンし、
    (file_path, metadata, full_text) を yield する。
    """
    feedback_dir = os.path.join(vault_path, _FEEDBACK_SUBDIR)
    if not os.path.isdir(feedback_dir):
        logger.debug(f"[FeedbackWatcher] Feedback/ dir not found: {feedback_dir}")
        return

    for root, _, files in os.walk(feedback_dir):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"[FeedbackWatcher] read error {fpath}: {e}")
                continue

            meta, body = _parse_frontmatter(content)
            # README.md は説明用のためスキップ
            if fname.lower() == "readme.md":
                continue
            # case_id か correction がなければスキップ
            if not meta.get("case_id") and not meta.get("correction"):
                continue

            # ChromaDB に保存するテキスト: correction を軸に body も含める
            correction = meta.get("correction", "")
            full_text = correction
            if body.strip():
                full_text = f"{correction}\n{body.strip()}" if correction else body.strip()

            yield fpath, meta, full_text


# ── 公開 API ─────────────────────────────────────────────────────────────────────

def load_all_feedback(vault_path: str = _VAULT_PATH) -> int:
    """
    Feedback/ 配下のファイルをすべて読み込んで ChromaDB にインデックスする。
    追加件数を返す。
    """
    store = _get_feedback_store()
    count = 0
    for fpath, meta, text in _iter_feedback_files(vault_path):
        fb_id = _file_id(fpath)
        chroma_meta = {
            "case_id": meta.get("case_id", ""),
            "agent": meta.get("agent", ""),
            "correction": meta.get("correction", ""),
            "file_name": os.path.basename(fpath),
            "file_path": fpath,
        }
        try:
            store.upsert(fb_id, text, chroma_meta)
            count += 1
        except Exception as e:
            logger.warning(f"[FeedbackWatcher] upsert failed {fpath}: {e}")

    logger.info(f"[FeedbackWatcher] loaded {count} feedback entries")
    return count


def search_feedback(query: str, top_k: int = 3) -> list[dict]:
    """クエリに関連する過去の訂正フィードバックを検索する。"""
    return _get_feedback_store().search(query, top_k=top_k)


def feedback_count() -> int:
    """インデックス済みフィードバック数を返す。"""
    return _get_feedback_store().count()


def start_background_feedback_loading(vault_path: str = _VAULT_PATH) -> threading.Thread:
    """バックグラウンドスレッドでフィードバックを読み込む。"""
    def _task():
        try:
            load_all_feedback(vault_path)
        except Exception as e:
            logger.warning(f"[FeedbackWatcher] background loading failed: {e}")

    t = threading.Thread(target=_task, name="feedback-watcher", daemon=True)
    t.start()
    return t
