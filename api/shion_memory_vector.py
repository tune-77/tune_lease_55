"""紫苑記憶索引の埋め込み検索レイヤー（ハイブリッド想起のベクトル側）。

Obsidian RAG と同じ埋め込みモデル（api/knowledge/vector_store.py の
paraphrase-multilingual-MiniLM-L12-v2 系）を再利用し、`data/shion_memory_index.json`
のレコードを ChromaDB コレクション `shion_memory` に同期する。

依存（chromadb / sentence-transformers）が無い環境では `is_available()` が
False を返し、想起はキーワード検索のみで動く。Cloud Run など軽量環境では
環境変数 `SHION_MEMORY_HYBRID` を設定しない限り呼ばれない。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = _REPO_ROOT / "data" / "shion_memory_index.json"
_CHROMA_DIR = str(_REPO_ROOT / "api" / "chroma_db")
_COLLECTION_NAME = "shion_memory"

_lock = threading.Lock()
_client: Any = None
_encoder: Any = None
_import_failed = False
_background_sync_started = False


def hybrid_enabled() -> bool:
    """環境変数でハイブリッド想起が有効化されているか。"""
    raw = os.environ.get("SHION_MEMORY_HYBRID", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_client() -> Any:
    global _client, _import_failed
    if _client is not None or _import_failed:
        return _client
    with _lock:
        if _client is not None or _import_failed:
            return _client
        try:
            import chromadb

            _client = chromadb.PersistentClient(path=_CHROMA_DIR)
        except Exception as exc:
            logger.info("[ShionMemoryVector] chromadb unavailable: %s", exc)
            _import_failed = True
    return _client


def _get_encoder() -> Any:
    global _encoder, _import_failed
    if _encoder is not None or _import_failed:
        return _encoder
    with _lock:
        if _encoder is not None or _import_failed:
            return _encoder
        # まず Obsidian RAG 側の初期化済みエンコーダーを共有する（同一モデルの
        # 二重ロードで ~500MB を余分に食わないため）。
        try:
            from api.knowledge.vector_store import get_shared_encoder

            shared = get_shared_encoder()
            if shared is not None:
                _encoder = shared
                return _encoder
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer

            # Obsidian RAG と同じモデル解決（ローカルキャッシュ優先）を使う
            from api.knowledge.vector_store import _MODEL_NAME

            _encoder = SentenceTransformer(_MODEL_NAME, device="cpu")
        except Exception as exc:
            logger.info("[ShionMemoryVector] encoder unavailable: %s", exc)
            _import_failed = True
    return _encoder


def _get_collection() -> Any:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_or_create_collection(name=_COLLECTION_NAME)
    except Exception as exc:
        logger.warning("[ShionMemoryVector] collection error: %s", exc)
        return None


def is_available() -> bool:
    """ベクトル検索が使える状態か（依存あり・コレクションに記憶あり）。"""
    collection = _get_collection()
    if collection is None:
        return False
    try:
        return collection.count() > 0
    except Exception:
        return False


def sync_from_index(index_path: Path = _INDEX_PATH, *, batch_size: int = 64) -> dict[str, int]:
    """記憶索引の内容をベクトルコレクションへ同期する（全量再構築）。

    private / deprecated は想起対象外なので同期しない。
    """
    collection = _get_collection()
    encoder = _get_encoder()
    if collection is None or encoder is None:
        return {"synced": 0, "skipped": 0, "available": 0}

    try:
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[ShionMemoryVector] index unreadable: %s", exc)
        return {"synced": 0, "skipped": 0, "available": 0}

    records = [r for r in data.get("records") or [] if isinstance(r, dict)]
    targets: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        status = str(record.get("status") or "active")
        rid = str(record.get("id") or "")
        content = str(record.get("content") or "").strip()
        if not rid or not content or status in {"private", "deprecated"}:
            skipped += 1
            continue
        targets.append(record)

    # 全量再構築: コレクションごと作り直す。get(include=[]) は chromadb の
    # バージョンによって挙動が違うため、delete_collection の方が版差に強い。
    client = _get_client()
    try:
        if client is not None:
            client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass  # 初回は存在しないだけなので無視してよい
    collection = _get_collection()
    if collection is None:
        return {"synced": 0, "skipped": skipped, "available": len(targets)}

    synced = 0
    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        contents = [str(r["content"])[:512] for r in batch]
        try:
            embeddings = encoder.encode(contents, show_progress_bar=False).tolist()
            collection.add(
                ids=[str(r["id"]) for r in batch],
                embeddings=embeddings,
                documents=contents,
                metadatas=[
                    {
                        "memory_type": str(r.get("memory_type") or ""),
                        "status": str(r.get("status") or "active"),
                        "source_path": str(r.get("source_path") or ""),
                    }
                    for r in batch
                ],
            )
            synced += len(batch)
        except Exception as exc:
            logger.warning("[ShionMemoryVector] batch add failed: %s", exc)
    return {"synced": synced, "skipped": skipped, "available": len(targets)}


def _ensure_background_sync() -> None:
    """コレクションが空のとき、初回だけバックグラウンドで索引から構築する。

    Cloud Run のイメージには api/chroma_db が含まれない（.dockerignore）ため、
    SHION_MEMORY_HYBRID=1 を設定するだけで初回起動時に自動構築される必要がある。
    構築完了まで想起はキーワードのみで動き、完了後の質問からハイブリッドになる。
    """
    global _background_sync_started
    if _background_sync_started:
        return
    with _lock:
        if _background_sync_started:
            return
        _background_sync_started = True
    thread = threading.Thread(
        target=_background_sync_worker, name="shion-memory-vector-sync", daemon=True
    )
    thread.start()


def _background_sync_worker() -> None:
    try:
        from api.shion_memory_recall import resolve_index_path

        summary = sync_from_index(resolve_index_path())
        logger.info("[ShionMemoryVector] background sync done: %s", summary)
    except Exception as exc:
        logger.warning("[ShionMemoryVector] background sync failed: %s", exc)


def similarity_scores(question: str, *, top_k: int = 24) -> dict[str, float]:
    """質問に近い記憶ID → 類似度(0..1) を返す。失敗時は空 dict。"""
    text = (question or "").strip()
    if not text:
        return {}
    collection = _get_collection()
    if collection is None:
        return {}
    try:
        count = collection.count()
        if count == 0:
            _ensure_background_sync()
            return {}
        encoder = _get_encoder()
        if encoder is None:
            return {}
        embedding = encoder.encode([text], show_progress_bar=False).tolist()[0]
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["distances"],
        )
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        scores: dict[str, float] = {}
        for rid, distance in zip(ids, distances):
            try:
                similarity = 1.0 / (1.0 + max(0.0, float(distance)))
            except (TypeError, ValueError):
                continue
            scores[str(rid)] = similarity
        return scores
    except Exception as exc:
        logger.warning("[ShionMemoryVector] query failed: %s", exc)
        return {}
