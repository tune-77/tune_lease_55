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
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = _REPO_ROOT / "data" / "shion_memory_index.json"
_CHROMA_DIR = str(_REPO_ROOT / "api" / "chroma_db")
_COLLECTION_NAME = "shion_memory"
# 最後に同期した索引の指紋を置くサイドカー。コレクションの metadata を使うと
# chromadb の版差に依存するため、ファイル1枚で持つ方が壊れにくい。
_SYNC_STATE_PATH = Path(_CHROMA_DIR) / ".shion_memory_sync_state.json"

_lock = threading.Lock()
_client: Any = None
_encoder: Any = None
_import_failed = False
_background_sync_started = False
# 直前に同期を試みた索引指紋。同じ指紋で何度も再同期しないための空回り防止。
_last_sync_attempt_fingerprint = ""


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


def index_fingerprint(index_path: Path = _INDEX_PATH) -> str:
    """索引ファイルの指紋（mtime+size）。読めなければ空文字。

    中身のハッシュではなく stat のみなのは、想起1回ごとに呼ばれる鮮度判定を
    索引全体のパースなしで済ませるため。
    """
    try:
        st = Path(index_path).stat()
    except OSError:
        return ""
    return f"{st.st_mtime_ns}:{st.st_size}"


def _read_synced_fingerprint() -> str:
    try:
        data = json.loads(_SYNC_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("index_fingerprint") or "") if isinstance(data, dict) else ""


def _write_synced_fingerprint(fingerprint: str, synced: int) -> None:
    if not fingerprint:
        return
    payload = {
        "index_fingerprint": fingerprint,
        "synced": synced,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        _SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SYNC_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("[ShionMemoryVector] sync state unwritable: %s", exc)


def index_sync_is_stale(index_path: Path = _INDEX_PATH) -> bool:
    """索引がベクトルコレクションより新しいか（=同期漏れがあるか）。

    記憶の改訂（scripts/revise_shion_memory.py）は索引へ後継記憶を追記するが、
    ベクトル側は全量再構築バッチでしか更新されない。コレクションが空でない限り
    自動復元も走らないため、これを見ないと後継記憶が埋め込みブーストを受けられず、
    改訂済みの旧結論（コレクションに残っている）の方が言い換え質問で上位に来る。
    """
    fingerprint = index_fingerprint(index_path)
    if not fingerprint:
        return False  # 索引が読めない時は判定不能。無用な再同期を避ける。
    return fingerprint != _read_synced_fingerprint()


def sync_from_index(index_path: Path = _INDEX_PATH, *, batch_size: int = 64) -> dict[str, int]:
    """記憶索引の内容をベクトルコレクションへ同期する（全量再構築）。

    private / deprecated は想起対象外なので同期しない。

    呼び出し元は2箇所ある:
      1. scripts/build_shion_memory_vector_index.py（バッチ、通常経路）
      2. _background_sync_worker（本モジュール下部、Cloud Run 初回起動時の
         ランタイム自動復元）— 他の記憶系ファイルが「バッチ実行のみ・
         ランタイムでは書き込まない」方針を採る中で、これが唯一の意図的な例外。
         Cloud Run はデプロイの度にベクトルコレクションが空になるため、
         起動時に一度だけ index から再構築する必要がある。
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

    from api.shion_memory_taxonomy import NON_RECALLABLE_STATUSES

    records = [r for r in data.get("records") or [] if isinstance(r, dict)]
    targets: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        status = str(record.get("status") or "active")
        rid = str(record.get("id") or "")
        content = str(record.get("content") or "").strip()
        # 除外集合は想起側と共有する。片方だけ広げるとベクトル表現の無い
        # 想起候補が生まれる（NON_RECALLABLE_STATUSES の定義コメント参照）。
        if not rid or not content or status in NON_RECALLABLE_STATUSES:
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
    started = time.monotonic()
    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        # topic（ノートタイトル）があれば前置して埋め込む。分割スニペットは
        # 主題語（例: 法定耐用年数）を失いやすく、topic 併用で想起精度が上がる。
        contents = [
            (f"{topic}: {r['content']}" if (topic := str(r.get("topic") or "").strip()) else str(r["content"]))[:512]
            for r in batch
        ]
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
                        "domain": str(r.get("domain") or ""),
                    }
                    for r in batch
                ],
            )
            synced += len(batch)
        except Exception as exc:
            logger.warning("[ShionMemoryVector] batch add failed: %s", exc)

    from api.memory_cost_log import log_memory_cost

    log_memory_cost(
        phase="construction",
        func="shion_memory_vector.sync_from_index",
        elapsed_ms=(time.monotonic() - started) * 1000,
        item_count=synced,
    )
    # 同期した索引の指紋を残す。次回以降 index_sync_is_stale() が
    # 「索引だけ進んでコレクションが取り残された」状態を検出できる。
    _write_synced_fingerprint(index_fingerprint(index_path), synced)
    return {"synced": synced, "skipped": skipped, "available": len(targets)}


def _ensure_background_sync() -> None:
    """コレクションが空のとき、初回だけバックグラウンドで索引から構築する。

    Cloud Run のイメージには api/chroma_db が含まれない（.dockerignore）ため、
    SHION_MEMORY_HYBRID=1 を設定するだけで初回起動時に自動構築される必要がある。
    構築完了まで想起はキーワードのみで動き、完了後の質問からハイブリッドになる。
    """
    global _background_sync_started, _last_sync_attempt_fingerprint
    if _background_sync_started:
        return  # 同期スレッドが実行中
    fingerprint = index_fingerprint(_resolve_index_path_safe())
    with _lock:
        if _background_sync_started:
            return
        # 同じ索引指紋で再試行し続けない（同期が失敗し続ける環境での空回り防止）。
        # 索引が更新されれば指紋が変わり、改訂後の再同期は改めて走る。
        if fingerprint and fingerprint == _last_sync_attempt_fingerprint:
            return
        _last_sync_attempt_fingerprint = fingerprint
        _background_sync_started = True
    thread = threading.Thread(
        target=_background_sync_worker, name="shion-memory-vector-sync", daemon=True
    )
    thread.start()


def _resolve_index_path_safe() -> Path:
    try:
        from api.shion_memory_recall import resolve_index_path

        return resolve_index_path()
    except Exception:
        return _INDEX_PATH


def _background_sync_worker() -> None:
    global _background_sync_started
    try:
        summary = sync_from_index(_resolve_index_path_safe())
        logger.info("[ShionMemoryVector] background sync done: %s", summary)
    except Exception as exc:
        logger.warning("[ShionMemoryVector] background sync failed: %s", exc)
    finally:
        # 実行中フラグを下ろす。改訂で索引が再び進んだ時に再同期できるようにする
        # （旧実装は「起動後1回だけ」で、改訂後の同期漏れを永久に拾えなかった）。
        _background_sync_started = False


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
        if index_sync_is_stale(_resolve_index_path_safe()):
            # 索引だけ進んでいる（改訂で後継記憶が増えた等）。この状態で
            # 埋め込みブーストを返すと、コレクションに残っている改訂前の旧結論だけが
            # 加点され、後継記憶が沈む。再同期を促し、今回はキーワードのみで想起する。
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
