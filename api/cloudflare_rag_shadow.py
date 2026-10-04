"""ChromaとCloudflare Vectorize/Rerankerの非同期シャドー比較。

回答経路は変更せず、一般的なリース知識質問だけをCloudflareへ送り、
検索順位の差をJSONLへ記録する。個別審査・PII・金額を含む質問は送信しない。
"""
from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

from api.chat_routing import is_potentially_sensitive_screening_message
from api.vertex_query_mask import mask_for_vertex

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "config" / "cloudflare_rag_shadow.json"
_DEFAULT_LOG_PATH = _REPO_ROOT / "data" / "cloudflare_rag_shadow_log.jsonl"
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cf-rag-shadow")
_QUEUE_SLOTS = threading.BoundedSemaphore(2)
_LOG_LOCK = threading.Lock()
_SAFE_DOMAIN_TERMS = (
    "リース", "再リース", "満了", "残価", "残存価値", "耐用年数", "使用期間",
    "補助金", "保険", "会計", "税", "金利", "料率", "所有権", "契約",
    "メンテナンス", "オペレーティング", "ファイナンス", "リスク", "中途解約",
    "物件", "設備", "動産", "自動車", "機械", "リースバック", "キャッシュフロー",
    "資金繰り", "選択肢", "再販", "中古", "売却", "処分", "償却",
)
_SAFE_DOMAIN_PATTERN = re.compile(
    "|".join(re.escape(term) for term in sorted(_SAFE_DOMAIN_TERMS, key=len, reverse=True))
)
_GENERIC_QUERY_FRAGMENTS = (
    "するとき", "する", "とき", "場合", "前", "後", "注意点", "注意",
    "違い", "意味", "見方", "使い方", "使い分け", "条件", "方法", "理由",
    "原因", "対策", "手順",
)


def _safe_decomposed_query(text: str) -> str:
    """Return allowlisted concepts only when decomposition loses no specific concept."""
    from obsidian_query import split_query_terms

    terms = split_query_terms(text)
    if not terms:
        return ""
    safe_terms: list[str] = []
    for term in terms:
        matched = [match.group(0) for match in _SAFE_DOMAIN_PATTERN.finditer(term)]
        residue = _SAFE_DOMAIN_PATTERN.sub("", term)
        for fragment in _GENERIC_QUERY_FRAGMENTS:
            residue = residue.replace(fragment, "")
        if residue.strip(" -_・"):
            return ""
        for matched_term in matched:
            if matched_term not in safe_terms:
                safe_terms.append(matched_term)
    return " ".join(safe_terms)[:300].strip()


def _load_config() -> dict[str, Any]:
    try:
        body = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        return body if isinstance(body, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def shadow_enabled(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    configured = str(env.get("CLOUDFLARE_RAG_SHADOW_MODE") or "").strip().lower()
    if configured:
        return configured == "shadow"
    if env.get("PYTEST_CURRENT_TEST"):
        return False
    return str(_load_config().get("mode") or "off").strip().lower() == "shadow"


def eligible_shadow_query(
    message: str,
    *,
    question_category: str,
    is_general_response_mode: bool,
) -> tuple[bool, str, str]:
    """外部送信可否と、送信用の短いクエリを返す。"""
    text = str(message or "").strip()
    if not shadow_enabled():
        return False, "disabled", ""
    if question_category != "lease_knowledge" or is_general_response_mode:
        return False, "category_excluded", ""
    if not text or len(text) > 360 or "http://" in text.lower() or "https://" in text.lower():
        return False, "shape_excluded", ""
    if is_potentially_sensitive_screening_message(text):
        return False, "sensitive", ""
    masked = mask_for_vertex(text)
    if masked != text:
        return False, "redaction_required", ""
    # 未知の固有名詞は送らず、同時に意味のある語を落とすクエリは比較対象にしない。
    external_query = _safe_decomposed_query(text)
    if not external_query:
        return False, "semantic_loss", ""
    return True, "eligible", external_query


def _safe_local_refs(local_hits: list[dict[str, Any]], limit: int = 5) -> list[str]:
    refs: list[str] = []
    for hit in local_hits[:limit]:
        ref = str(hit.get("ref") or hit.get("file_name") or "").strip()
        if ref and ref not in refs:
            refs.append(ref[:300])
    return refs


def _normalize_ref(ref: str) -> str:
    value = str(ref or "").strip()
    if value.startswith("[[") and value.endswith("]]" ):
        value = value[2:-2]
    value = value.split("#", 1)[0]
    value = Path(value.strip("/")).name
    return value[:-3] if value.endswith(".md") else value


def _log_path() -> Path:
    configured = str(os.environ.get("CLOUDFLARE_RAG_SHADOW_LOG_PATH") or "").strip()
    return Path(configured).expanduser() if configured else _DEFAULT_LOG_PATH


def _append_log(entry: dict[str, Any]) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def compare_cloudflare_shadow(
    external_query: str,
    local_hits: list[dict[str, Any]],
    *,
    client_factory: Callable[[], tuple[Any, str]] | None = None,
) -> dict[str, Any]:
    """同期比較本体。呼び出し元はバックグラウンドで実行する。"""
    if client_factory is None:
        from scripts.cloudflare_vectorize_shadow import (
            CloudflareEmbedder,
            CloudflareReranker,
            CLOUDFLARE_BGE_M3_MODEL,
            load_client,
        )

        client, token = load_client()
        embedder = CloudflareEmbedder(client.account_id, token, model=CLOUDFLARE_BGE_M3_MODEL)
        reranker = CloudflareReranker(client.account_id, token)
    else:
        client, token = client_factory()
        embedder = token["embedder"]
        reranker = token["reranker"]

    vector = embedder.embed([external_query], kind="query")[0]
    vector_matches = client.query(vector, top_k=10)
    reranked = reranker.rerank(external_query, vector_matches, top_k=5)
    vector_refs = [
        str((match.get("metadata") or {}).get("path") or match.get("id") or "")
        for match in vector_matches[:5]
    ]
    reranker_refs = [
        str((match.get("metadata") or {}).get("path") or match.get("id") or "")
        for match in reranked
    ]
    local_refs = _safe_local_refs(local_hits)
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query_sha256": hashlib.sha256(external_query.encode("utf-8")).hexdigest(),
        "query_preview": external_query[:120],
        "local_refs": local_refs,
        "vectorize_refs": vector_refs,
        "reranker_refs": reranker_refs,
        "local_vectorize_overlap_at_5": len(
            {_normalize_ref(ref) for ref in local_refs}
            & {_normalize_ref(ref) for ref in vector_refs}
        ),
        "vectorize_reranker_changed_top1": bool(vector_refs and reranker_refs and vector_refs[0] != reranker_refs[0]),
        "status": "ok",
    }
    _append_log(entry)
    return entry


def _run_and_release(external_query: str, local_hits: list[dict[str, Any]]) -> None:
    try:
        compare_cloudflare_shadow(external_query, local_hits)
    except Exception as exc:  # shadowは回答経路を絶対に止めない
        _append_log({
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query_sha256": hashlib.sha256(external_query.encode("utf-8")).hexdigest(),
            "status": "error",
            "error_type": type(exc).__name__,
        })
    finally:
        _QUEUE_SLOTS.release()


def submit_cloudflare_shadow(
    message: str,
    local_hits: list[dict[str, Any]],
    *,
    question_category: str,
    is_general_response_mode: bool,
) -> dict[str, Any]:
    eligible, reason, external_query = eligible_shadow_query(
        message,
        question_category=question_category,
        is_general_response_mode=is_general_response_mode,
    )
    if not eligible:
        return {"queued": False, "status": reason}
    if not _QUEUE_SLOTS.acquire(blocking=False):
        return {"queued": False, "status": "queue_full"}
    try:
        _EXECUTOR.submit(_run_and_release, external_query, list(local_hits[:5]))
    except Exception:
        _QUEUE_SLOTS.release()
        return {"queued": False, "status": "submit_error"}
    return {"queued": True, "status": "shadow", "query_sha256": hashlib.sha256(external_query.encode()).hexdigest()}
