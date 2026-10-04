#!/usr/bin/env python3
"""匿名化済みリース知識をWorkers AI + Vectorizeでシャドー検索する。

本番ChromaDBには接続せず、`lease_knowledge_export` の選別済み文書だけを扱う。
API tokenは環境変数またはmacOSキーチェーンから読み、ファイルへ保存しない。

例:
    python scripts/cloudflare_vectorize_shadow.py status
    python scripts/cloudflare_vectorize_shadow.py sync --apply
    python scripts/cloudflare_vectorize_shadow.py query "再リースの注意点" --confirm-sanitized-query
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_embedding_models import (  # noqa: E402
    CLOUDFLARE_API_BASE,
    CLOUDFLARE_BGE_M3_MODEL,
    DEFAULT_EXPORT_DIR,
    EVAL_SET_PATH,
    CloudflareEmbedder,
    build_exported_corpus,
    cached_embed,
    evaluate_case,
    load_cloudflare_token,
    resolve_cloudflare_account_id,
    summarize,
)

INDEX_NAME = "tune-lease-rag-bge-m3-shadow-v1"
DIMENSIONS = 1024
METRIC = "cosine"
UPSERT_BATCH_SIZE = 500
DELETE_BATCH_SIZE = 1000
LIST_PAGE_SIZE = 1000
RERANKER_MODEL = "@cf/baai/bge-reranker-base"


def stable_vector_id(source_path: str) -> str:
    """Vectorizeの64-byte制約内で再同期可能な決定的IDを作る。"""
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:32]


def build_vector_records(corpus: list[dict], vectors: list[list[float]]) -> list[dict]:
    if len(corpus) != len(vectors):
        raise ValueError(f"文書とベクトルの件数不一致: {len(corpus)} != {len(vectors)}")
    records: list[dict] = []
    for document, vector in zip(corpus, vectors):
        if len(vector) != DIMENSIONS:
            raise ValueError(f"ベクトル次元不一致: {len(vector)} != {DIMENSIONS}")
        source_path = str(document["rel_path"])
        record_key = str(document.get("key") or source_path)
        records.append({
            "id": stable_vector_id(record_key),
            "values": vector,
            "metadata": {
                "path": source_path,
                "text": str(document["text"])[:1200],
                "corpus": "lease-knowledge-sanitized-v1",
            },
        })
    return records


def stale_vector_ids(existing_ids: list[str], records: list[dict]) -> list[str]:
    """専用indexに残っている、現行の匿名化exportに存在しないIDを返す。"""
    current_ids = {str(record.get("id") or "") for record in records}
    return sorted({str(vector_id) for vector_id in existing_ids if str(vector_id)} - current_ids)


def mutation_is_processed(info: dict, mutation_id: str, expected_count: int) -> bool:
    """最後のmutationが反映済みで、index件数もexportと一致するか確認する。"""
    processed = str(
        info.get("processedUpToMutation")
        or info.get("processed_up_to_mutation")
        or ""
    )
    count = int(info.get("vectorCount", info.get("vector_count", 0)) or 0)
    return bool(mutation_id) and processed == mutation_id and count == expected_count


def _error_message(payload: Any, status_code: int) -> str:
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if isinstance(errors, list):
        messages = [str(item.get("message") or "") for item in errors if isinstance(item, dict)]
        if any(messages):
            return f"Cloudflare API HTTP {status_code}: {'; '.join(filter(None, messages))}"
    return f"Cloudflare API HTTP {status_code}"


class VectorizeClient:
    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        request_fn: Callable[..., Any] | None = None,
    ):
        if not account_id or not api_token:
            raise ValueError("Cloudflare account ID と API token が必要です")
        self.account_id = account_id
        self.api_token = api_token
        self._request_fn = request_fn

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def _request(self, method: str, path: str, **kwargs) -> tuple[int, dict]:
        import requests

        request = self._request_fn or requests.request
        response = request(
            method,
            f"{CLOUDFLARE_API_BASE}{path}",
            headers={**self._headers, **kwargs.pop("headers", {})},
            timeout=kwargs.pop("timeout", 120),
            **kwargs,
        )
        status = int(response.status_code)
        try:
            payload = response.json()
        except Exception:
            payload = {}
        if status >= 400 or (isinstance(payload, dict) and payload.get("success") is False):
            raise RuntimeError(_error_message(payload, status))
        return status, payload if isinstance(payload, dict) else {}

    def get_index(self, index_name: str = INDEX_NAME) -> dict | None:
        try:
            _status, payload = self._request(
                "GET",
                f"/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
                timeout=30,
            )
            return payload.get("result") or {}
        except RuntimeError as exc:
            # Cloudflareの存在しないindexは通常404/コード1001。その他は隠さない。
            if "HTTP 404" in str(exc):
                return None
            raise

    def ensure_index(self, index_name: str = INDEX_NAME) -> tuple[dict, bool]:
        current = self.get_index(index_name)
        if current is not None:
            config = current.get("config") or {}
            if int(config.get("dimensions") or 0) != DIMENSIONS or config.get("metric") != METRIC:
                raise RuntimeError(
                    f"既存index設定が不一致です: {config}（必要: {DIMENSIONS}/{METRIC}）"
                )
            return current, False
        _status, payload = self._request(
            "POST",
            f"/accounts/{self.account_id}/vectorize/v2/indexes",
            json={
                "name": index_name,
                "description": "Sanitized lease knowledge shadow index; BGE-M3; no production traffic",
                "config": {"dimensions": DIMENSIONS, "metric": METRIC},
            },
        )
        return payload.get("result") or {}, True

    def upsert(self, records: list[dict], index_name: str = INDEX_NAME) -> list[str]:
        mutation_ids: list[str] = []
        for start in range(0, len(records), UPSERT_BATCH_SIZE):
            batch = records[start:start + UPSERT_BATCH_SIZE]
            ndjson = "\n".join(json.dumps(row, ensure_ascii=False) for row in batch) + "\n"
            _status, payload = self._request(
                "POST",
                f"/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}/upsert",
                # Vectorize REST APIはmultipartのフィールド名を `vectors` として要求する。
                files={"vectors": ("vectors.ndjson", ndjson.encode("utf-8"), "application/x-ndjson")},
            )
            mutation_id = str((payload.get("result") or {}).get("mutationId") or "")
            if mutation_id:
                mutation_ids.append(mutation_id)
        return mutation_ids

    def list_vector_ids(self, index_name: str = INDEX_NAME) -> list[str]:
        vector_ids: list[str] = []
        cursor = ""
        while True:
            params: dict[str, Any] = {"count": LIST_PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            _status, payload = self._request(
                "GET",
                f"/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}/list",
                params=params,
                timeout=30,
            )
            result = payload.get("result") or {}
            vector_ids.extend(
                str(item.get("id") or "")
                for item in result.get("vectors") or []
                if isinstance(item, dict) and item.get("id")
            )
            cursor = str(result.get("nextCursor") or "")
            if not result.get("isTruncated") or not cursor:
                return vector_ids

    def delete_by_ids(self, vector_ids: list[str], index_name: str = INDEX_NAME) -> list[str]:
        mutation_ids: list[str] = []
        for start in range(0, len(vector_ids), DELETE_BATCH_SIZE):
            batch = [str(vector_id) for vector_id in vector_ids[start:start + DELETE_BATCH_SIZE] if vector_id]
            if not batch:
                continue
            _status, payload = self._request(
                "POST",
                f"/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}/delete_by_ids",
                json={"ids": batch},
            )
            mutation_id = str((payload.get("result") or {}).get("mutationId") or "")
            if mutation_id:
                mutation_ids.append(mutation_id)
        return mutation_ids

    def info(self, index_name: str = INDEX_NAME) -> dict:
        _status, payload = self._request(
            "GET",
            f"/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}/info",
            timeout=30,
        )
        return payload.get("result") or {}

    def query(self, vector: list[float], top_k: int = 5, index_name: str = INDEX_NAME) -> list[dict]:
        if len(vector) != DIMENSIONS:
            raise ValueError(f"検索ベクトル次元不一致: {len(vector)} != {DIMENSIONS}")
        _status, payload = self._request(
            "POST",
            f"/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}/query",
            json={
                "vector": vector,
                "topK": max(1, min(int(top_k), 20)),
                "returnMetadata": "all",
                "returnValues": False,
            },
        )
        return list((payload.get("result") or {}).get("matches") or [])


class CloudflareReranker:
    """Vectorize候補を質問との直接関連度で再ランキングする。"""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        post_fn: Callable[..., Any] | None = None,
    ):
        if not account_id or not api_token:
            raise ValueError("Cloudflare account ID と API token が必要です")
        self.account_id = account_id
        self.api_token = api_token
        self._post_fn = post_fn

    def rerank(self, query_text: str, matches: list[dict], top_k: int = 5) -> list[dict]:
        import requests

        candidates = [
            match for match in matches
            if str((match.get("metadata") or {}).get("text") or "").strip()
        ]
        if not candidates:
            return []
        post = self._post_fn or requests.post
        response = post(
            f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/ai/run/{RERANKER_MODEL}",
            headers={"Authorization": f"Bearer {self.api_token}"},
            json={
                "query": query_text,
                "contexts": [{"text": (match.get("metadata") or {})["text"]} for match in candidates],
                "top_k": max(1, min(int(top_k), len(candidates))),
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result", payload) if isinstance(payload, dict) else {}
        scores = result.get("response") if isinstance(result, dict) else None
        if not isinstance(scores, list):
            raise RuntimeError("Workers AI rerankerの応答形式が不正です")

        ranked: list[dict] = []
        for item in scores:
            if not isinstance(item, dict):
                continue
            try:
                candidate_index = int(item["id"])
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= candidate_index < len(candidates):
                continue
            match = dict(candidates[candidate_index])
            match["rerank_score"] = score
            match["vector_rank"] = matches.index(candidates[candidate_index]) + 1
            ranked.append(match)
        ranked.sort(key=lambda item: -float(item["rerank_score"]))
        if not ranked:
            raise RuntimeError("Workers AI rerankerが有効なスコアを返しませんでした")
        return ranked[: max(1, min(int(top_k), len(ranked)))]


def load_client() -> tuple[VectorizeClient, str]:
    token = load_cloudflare_token()
    if not token:
        raise RuntimeError("CLOUDFLARE_API_TOKENが見つかりません")
    account_id = resolve_cloudflare_account_id(token)
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_IDを解決できません")
    return VectorizeClient(account_id, token), token


def command_status(client: VectorizeClient) -> int:
    index = client.get_index()
    if index is None:
        print(f"未作成: {INDEX_NAME}")
        return 0
    info = client.info()
    config = index.get("config") or {}
    print(
        f"index={INDEX_NAME} dimensions={config.get('dimensions')} metric={config.get('metric')} "
        f"vectors={info.get('vectorCount', info.get('vector_count', 'unknown'))}"
    )
    return 0


def command_sync(client: VectorizeClient, token: str, *, apply: bool) -> int:
    if not apply:
        print("dry-run: 外部indexは変更していません。実行には sync --apply を指定してください。")
        return 0
    corpus = build_exported_corpus(DEFAULT_EXPORT_DIR)
    if not corpus:
        raise RuntimeError("匿名化済みコーパスが空です")
    embedder = CloudflareEmbedder(client.account_id, token, model=CLOUDFLARE_BGE_M3_MODEL)
    vectors = cached_embed(
        embedder,
        CLOUDFLARE_BGE_M3_MODEL,
        [item["text"] for item in corpus],
        "corpus",
        kind="document",
    )
    records = build_vector_records(corpus, vectors)
    _index, created = client.ensure_index()
    existing_ids = [] if created else client.list_vector_ids()
    removed_ids = stale_vector_ids(existing_ids, records)
    delete_mutations = client.delete_by_ids(removed_ids)
    mutation_ids = client.upsert(records)
    submitted_mutations = delete_mutations + mutation_ids
    if not submitted_mutations:
        raise RuntimeError("Vectorize APIがmutation IDを返さなかったため、反映確認できません")
    final_mutation_id = submitted_mutations[-1]
    print(
        f"{'作成' if created else '再利用'} index={INDEX_NAME} / "
        f"delete={len(removed_ids)} / upsert={len(records)} / "
        f"mutations={len(submitted_mutations)}"
    )
    # 件数は更新前から同じ場合があるため、最後のmutation IDの処理完了も確認する。
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        info = client.info()
        if mutation_is_processed(info, final_mutation_id, len(records)):
            print(f"反映確認: {len(records)} vectors / mutation={final_mutation_id}")
            return 0
        time.sleep(2)
    print("反映待ち: Vectorizeの非同期処理は継続中です。statusで再確認してください。")
    return 0


def command_query(
    client: VectorizeClient,
    token: str,
    query_text: str,
    top_k: int,
    candidate_k: int,
    use_reranker: bool,
) -> int:
    if not query_text.strip():
        raise ValueError("検索文が空です")
    embedder = CloudflareEmbedder(client.account_id, token, model=CLOUDFLARE_BGE_M3_MODEL)
    vector = embedder.embed([query_text], kind="query")[0]
    candidate_k = max(top_k, min(int(candidate_k), 20))
    matches = client.query(vector, top_k=candidate_k)
    if use_reranker:
        reranker = CloudflareReranker(client.account_id, token)
        matches = reranker.rerank(query_text, matches, top_k=top_k)
    else:
        matches = matches[:top_k]
    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        if use_reranker:
            print(
                f"{rank}. rerank={float(match.get('rerank_score') or 0):.4f} "
                f"vector_rank={match.get('vector_rank', '?')} "
                f"vector_score={float(match.get('score') or 0):.4f} "
                f"{metadata.get('path', match.get('id', ''))}"
            )
        else:
            print(
                f"{rank}. vector_score={float(match.get('score') or 0):.4f} "
                f"{metadata.get('path', match.get('id', ''))}"
            )
    return 0


def command_evaluate(client: VectorizeClient, token: str, top_k: int, candidate_k: int) -> int:
    corpus = build_exported_corpus(DEFAULT_EXPORT_DIR)
    paths = [str(item["rel_path"]) for item in corpus]
    cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    evaluable = [
        case for case in cases
        if any(expected in path for expected in case["expected_path_any"] for path in paths)
    ]
    embedder = CloudflareEmbedder(client.account_id, token, model=CLOUDFLARE_BGE_M3_MODEL)
    reranker = CloudflareReranker(client.account_id, token)
    base_results: list[dict] = []
    rerank_results: list[dict] = []
    candidate_k = max(top_k, min(int(candidate_k), 20))
    for case in evaluable:
        vector = embedder.embed([case["query"]], kind="query")[0]
        matches = client.query(vector, top_k=candidate_k)
        base_paths = [str((match.get("metadata") or {}).get("path") or "") for match in matches[:top_k]]
        reranked = reranker.rerank(case["query"], matches, top_k=top_k)
        reranked_paths = [str((match.get("metadata") or {}).get("path") or "") for match in reranked]
        base_results.append(evaluate_case(base_paths, case["expected_path_any"], case["forbidden_path_any"], top_k))
        rerank_results.append(
            evaluate_case(reranked_paths, case["expected_path_any"], case["forbidden_path_any"], top_k)
        )
        print(
            f"{case['id']}: vector={base_results[-1]['first_hit_rank']} "
            f"reranker={rerank_results[-1]['first_hit_rank']}"
        )
    base = summarize(base_results)
    reranked = summarize(rerank_results)
    print(
        f"Vectorize: cases={base['cases']} hit@1={base['hit_at_1']:.0%} "
        f"hit@5={base['hit_at_5']:.0%} MRR={base['mrr']:.3f}"
    )
    print(
        f"Reranker: cases={reranked['cases']} hit@1={reranked['hit_at_1']:.0%} "
        f"hit@5={reranked['hit_at_5']:.0%} MRR={reranked['mrr']:.3f}"
    )
    if (
        reranked["hit_at_1"] >= base["hit_at_1"]
        and reranked["hit_at_5"] >= base["hit_at_5"]
        and reranked["mrr"] >= base["mrr"] + 0.02
    ):
        print("判定: Reranker採用候補（シャドー継続）")
    else:
        print("判定: Rerankerは既定無効のまま（Vectorize順位を維持）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--apply", action="store_true")
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query_text")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--candidates", type=int, default=10, help="Vectorizeからrerankerへ渡す候補数（最大20）")
    query_parser.add_argument("--rerank", action="store_true", help="BGE Rerankerで候補を並べ替える")
    query_parser.add_argument(
        "--confirm-sanitized-query",
        action="store_true",
        help="検索文に個人情報・案件情報・秘密情報がないことを確認する",
    )
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--candidates", type=int, default=10)
    eval_parser.add_argument(
        "--confirm-sanitized-query",
        action="store_true",
        help="評価セットに個人情報・案件情報・秘密情報がないことを確認する",
    )
    args = parser.parse_args()

    if args.command in {"query", "evaluate"} and not args.confirm_sanitized_query:
        print("❌ Cloudflareへ送る検索文に機密情報がないことを確認し、--confirm-sanitized-query を指定してください。")
        return 2

    try:
        client, token = load_client()
        if args.command == "status":
            return command_status(client)
        if args.command == "sync":
            return command_sync(client, token, apply=args.apply)
        if args.command == "evaluate":
            return command_evaluate(client, token, args.top_k, args.candidates)
        return command_query(client, token, args.query_text, args.top_k, args.candidates, args.rerank)
    except Exception as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
