#!/usr/bin/env python3
"""日本語RAG向け埋め込みモデル比較。

現行のローカル埋め込み（paraphrase-multilingual-MiniLM-L12-v2）に対して、
- Cloudflare Workers AI の日本語・多言語埋め込みモデル
  - `@cf/pfnet/plamo-embedding-1b`（2,048次元のためVectorize非互換）
  - `@cf/qwen/qwen3-embedding-0.6b`（1,024次元）
  - `@cf/baai/bge-m3`（1,024次元）
- Gemini 埋め込みAPI `gemini-embedding-001`（手持ちの GEMINI_API_KEY で実行可能）
- ローカル大型モデル `intfloat/multilingual-e5-large`（APIキー不要・無料）
を `api/knowledge/rag_eval_set.json` のケースで比較する
（hit@k / MRR / forbidden率 / 埋め込み時間 / 概算コスト）。

使い方（Obsidian Vault があるローカル環境で実行）:

    python3 scripts/compare_embedding_models.py                    # local + Workers AI 3モデル
    python3 scripts/compare_embedding_models.py --models local,gemini
    python3 scripts/compare_embedding_models.py --vault "/path/to/Obsidian Vault"

- Workers AI比較は、個人情報や生チャットを除外済みの
  `data/agent_search/lease_knowledge_export/` だけを外部送信する
- CLOUDFLARE_API_TOKEN は環境変数、または macOSキーチェーンの
  service=cloudflare-api-token / account=tune-lease-55 から読む
- CLOUDFLARE_ACCOUNT_ID は環境変数を優先し、無ければ tune77.com のZone情報から解決する
- GEMINI_API_KEY は環境変数 または .streamlit/secrets.toml から読む
- e5-large は初回に約2.2GBのモデルをダウンロードする（以後はローカルキャッシュ）
- 埋め込みは data/embedding_eval_cache/ にキャッシュし、再実行時のAPI課金を防ぐ
- レポートは reports/embedding_model_comparison_<日付>.md に出力する

出典: docs/archive/代替案実装計画_精度改善Ver2.md「代替案B: 埋め込みモデル更新」ステップ1。
計画書は OpenAI text-embedding-3-large を想定していたが、OpenAI APIキーを
保有していないため、比較対象を Gemini とローカル大型モデルに変更した。
また計画書は ada-002 を「現状」としているが、実際の現行モデルはローカル
MiniLM のため比較基準は local とする。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_SET_PATH = REPO_ROOT / "api" / "knowledge" / "rag_eval_set.json"
CACHE_DIR = REPO_ROOT / "data" / "embedding_eval_cache"  # data/ はコミット禁止領域
REPORT_DIR = REPO_ROOT / "reports"
DEFAULT_EXPORT_DIR = REPO_ROOT / "data" / "agent_search" / "lease_knowledge_export"

GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_BATCH_SIZE = 100
CLOUDFLARE_PLAMO_MODEL = "@cf/pfnet/plamo-embedding-1b"
CLOUDFLARE_QWEN3_MODEL = "@cf/qwen/qwen3-embedding-0.6b"
CLOUDFLARE_BGE_M3_MODEL = "@cf/baai/bge-m3"
CLOUDFLARE_BATCH_SIZE = 50
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"
CLOUDFLARE_ZONE_NAME = "tune77.com"
VECTORIZE_MAX_DIMENSIONS = 1536
# 1Mトークンあたりの単価（USD）。埋め込みAPIのusageに依存せず、
# トークン数は文字数からの概算（日本語 ≈ 1文字1トークン強）で表示する。
PRICE_PER_MTOK = {
    GEMINI_EMBED_MODEL: 0.15,
    CLOUDFLARE_PLAMO_MODEL: 0.0186,
    CLOUDFLARE_QWEN3_MODEL: 0.0118,
    CLOUDFLARE_BGE_M3_MODEL: 0.0118,
}

MODEL_ALIASES = {
    "local": "local",  # 現行: paraphrase-multilingual-MiniLM-L12-v2
    "plamo": CLOUDFLARE_PLAMO_MODEL,
    "qwen3": CLOUDFLARE_QWEN3_MODEL,
    "bge-m3": CLOUDFLARE_BGE_M3_MODEL,
    "gemini": GEMINI_EMBED_MODEL,
    "e5-large": "intfloat/multilingual-e5-large",
}

CLOUDFLARE_MODEL_KEYS = frozenset({"plamo", "qwen3", "bge-m3"})

CHUNK_TEXT_LIMIT = 1200  # コスト・実行時間を抑えるためチャンク本文は先頭のみ使う


# ------------------------------------------------------------------
# 純粋関数（テスト対象）
# ------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


def path_matches_any(rel_path: str, patterns: list[str]) -> bool:
    """評価セットの expected/forbidden パターンとの照合（部分一致）。"""
    return any(pattern and pattern in rel_path for pattern in patterns)


def evaluate_case(ranked_paths: list[str], expected: list[str], forbidden: list[str], top_k: int = 5) -> dict:
    """1ケースの検索結果（パスの降順リスト）から指標を計算する。"""
    top = ranked_paths[:top_k]
    first_hit_rank = 0
    for rank, path in enumerate(top, start=1):
        if path_matches_any(path, expected):
            first_hit_rank = rank
            break
    return {
        "hit_at_1": first_hit_rank == 1,
        "hit_at_3": 0 < first_hit_rank <= 3,
        "hit_at_5": 0 < first_hit_rank <= 5,
        "reciprocal_rank": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "forbidden_in_top": any(path_matches_any(path, forbidden) for path in top),
        "first_hit_rank": first_hit_rank,
    }


def summarize(case_results: list[dict]) -> dict:
    n = len(case_results) or 1
    return {
        "cases": len(case_results),
        "hit_at_1": sum(r["hit_at_1"] for r in case_results) / n,
        "hit_at_3": sum(r["hit_at_3"] for r in case_results) / n,
        "hit_at_5": sum(r["hit_at_5"] for r in case_results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in case_results) / n,
        "forbidden_rate": sum(r["forbidden_in_top"] for r in case_results) / n,
    }


def batched(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def estimate_tokens(texts: list[str]) -> int:
    """日本語主体テキストの概算トークン数（1文字 ≈ 1.2トークンで見積もる）。"""
    return int(sum(len(t) for t in texts) * 1.2)


# ------------------------------------------------------------------
# 埋め込みバックエンド
# ------------------------------------------------------------------

def load_api_key(name: str) -> str | None:
    """環境変数 → .streamlit/secrets.toml の順でAPIキーを読む。

    secret_manager.py は streamlit を import するため、ここでは直接読む。
    """
    key = os.environ.get(name)
    if key:
        return key
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import toml

            return toml.load(secrets_path).get(name)
        except Exception as exc:
            print(f"⚠️  secrets.toml の読み込みに失敗: {exc}")
    return None


class GeminiEmbedder:
    """gemini-embedding-001（REST v1beta、プロジェクト内の他API呼び出しと同方式）。"""

    def __init__(self, api_key: str):
        self.model = GEMINI_EMBED_MODEL
        self.api_key = api_key
        self.total_tokens = 0  # APIが usage を返さないため文字数からの概算

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        import requests

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents?key={self.api_key}"
        )
        task_type = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for batch in batched(texts, GEMINI_BATCH_SIZE):
            body = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": task_type,
                    }
                    for text in batch
                ]
            }
            for attempt in range(4):
                try:
                    res = requests.post(url, json=body, timeout=120)
                    if res.status_code == 429 or res.status_code >= 500:
                        raise RuntimeError(f"HTTP {res.status_code}: {res.text[:200]}")
                    res.raise_for_status()
                    embeddings = res.json().get("embeddings") or []
                    if len(embeddings) != len(batch):
                        raise RuntimeError(f"embeddings 件数不一致: {len(embeddings)} != {len(batch)}")
                    vectors.extend(e["values"] for e in embeddings)
                    self.total_tokens += estimate_tokens(batch)
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise
                    wait = 2 ** (attempt + 1)
                    print(f"  ⚠️  リトライ {attempt + 1}/3（{wait}s待機）: {exc}")
                    time.sleep(wait)
        return vectors

    def cost_usd(self) -> float:
        return self.total_tokens / 1_000_000 * PRICE_PER_MTOK.get(self.model, 0.0)


class CloudflareEmbedder:
    """Workers AI埋め込み。プロンプトや応答本文はログに残さない。"""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        model: str = CLOUDFLARE_PLAMO_MODEL,
        post_fn: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if not account_id or not api_token:
            raise ValueError("Cloudflare account ID と API token が必要です")
        if model not in PRICE_PER_MTOK or not model.startswith("@cf/"):
            raise ValueError(f"未対応のWorkers AIモデルです: {model}")
        self.model = model
        self.account_id = account_id
        self.api_token = api_token
        self.total_tokens = 0
        self._post_fn = post_fn
        self._sleep_fn = sleep_fn

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        # 対象モデルはquery/documentで入力形式を分けない。kindは共通interfaceのため受け取る。
        del kind
        import requests

        post = self._post_fn or requests.post
        url = f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/ai/run/{self.model}"
        vectors: list[list[float]] = []
        for batch in batched(texts, CLOUDFLARE_BATCH_SIZE):
            for attempt in range(4):
                response = None
                try:
                    response = post(
                        url,
                        headers={"Authorization": f"Bearer {self.api_token}"},
                        json={"text": batch},
                        timeout=120,
                    )
                    status = int(getattr(response, "status_code", 200))
                    if status == 429 or status >= 500:
                        raise RuntimeError(f"Cloudflare Workers AI HTTP {status}")
                    response.raise_for_status()
                    payload = response.json()
                    result = payload.get("result", payload) if isinstance(payload, dict) else {}
                    embeddings = result.get("data") if isinstance(result, dict) else None
                    if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                        actual = len(embeddings) if isinstance(embeddings, list) else 0
                        raise RuntimeError(f"embeddings 件数不一致: {actual} != {len(batch)}")
                    if any(not isinstance(vector, list) or not vector for vector in embeddings):
                        raise RuntimeError("Workers AI が空または不正な埋め込みを返しました")
                    vectors.extend([[float(value) for value in vector] for vector in embeddings])
                    self.total_tokens += estimate_tokens(batch)
                    break
                except Exception:
                    status = int(getattr(response, "status_code", 0)) if response is not None else 0
                    retryable = response is None or status == 429 or status >= 500
                    if attempt == 3 or not retryable:
                        raise
                    wait = 2 ** (attempt + 1)
                    print(f"  ⚠️  Workers AI リトライ {attempt + 1}/3（{wait}s待機）")
                    self._sleep_fn(wait)
        return vectors

    def cost_usd(self) -> float:
        return self.total_tokens / 1_000_000 * PRICE_PER_MTOK[self.model]


class LocalEmbedder:
    """sentence-transformers 系ローカルモデル。model_name 省略時は現行RAGと同じモデル。"""

    def __init__(self, model_name: str = ""):
        from sentence_transformers import SentenceTransformer

        if not model_name:
            from api.knowledge.vector_store import _MODEL_NAME

            model_name = _MODEL_NAME
        self.model = model_name
        # e5 系は "query: " / "passage: " プレフィックスが検索精度の前提
        self._is_e5 = "e5" in model_name.lower()
        self._encoder = SentenceTransformer(model_name)
        self.total_tokens = 0

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        if self._is_e5:
            prefix = "query: " if kind == "query" else "passage: "
            texts = [prefix + t for t in texts]
        return [list(map(float, v)) for v in self._encoder.encode(texts, show_progress_bar=False)]

    def cost_usd(self) -> float:
        return 0.0


# ------------------------------------------------------------------
# コーパス・キャッシュ
# ------------------------------------------------------------------

def build_corpus(vault: Path) -> list[dict]:
    from api.knowledge.obsidian_loader import scan_vault

    corpus: list[dict] = []
    seen: set[str] = set()
    for chunk in scan_vault(str(vault)):
        rel = os.path.relpath(chunk.file_path, str(vault)).replace(os.sep, "/")
        key = f"{rel}#{chunk.section}"
        if key in seen:
            continue
        seen.add(key)
        text = f"{chunk.file_name} {chunk.section} {chunk.text}"[:CHUNK_TEXT_LIMIT]
        corpus.append({"key": key, "rel_path": rel, "text": text})
    return corpus


def build_exported_corpus(export_dir: Path) -> list[dict]:
    """外部AI投入用に選別・サニタイズ済みのAgent Search corpusを読む。"""
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"サニタイズ済みmanifestがありません: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = export_dir.resolve()
    corpus: list[dict] = []
    for document in manifest.get("documents") or []:
        output_path = str(document.get("output_path") or "").strip()
        source_path = str(document.get("source_path") or "").strip()
        if not output_path or not source_path:
            continue
        candidate = (export_dir / output_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")[:CHUNK_TEXT_LIMIT]
        if text.strip():
            corpus.append({"key": source_path, "rel_path": source_path, "text": text})
    return corpus


def cached_embed(embedder, label: str, texts: list[str], cache_name: str, kind: str) -> list[list[float]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps([label, kind, texts], ensure_ascii=False).encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{cache_name}_{label.replace('/', '_')}_{digest}.json"
    if cache_path.exists():
        print(f"  ♻️  キャッシュ使用: {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))
    vectors = embedder.embed(texts, kind=kind)
    cache_path.write_text(json.dumps(vectors), encoding="utf-8")
    return vectors


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------

def find_default_vault() -> Path | None:
    try:
        from lease_news_digest import find_vault

        return find_vault()
    except Exception:
        return None


def load_cloudflare_token() -> str | None:
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if token:
        return token
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password", "-s", "cloudflare-api-token",
                "-a", "tune-lease-55", "-w",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_cloudflare_account_id(api_token: str, get_fn: Callable[..., Any] | None = None) -> str | None:
    configured = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if configured:
        return configured
    import requests

    get = get_fn or requests.get
    response = get(
        f"{CLOUDFLARE_API_BASE}/zones",
        params={"name": CLOUDFLARE_ZONE_NAME},
        headers={"Authorization": f"Bearer {api_token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    zones = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(zones, list) or not zones:
        return None
    account = zones[0].get("account") if isinstance(zones[0], dict) else None
    return str((account or {}).get("id") or "").strip() or None


def _build_embedder(
    model_key: str,
    gemini_key: str | None,
    cloudflare_account_id: str | None = None,
    cloudflare_token: str | None = None,
):
    if model_key == "local":
        return LocalEmbedder()
    if model_key in CLOUDFLARE_MODEL_KEYS:
        return CloudflareEmbedder(
            cloudflare_account_id or "",
            cloudflare_token or "",
            model=MODEL_ALIASES[model_key],
        )
    if model_key == "gemini":
        return GeminiEmbedder(gemini_key)
    return LocalEmbedder(MODEL_ALIASES[model_key])


def run(args: argparse.Namespace) -> int:
    cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODEL_ALIASES]
    if unknown:
        print(f"❌ 不明なモデル指定: {unknown}（local / plamo / qwen3 / bge-m3 / gemini / e5-large）")
        return 2

    gemini_key = load_api_key("GEMINI_API_KEY") if "gemini" in model_keys else None
    if "gemini" in model_keys and not gemini_key:
        print("❌ GEMINI_API_KEY が見つかりません（環境変数 or .streamlit/secrets.toml）。")
        print("   Gemini 埋め込みのテストにはキーが必要です。--models local,e5-large ならキー不要です。")
        return 2

    uses_cloudflare = bool(CLOUDFLARE_MODEL_KEYS.intersection(model_keys))
    cloudflare_token = load_cloudflare_token() if uses_cloudflare else None
    cloudflare_account_id = None
    if uses_cloudflare:
        if not cloudflare_token:
            print("❌ CLOUDFLARE_API_TOKEN が見つかりません（環境変数またはmacOSキーチェーン）。")
            return 2
        try:
            cloudflare_account_id = resolve_cloudflare_account_id(cloudflare_token)
        except Exception as exc:
            print(f"❌ Cloudflare Account ID の解決に失敗しました: {type(exc).__name__}")
            return 2
        if not cloudflare_account_id:
            print("❌ CLOUDFLARE_ACCOUNT_ID を解決できませんでした。環境変数で指定してください。")
            return 2

    corpus_source = getattr(args, "corpus_source", "vault")
    if corpus_source == "agent-search-export":
        export_dir = Path(getattr(args, "export_dir", "") or DEFAULT_EXPORT_DIR)
        try:
            corpus = build_exported_corpus(export_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"❌ サニタイズ済みコーパスを読めません: {exc}")
            return 2
        print(f"📚 サニタイズ済みコーパス: {len(corpus)} 文書")
    else:
        if uses_cloudflare and not getattr(args, "allow_unsanitized_cloud_corpus", False):
            print("❌ Workers AIへ未選別Vaultを送信しません。--corpus-source agent-search-export を使ってください。")
            return 2
        vault = Path(args.vault) if args.vault else find_default_vault()
        if not vault or not vault.is_dir():
            print("❌ Obsidian Vault が見つかりません。--vault で指定してください。")
            return 2
        print(f"📚 コーパス構築中: {vault}")
        corpus = build_corpus(vault)
    if not corpus:
        print("❌ 比較対象のコーパスが空です。")
        return 2
    if args.max_chunks and len(corpus) > args.max_chunks:
        corpus = corpus[: args.max_chunks]
    print(f"   → {len(corpus)} チャンク / 評価 {len(cases)} ケース")

    corpus_texts = [c["text"] for c in corpus]
    query_texts = [c["query"] for c in cases]

    results: dict[str, dict] = {}
    for model_key in model_keys:
        model_name = MODEL_ALIASES[model_key]
        print(f"\n🔎 モデル: {model_name}")
        try:
            embedder = _build_embedder(
                model_key,
                gemini_key,
                cloudflare_account_id,
                cloudflare_token,
            )
        except Exception as exc:
            print(f"  ❌ 初期化失敗（スキップ）: {exc}")
            continue

        started = time.time()
        try:
            corpus_vecs = cached_embed(embedder, model_name, corpus_texts, "corpus", kind="document")
            query_vecs = cached_embed(embedder, model_name, query_texts, "queries", kind="query")
        except Exception as exc:
            print(f"  ❌ 埋め込み失敗（スキップ）: {exc}")
            continue
        embed_seconds = time.time() - started

        case_results = []
        per_case_rows = []
        for case, qvec in zip(cases, query_vecs):
            expected_present = any(
                path_matches_any(item["rel_path"], [expected])
                for expected in case["expected_path_any"]
                for item in corpus
            )
            scored = sorted(
                ((cosine_similarity(qvec, cvec), c["rel_path"]) for cvec, c in zip(corpus_vecs, corpus)),
                key=lambda item: -item[0],
            )
            ranked_paths = [path for _score, path in scored[: args.top_k]]
            metrics = evaluate_case(ranked_paths, case["expected_path_any"], case["forbidden_path_any"], args.top_k)
            if expected_present:
                case_results.append(metrics)
            per_case_rows.append({
                "id": case["id"],
                "evaluable": expected_present,
                "first_hit_rank": metrics["first_hit_rank"],
                "top1": ranked_paths[0] if ranked_paths else "",
            })

        summary = summarize(case_results)
        summary["total_cases"] = len(cases)
        summary["coverage"] = len(case_results) / (len(cases) or 1)
        summary["embed_seconds"] = round(embed_seconds, 1)
        summary["dimensions"] = len(corpus_vecs[0]) if corpus_vecs else 0
        summary["vectorize_compatible"] = 0 < summary["dimensions"] <= VECTORIZE_MAX_DIMENSIONS
        if model_key in CLOUDFLARE_MODEL_KEYS or model_key == "gemini":
            # キャッシュ有無に左右されない、新規インデックス構築1回分の比較値。
            summary["api_tokens_est"] = estimate_tokens(corpus_texts + query_texts)
            summary["api_cost_usd_est"] = round(
                summary["api_tokens_est"] / 1_000_000 * PRICE_PER_MTOK[model_name],
                4,
            )
        else:
            summary["api_tokens_est"] = 0
            summary["api_cost_usd_est"] = 0.0
        results[model_name] = {"summary": summary, "cases": per_case_rows}
        print(
            f"  評価可能 {summary['cases']}/{summary['total_cases']} / "
            f"hit@1 {summary['hit_at_1']:.0%} / hit@3 {summary['hit_at_3']:.0%} / "
            f"hit@5 {summary['hit_at_5']:.0%} / MRR {summary['mrr']:.3f} / "
            f"forbidden {summary['forbidden_rate']:.0%} / {summary['embed_seconds']}s / "
            f"${summary['api_cost_usd_est']}(概算)"
        )

    if not results:
        print("❌ 実行できたモデルがありません。")
        return 1

    report_path = write_report(results, len(corpus), corpus_source=corpus_source)
    print(f"\n📝 レポート出力: {report_path}")
    return 0


def write_report(results: dict[str, dict], corpus_size: int, *, corpus_source: str = "vault") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    path = REPORT_DIR / f"embedding_model_comparison_{today}.md"
    lines = [
        "# 埋め込みモデル比較レポート（代替案B ステップ1）",
        "",
        f"- 実行日: {datetime.date.today().isoformat()}",
        f"- コーパス: {corpus_size} チャンク / 評価セット: api/knowledge/rag_eval_set.json",
        f"- コーパス経路: {corpus_source}",
        "- トークン・コストは、キャッシュを使わず新規インデックスを1回構築する場合の文字数ベース概算",
        f"- Vectorize互換判定: {VECTORIZE_MAX_DIMENSIONS}次元以下",
        "",
        "| モデル | 次元 | Vectorize | 評価可能 | hit@1 | hit@3 | hit@5 | MRR | forbidden率 | 今回時間(cache含む) | 概算トークン | 概算コスト(USD) |",
        "|--------|------|-----------|----------|-------|-------|-------|-----|------------|------------|------------|----------------|",
    ]
    for model, data in results.items():
        s = data["summary"]
        lines.append(
            f"| {model} | {s['dimensions']} | {'可' if s['vectorize_compatible'] else '不可'} | "
            f"{s['cases']}/{s['total_cases']} ({s['coverage']:.0%}) | "
            f"{s['hit_at_1']:.0%} | {s['hit_at_3']:.0%} | {s['hit_at_5']:.0%} | "
            f"{s['mrr']:.3f} | {s['forbidden_rate']:.0%} | {s['embed_seconds']}s | "
            f"{s['api_tokens_est']} | {s['api_cost_usd_est']} |"
        )
    lines += ["", "## ケース別 first hit rank（0=圏外）", ""]
    for model, data in results.items():
        lines.append(f"### {model}")
        lines.append("")
        for row in data["cases"]:
            if row["evaluable"]:
                lines.append(f"- {row['id']}: rank={row['first_hit_rank']} (top1: {row['top1']})")
            else:
                lines.append(f"- {row['id']}: 対象外（正解文書がサニタイズ済みコーパスに無い）")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vault", default="", help="Obsidian Vault のパス（省略時は自動探索）")
    parser.add_argument(
        "--models",
        default="local,plamo,qwen3,bge-m3",
        help="比較モデル: local,plamo,qwen3,bge-m3,gemini,e5-large",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0, help="コーパス上限（0=無制限）")
    parser.add_argument(
        "--corpus-source",
        choices=("agent-search-export", "vault"),
        default="agent-search-export",
        help="既定は外部AI投入用に選別済みのAgent Search export",
    )
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument(
        "--allow-unsanitized-cloud-corpus",
        action="store_true",
        help="未選別Vaultを外部AIへ送る危険な明示オプション（通常は使用しない）",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
