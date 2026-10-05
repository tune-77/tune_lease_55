#!/usr/bin/env python3
"""Cloudflare Web Search API と Gemini 検索グラウンディングを同じ質問で並べて比較する（REV-479）。

本番の挙動は変えない評価専用スクリプト。題材は審査の企業調べ（中小・地方企業）と
リースニュース収集。網羅性（件数・ドメイン数）、日本語率、鮮度（本文中の最新日付）、
正確さの代理指標（期待キーワードの一致率）、一次情報の数、レイテンシ、概算費用を出す。

Cloudflare 側は AI Gateway のクレジット残高から引き落とされる（残高0だと 402
web_search_payment_required）。試験費用を数ドル以内に抑えるため、実行前に概算費用を
表示し、--max-cost-usd を超える計画は実行しない。

    python scripts/compare_web_search.py --dry-run
    python scripts/compare_web_search.py --backends gemini,ceramic,linkup --out reports/web_search_compare
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"
CF_PROVIDERS = ("ceramic", "linkup", "exa")
# 1リクエストあたりの単価（USD）。Cloudflare 公式 providers ページの list price（2026-10）。
# Unified Billing のクレジット購入時に別途5%かかる。
CF_PRICE_PER_REQUEST = {"ceramic": 0.25 / 1000, "linkup": 5.0 / 1000, "exa": 7.0 / 1000}
# Gemini 側は検索1回あたりの接地料金（Gemini 3 系 $14/1000 クエリ）+ トークン概算。
# モデルが1回の呼び出しで複数クエリを投げるため web_search_queries 数で数える。
GEMINI_GROUNDING_PER_QUERY = 14.0 / 1000
GEMINI_PRICE_PER_MTOKEN = {"in": 0.30, "out": 2.50}  # flash 系の概算。厳密な請求額ではない。
GEMINI_EST_TOKENS = {"in": 600, "out": 900}

_DATE_RE = re.compile(r"(20\d{2})\s*[年/\-.]\s*(\d{1,2})(?:\s*[月/\-.]\s*(\d{1,2}))?")
_CJK_RE = re.compile(r"[぀-ヿ一-鿿]")


@dataclass(frozen=True)
class Case:
    key: str
    kind: str  # "company" | "news"
    query: str
    expect: tuple[str, ...]  # 正しい結果なら本文に現れるはずの語（正確さの代理指標）


# 企業は公開情報が豊富で取り違えが判定しやすい地方・中小の実在企業を選ぶ（顧客情報は使わない）。
CASES: tuple[Case, ...] = (
    Case("co-nakamura-brace", "company", "中村ブレイス 島根県 会社概要 事業内容", ("大田市", "義肢")),
    Case("co-hamano", "company", "浜野製作所 墨田区 会社概要 事業内容", ("墨田", "金属加工")),
    Case("co-tsubame", "company", "山崎金属工業 燕市 会社概要 事業内容", ("燕", "カトラリー")),
    Case("co-imabari", "company", "池内タオル 今治 会社概要 事業内容", ("今治", "タオル")),
    Case("co-risk", "company", "中小企業 民事再生 申請 2026年 製造業 地方", ("民事再生", "負債")),
    Case("news-accounting", "news", "新リース会計基準 2027年4月 適用 借手 影響", ("2027", "使用権資産")),
    Case("news-industry", "news", "リース取扱高 2026年 リース事業協会 統計", ("リース事業協会", "取扱高")),
    Case("news-subsidy", "news", "ものづくり補助金 2026年 公募 締切", ("公募", "締切")),
)


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def _latest_date(text: str, today: dt.date) -> str:
    latest: dt.date | None = None
    for year, month, day in _DATE_RE.findall(text):
        try:
            found = dt.date(int(year), int(month), int(day or 1))
        except ValueError:
            continue
        if found <= today and (latest is None or found > latest):
            latest = found
    return latest.isoformat() if latest else ""


def score_result(case: Case, text: str, sources: list[dict[str, str]], today: dt.date) -> dict[str, Any]:
    """本文と出典から比較指標を計算する（API呼び出しなし・テスト対象）。"""
    from scripts.auto_research_lease_judgment import _source_quality

    domains = {_domain(item.get("url", "")) for item in sources} - {""}
    cjk = len(_CJK_RE.findall(text))
    hits = [word for word in case.expect if word in text]
    return {
        "result_count": len(sources),
        "unique_domains": len(domains),
        "ja_ratio": round(cjk / max(1, len(re.sub(r"\s", "", text))), 2),
        "latest_date": _latest_date(text, today),
        "expect_hit": f"{len(hits)}/{len(case.expect)}",
        "expect_hit_ratio": round(len(hits) / max(1, len(case.expect)), 2),
        "primary_sources": sum(
            1 for item in sources if _source_quality(item.get("title", ""), item.get("url", "")) == "primary"
        ),
        "text_chars": len(text),
    }


def cloudflare_credentials() -> tuple[str, str]:
    from scripts.compare_embedding_models import load_cloudflare_token, resolve_cloudflare_account_id

    token = load_cloudflare_token() or ""
    if not token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN がありません（キーチェーン cloudflare-api-token / tune-lease-55）")
    account_id = resolve_cloudflare_account_id(token) or ""
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID を解決できません")
    return token, account_id


def cloudflare_search(
    query: str,
    provider: str,
    *,
    token: str,
    account_id: str,
    limit: int = 10,
    gateway_id: str = "",
    post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if post is None:
        import requests

        post = requests.post
    started = time.perf_counter()
    response = post(
        f"{CLOUDFLARE_API_BASE}/accounts/{account_id}/ai/websearch/",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "query": query[:1024],
            "provider": provider,
            "limit": max(1, min(10, limit)),
            "options": {"gateway": {"id": gateway_id or os.environ.get("CLOUDFLARE_AI_GATEWAY_ID", "default")}},
        },
        timeout=60,
    )
    latency = time.perf_counter() - started
    payload = response.json() if response.content else {}
    if response.status_code != 200:
        error = payload.get("error") or payload.get("errors") or payload
        raise RuntimeError(f"Cloudflare web search {response.status_code}: {json.dumps(error, ensure_ascii=False)[:300]}")
    body = payload.get("result", payload) if isinstance(payload, dict) else {}
    items = [item for item in body.get("items") or [] if isinstance(item, dict)]
    sources = [{"title": str(item.get("title") or ""), "url": str(item.get("url") or "")} for item in items]
    text = "\n".join(f"{item.get('title') or ''}\n{item.get('description') or ''}" for item in items)
    return {
        "text": text,
        "sources": sources,
        "latency_s": round(latency, 2),
        "server_latency_ms": (body.get("metadata") or {}).get("latencyMs"),
        "cost_usd": CF_PRICE_PER_REQUEST[provider],
    }


def gemini_search(query: str) -> dict[str, Any]:
    """紫苑の既存グラウンディング（api.vertex_agent_search.google_search_grounding）をそのまま呼ぶ。"""
    from api.vertex_agent_search import google_search_grounding

    started = time.perf_counter()
    result = google_search_grounding(query)
    latency = time.perf_counter() - started
    if result.get("status") != "ok":
        raise RuntimeError(f"Gemini grounding {result.get('status')}: {result.get('error', '')}")
    queries = len(result.get("web_search_queries") or []) or 1
    token_cost = sum(GEMINI_EST_TOKENS[k] * GEMINI_PRICE_PER_MTOKEN[k] / 1_000_000 for k in GEMINI_EST_TOKENS)
    return {
        "text": str(result.get("text") or ""),
        # 出典URLは vertexaisearch のリダイレクトURLで、実ドメインは title 側に入る。
        "sources": [
            {"title": str(item.get("title") or ""), "url": f"https://{item.get('title')}" if item.get("title") else ""}
            for item in result.get("sources") or []
        ],
        "latency_s": round(latency, 2),
        "web_search_queries": queries,
        "cost_usd": round(queries * GEMINI_GROUNDING_PER_QUERY + token_cost, 5),
    }


def estimate_cost(backends: list[str], cases: int) -> float:
    per_case = 0.0
    for backend in backends:
        if backend == "gemini":
            per_case += 2 * GEMINI_GROUNDING_PER_QUERY  # 平均2クエリ想定
        else:
            per_case += CF_PRICE_PER_REQUEST[backend]
    return round(per_case * cases, 4)


def render_markdown(rows: list[dict[str, Any]], backends: list[str], today: dt.date) -> str:
    lines = [f"# Web検索 比較（{today.isoformat()}）", ""]
    lines.append("| backend | 平均遅延(s) | 件数 | ドメイン | 日本語率 | 期待語一致 | 一次情報 | 最新日付あり | 費用/件($) | 失敗 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for backend in backends:
        mine = [row for row in rows if row["backend"] == backend]
        ok = [row for row in mine if not row.get("error")]
        if not ok:
            lines.append(f"| {backend} | - | - | - | - | - | - | - | - | {len(mine)} |")
            continue

        def avg(key: str) -> float:
            return round(sum(float(row[key]) for row in ok) / len(ok), 2)

        dated = sum(1 for row in ok if row["latest_date"])
        lines.append(
            f"| {backend} | {avg('latency_s')} | {avg('result_count')} | {avg('unique_domains')} | {avg('ja_ratio')} "
            f"| {avg('expect_hit_ratio')} | {avg('primary_sources')} | {dated}/{len(ok)} | {avg('cost_usd')} "
            f"| {len(mine) - len(ok)} |"
        )
    lines += ["", "## ケース別", "", "| case | backend | 遅延 | 件数 | 期待語 | 最新日付 | 上位ドメイン / エラー |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        detail = row.get("error") or ", ".join(row.get("top_domains") or [])
        lines.append(
            f"| {row['case']} | {row['backend']} | {row.get('latency_s', '-')} | {row.get('result_count', '-')} "
            f"| {row.get('expect_hit', '-')} | {row.get('latest_date', '') or '-'} | {detail[:160]} |"
        )
    return "\n".join(lines) + "\n"


def run(backends: list[str], cases: tuple[Case, ...], limit: int) -> list[dict[str, Any]]:
    today = dt.date.today()
    credentials: tuple[str, str] | None = None
    rows: list[dict[str, Any]] = []
    for case in cases:
        for backend in backends:
            row: dict[str, Any] = {"case": case.key, "kind": case.kind, "backend": backend, "query": case.query}
            try:
                if backend == "gemini":
                    result = gemini_search(case.query)
                else:
                    if credentials is None:
                        credentials = cloudflare_credentials()
                    token, account_id = credentials
                    result = cloudflare_search(case.query, backend, token=token, account_id=account_id, limit=limit)
                row.update({key: value for key, value in result.items() if key not in {"text", "sources"}})
                row.update(score_result(case, result["text"], result["sources"], today))
                row["top_domains"] = [_domain(item["url"]) for item in result["sources"][:5]]
                row["sources"] = result["sources"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                # 残高不足は全件同じ結果になるので、その backend の残りは打たない。
                if "payment_required" in row["error"]:
                    backends = [item for item in backends if item != backend]
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("case", "backend", "latency_s", "result_count", "expect_hit", "error")}, ensure_ascii=False), file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backends", default="gemini,ceramic,linkup", help="gemini,ceramic,linkup,exa から選ぶ")
    parser.add_argument("--cases", default="", help="case key をカンマ区切り（既定は全件）")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--out", default="", help="結果を書くディレクトリ（JSON と Markdown）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    unknown = [item for item in backends if item != "gemini" and item not in CF_PROVIDERS]
    if unknown:
        parser.error(f"unknown backend: {unknown}")
    wanted = {item.strip() for item in args.cases.split(",") if item.strip()}
    cases = tuple(case for case in CASES if not wanted or case.key in wanted)
    estimate = estimate_cost(backends, len(cases))
    print(f"[plan] cases={len(cases)} backends={backends} estimated_cost_usd={estimate}", file=sys.stderr)
    if args.dry_run:
        return 0
    if estimate > args.max_cost_usd:
        print(f"ERROR: 概算 ${estimate} が上限 ${args.max_cost_usd} を超えます", file=sys.stderr)
        return 2

    rows = run(backends, cases, args.limit)
    today = dt.date.today()
    report = render_markdown(rows, backends, today)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        (out / f"web_search_compare_{stamp}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"web_search_compare_{stamp}.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
