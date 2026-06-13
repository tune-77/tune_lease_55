"""Investigation tools that Shion can call during dialogue to actually research things."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

_LEASE_WIKI_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "lease-wiki-vault"
)
_WIKI_CACHE_PATH = get_data_path("wiki_embedding_cache.json")

DB_PATH = get_data_path("lease_data.db")


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def search_cases(query: str, limit: int = 5) -> dict[str, Any]:
    """Search recent screening cases by company name or any field in input_snapshot."""
    like = f"%{query}%"
    with closing(_open_db()) as conn:
        rows = conn.execute(
            """
            SELECT case_id, screened_at, total_score, asset_score, tenant_score,
                   q_risk_score, outcome, input_snapshot
            FROM screening_records
            WHERE case_id LIKE ? OR input_snapshot LIKE ?
            ORDER BY screened_at DESC
            LIMIT ?
            """,
            (like, like, max(1, min(limit, 20))),
        ).fetchall()
    results = []
    for r in rows:
        snap: dict = {}
        try:
            snap = json.loads(r["input_snapshot"] or "{}")
        except Exception:
            pass
        results.append({
            "case_id": r["case_id"],
            "screened_at": r["screened_at"],
            "total_score": r["total_score"],
            "asset_score": r["asset_score"],
            "tenant_score": r["tenant_score"],
            "q_risk_score": r["q_risk_score"],
            "outcome": r["outcome"],
            "company_name": snap.get("company_name", r["case_id"]),
        })
    return {"cases": results, "count": len(results)}


def get_score_detail(company_name: str) -> dict[str, Any]:
    """Get latest score detail for a company with factor breakdown and risk flags."""
    like = f"%{company_name}%"
    with closing(_open_db()) as conn:
        row = conn.execute(
            """
            SELECT case_id, screened_at, total_score, asset_score, tenant_score,
                   q_risk_score, competitor_pressure_score, outcome, input_snapshot
            FROM screening_records
            WHERE case_id LIKE ? OR input_snapshot LIKE ?
            ORDER BY screened_at DESC
            LIMIT 1
            """,
            (like, like),
        ).fetchone()
    if not row:
        return {"found": False, "company_name": company_name}

    snap: dict = {}
    try:
        snap = json.loads(row["input_snapshot"] or "{}")
    except Exception:
        pass

    total = float(row["total_score"] or 0)
    asset = float(row["asset_score"] or 0)
    tenant = float(row["tenant_score"] or 0)
    q_risk = float(row["q_risk_score"] or 0)

    if total >= 70:
        verdict = "承認"
    elif total >= 60:
        verdict = "条件付き承認"
    else:
        verdict = "否決"

    risk_flags: list[str] = []
    if q_risk >= 60:
        risk_flags.append("信用リスク強警戒（Q_risk≥60）")
    elif q_risk >= 35:
        risk_flags.append("信用リスク要注意（Q_risk≥35）")
    if asset < 40:
        risk_flags.append("物件スコア低（asset<40）")
    if tenant < 40:
        risk_flags.append("借手スコア低（tenant<40）")

    # Pull a few representative input fields for context
    input_summary = {
        k: snap[k]
        for k in (
            "company_name", "industry_sub", "annual_revenue", "employees",
            "years_in_business", "asset_type", "lease_amount",
        )
        if k in snap
    }

    return {
        "found": True,
        "company_name": snap.get("company_name", row["case_id"]),
        "case_id": row["case_id"],
        "screened_at": row["screened_at"],
        "total_score": total,
        "asset_score": asset,
        "tenant_score": tenant,
        "q_risk_score": q_risk,
        "competitor_pressure_score": row["competitor_pressure_score"],
        "outcome": row["outcome"],
        "verdict": verdict,
        "risk_flags": risk_flags,
        "input_summary": input_summary,
    }


def compare_similar_cases(industry: str, score_min: float = 0.0, score_max: float = 100.0) -> dict[str, Any]:
    """Compare cases in a similar industry and score range to find patterns."""
    like = f"%{industry}%" if industry else "%"
    with closing(_open_db()) as conn:
        rows = conn.execute(
            """
            SELECT industry_sub, score, final_status, data
            FROM past_cases
            WHERE industry_sub LIKE ?
              AND score BETWEEN ? AND ?
            ORDER BY score DESC
            LIMIT 15
            """,
            (like, score_min, score_max),
        ).fetchall()

    approved = [r for r in rows if (r["final_status"] or "") in ("成約", "承認", "approved")]
    rejected = [r for r in rows if (r["final_status"] or "") in ("失注", "否決", "rejected")]
    avg_score = sum(float(r["score"] or 0) for r in rows) / len(rows) if rows else 0.0

    return {
        "industry": industry,
        "total": len(rows),
        "approved": len(approved),
        "rejected": len(rejected),
        "avg_score": round(avg_score, 1),
        "approval_rate_pct": round(len(approved) / len(rows) * 100, 1) if rows else 0.0,
        "samples": [
            {
                "industry": r["industry_sub"],
                "score": r["score"],
                "status": r["final_status"],
            }
            for r in rows[:5]
        ],
    }


def get_weekly_trend(weeks: int = 4) -> dict[str, Any]:
    """Return recent weekly aggregate stats from weekly_plot.json."""
    path = get_data_path("weekly_plot.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"error": str(e)}

    if isinstance(data, list):
        recent = data[-weeks:] if len(data) >= weeks else data
        return {"weeks": recent, "total_weeks_available": len(data)}
    if isinstance(data, dict):
        return {"data": data}
    return {"raw": str(data)[:500]}


def search_obsidian(query: str, vault: Path, limit: int = 3) -> dict[str, Any]:
    """Search the main Vault through the shared Obsidian RAG route."""
    from obsidian_ai_context import collect_obsidian_ai_context

    result = collect_obsidian_ai_context(
        query,
        limit=max(1, min(limit, 10)),
        max_chars=3200,
        heading="Obsidian業務記録・方針メモ",
    )
    hits = [
        {
            "file": str(hit.get("path") or ""),
            "snippet": str(hit.get("snippet") or "")[:700],
            "source": str(hit.get("source") or ""),
            "score": hit.get("score"),
        }
        for hit in result.get("hits", [])
    ]
    return {
        "results": hits,
        "count": len(hits),
        "query": query,
        "search_route": "obsidian_query -> obsidian_ai_context -> mobile_app.obsidian_bridge",
    }


def inspect_scoring_policy(topic: str = "") -> dict[str, Any]:
    """Return route-aware executable scoring policy."""
    from category_config import ASSET_WEIGHT
    from data_cases import get_score_weights
    from scoring_core import APPROVAL_LINE

    borrower_weight, asset_weight, _, _ = get_score_weights()
    category_weights = {
        category: {
            "asset_weight": float(config.get("asset_w", 0.0)),
            "obligor_weight": float(config.get("obligor_w", 0.0)),
        }
        for category, config in ASSET_WEIGHT.items()
    }
    facts = {
        "approval_line": APPROVAL_LINE,
        "requires_route_identification": True,
        "routes": {
            "quick_batch_scoring_core": {
                "asset_score_affects_final_score": False,
                "base_score_source": "score_borrower",
                "role": "warning_and_display",
                "source": "scoring_core.py",
            },
            "next_full_api": {
                "endpoint": "/api/score/full",
                "asset_score_affects_final_score": True,
                "uncategorized_weights": {
                    "asset_weight": round(float(asset_weight), 4),
                    "borrower_weight": round(float(borrower_weight), 4),
                },
                "category_weights": category_weights,
                "dynamic_asset_weight_cap": 0.5,
                "source": "components/score_calculation.py",
            },
        },
    }
    return {
        "topic": topic,
        "status": "current_implementation_route_split",
        "facts": facts,
        "explanation": (
            "現行実装は経路で異なる。scoring_coreを使う簡易・バッチ経路では"
            "asset_scoreを最終点へ直接加算しない。一方、Next主要画面の"
            "/api/score/full では物件スコアを借手側スコアと加重合成する。"
            "承認理由を説明する前に、対象案件が通ったAPI経路を特定する必要がある。"
        ),
        "sources": [
            "scoring_core.py: base_score = score_borrower",
            "frontend/src/app/page.tsx: POST /api/score/full",
            "components/score_calculation.py: weighted asset and obligor score",
            "category_config.py: ASSET_WEIGHT",
        ],
    }


# ── Wiki embedding helpers ────────────────────────────────────────────────────

def _gemini_api_key_for_tools() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    here = Path(__file__).parent
    for _ in range(4):
        sec = here / ".streamlit" / "secrets.toml"
        if sec.exists():
            for line in sec.read_text(encoding="utf-8").splitlines():
                m = re.match(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']', line.strip())
                if m:
                    return m.group(1)
        here = here.parent
    return ""


def _embed_text(text: str, api_key: str) -> list[float]:
    import requests
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    payload = {"content": {"parts": [{"text": text[:8000]}]}}
    resp = requests.post(url, json=payload, headers={"x-goog-api-key": api_key}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _load_wiki_cache() -> dict:
    try:
        if Path(_WIKI_CACHE_PATH).exists():
            return json.loads(Path(_WIKI_CACHE_PATH).read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_wiki_cache(cache: dict) -> None:
    path = Path(_WIKI_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _wiki_numbered_files() -> list[Path]:
    """Return files only from numbered directories (00_–10_) in the wiki."""
    if not _LEASE_WIKI_VAULT.exists():
        return []
    return [
        p for p in _LEASE_WIKI_VAULT.rglob("*.md")
        if p.relative_to(_LEASE_WIKI_VAULT).parts
        and p.relative_to(_LEASE_WIKI_VAULT).parts[0][:2].isdigit()
    ]


def _refresh_wiki_cache(cache: dict, api_key: str) -> tuple[dict, bool]:
    """Re-embed changed or new wiki files; prune deleted ones. Returns (cache, changed)."""
    files = _wiki_numbered_files()
    changed = False
    current_rels: set[str] = set()

    for md_file in files:
        rel = str(md_file.relative_to(_LEASE_WIKI_VAULT))
        current_rels.add(rel)
        mtime = md_file.stat().st_mtime
        if cache.get(rel, {}).get("mtime") == mtime:
            continue
        try:
            raw = md_file.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()
            embedding = _embed_text(text, api_key)
            cache[rel] = {"mtime": mtime, "embedding": embedding, "snippet": text[:300]}
            changed = True
        except Exception:
            continue

    for rel in list(cache.keys()):
        if rel not in current_rels:
            del cache[rel]
            changed = True

    return cache, changed


def _wiki_keyword_fallback(query: str, limit: int) -> dict[str, Any]:
    """Keyword fallback when embedding is unavailable."""
    results: list[dict] = []
    q = query.lower()
    for md_file in sorted(_wiki_numbered_files(), key=lambda p: str(p.relative_to(_LEASE_WIKI_VAULT))):
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            if q in text.lower():
                idx = text.lower().find(q)
                snippet = text[max(0, idx - 120): idx + 300].strip()
                results.append({"file": str(md_file.relative_to(_LEASE_WIKI_VAULT)), "snippet": snippet})
                if len(results) >= limit:
                    break
        except Exception:
            continue
    return {"results": results, "count": len(results), "mode": "keyword"}


def search_lease_wiki(query: str, limit: int = 3) -> dict[str, Any]:
    """Semantic search of the lease-wiki-vault using Gemini text-embedding-004.

    Covers: scoring thresholds, asset-category residual risk, interest rate
    benchmarks, LightGBM model specs, field definitions, design decisions.
    Falls back to keyword search if the embedding API is unavailable.
    """
    if not _LEASE_WIKI_VAULT.exists():
        return {"error": "lease-wiki-vault が見つかりません", "results": []}

    api_key = _gemini_api_key_for_tools()
    if not api_key:
        return _wiki_keyword_fallback(query, limit)

    try:
        cache = _load_wiki_cache()
        cache, changed = _refresh_wiki_cache(cache, api_key)
        if changed:
            _save_wiki_cache(cache)

        q_emb = _embed_text(query, api_key)
        scored = [
            (_cosine_sim(q_emb, entry["embedding"]), rel, entry.get("snippet", ""))
            for rel, entry in cache.items()
            if "embedding" in entry
        ]
        scored.sort(reverse=True)
        results = [
            {"file": rel, "snippet": snippet, "score": round(score, 3)}
            for score, rel, snippet in scored[:limit]
        ]
        return {"results": results, "count": len(results), "mode": "semantic"}
    except Exception:
        return _wiki_keyword_fallback(query, limit)


def execute_tool(name: str, args: dict, vault: Path | None = None) -> Any:
    """Dispatch a tool call by name and return its result."""
    if name == "search_cases":
        return search_cases(args.get("query", ""), int(args.get("limit", 5)))
    if name == "get_score_detail":
        return get_score_detail(args.get("company_name", ""))
    if name == "compare_similar_cases":
        return compare_similar_cases(
            args.get("industry", ""),
            float(args.get("score_min", 0)),
            float(args.get("score_max", 100)),
        )
    if name == "get_weekly_trend":
        return get_weekly_trend(int(args.get("weeks", 4)))
    if name == "search_obsidian":
        if vault is None:
            return {"error": "vault path not available"}
        return search_obsidian(args.get("query", ""), vault, int(args.get("limit", 3)))
    if name == "search_lease_wiki":
        return search_lease_wiki(args.get("query", ""), int(args.get("limit", 3)))
    if name == "inspect_scoring_policy":
        return inspect_scoring_policy(args.get("topic", ""))
    if name == "consult_senior_reasoner":
        if vault is None:
            return {"error": "vault path not available", "consulted": False}
        from lease_intelligence_consultation import consult_senior_reasoner

        return consult_senior_reasoner(
            question=args.get("question", ""),
            shion_hypothesis=args.get("shion_hypothesis", ""),
            confidence=args.get("confidence", 0.5),
            evidence_summary=args.get("evidence_summary", ""),
            vault=vault,
        )
    return {"error": f"unknown tool: {name}"}


# Gemini function_declarations schema for all tools
TOOL_DECLARATIONS: list[dict] = [
    {
        "name": "search_cases",
        "description": "審査履歴DBを検索する。会社名・業種・キーワードで直近の審査案件を取得できる。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索キーワード（会社名・業種等）"},
                "limit": {"type": "integer", "description": "取得件数（デフォルト5、最大20）"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_score_detail",
        "description": "指定した会社の最新スコアを詳細取得する。要因分解・リスクフラグ・入力サマリーを返す。",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "調査対象の会社名"},
            },
            "required": ["company_name"],
        },
    },
    {
        "name": "compare_similar_cases",
        "description": "同業種・同スコア帯の過去案件を比較し、成約率・失注率・平均スコアを返す。",
        "parameters": {
            "type": "object",
            "properties": {
                "industry": {"type": "string", "description": "業種キーワード（例: 建設・飲食・運輸）"},
                "score_min": {"type": "number", "description": "スコア下限（デフォルト0）"},
                "score_max": {"type": "number", "description": "スコア上限（デフォルト100）"},
            },
            "required": ["industry"],
        },
    },
    {
        "name": "get_weekly_trend",
        "description": "週次集計データ（週別スコア・件数推移）を取得する。",
        "parameters": {
            "type": "object",
            "properties": {
                "weeks": {"type": "integer", "description": "取得する週数（デフォルト4）"},
            },
        },
    },
    {
        "name": "search_obsidian",
        "description": "Obsidian Vaultのナレッジノートをキーワード検索する。業界情報・業務記録・Daily Brief・方針メモを参照できる。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索キーワード"},
                "limit": {"type": "integer", "description": "返す件数（デフォルト3）"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_lease_wiki",
        "description": (
            "リース審査専門Wikiをキーワード検索する。"
            "スコア閾値・物件カテゴリ別リスク・金利相場・LightGBMモデル仕様・設計決定ログなど"
            "「なぜそうスコアされるのか」「このカテゴリのリスクは？」といった審査ロジック系の質問に使う。"
            "過去案件の検索は search_cases を使うこと。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索キーワード（例: 残価リスク・承認ライン・工作機械・金利）"},
                "limit": {"type": "integer", "description": "返す件数（デフォルト3）"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "inspect_scoring_policy",
        "description": (
            "現行の実行コードに基づき、最終スコア・借手スコア・物件スコア・承認ラインの"
            "関係を確認する。審査ロジック、統合、重み付け、なぜ承認されたかを説明するときは、"
            "WikiやObsidian検索だけで結論を出さず必ずこのツールで実装仕様を照合する。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "確認したい論点"},
            },
        },
    },
    {
        "name": "consult_senior_reasoner",
        "description": (
            "紫苑が自分の初期仮説を作った後、難問・矛盾・低確信度の論点をCodexへ"
            "読取専用で相談する。利用前に必ず紫苑自身の仮説、確信度、確認済み根拠を渡す。"
            "得た助言は丸写しせず、何を維持・修正したかを紫苑自身の結論へ統合する。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "個人名・社名・生の財務数値を除いた抽象的な論点",
                },
                "shion_hypothesis": {
                    "type": "string",
                    "description": "相談前に紫苑自身が考えた初期仮説",
                },
                "confidence": {
                    "type": "number",
                    "description": "初期仮説の確信度（0から1）",
                },
                "evidence_summary": {
                    "type": "string",
                    "description": "紫苑がツール等で確認した根拠の要約",
                },
            },
            "required": [
                "question",
                "shion_hypothesis",
                "confidence",
                "evidence_summary",
            ],
        },
    },
]
