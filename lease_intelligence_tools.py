"""Investigation tools that Shion can call during dialogue to actually research things."""
from __future__ import annotations

import json
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


def _search_vault(query: str, vault: Path, limit: int) -> dict[str, Any]:
    """Shared keyword search implementation for any Obsidian vault directory."""
    results: list[dict] = []
    q = query.lower()
    try:
        for md_file in sorted(vault.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                if q in text.lower():
                    idx = text.lower().find(q)
                    start = max(0, idx - 120)
                    end = min(len(text), idx + 300)
                    snippet = text[start:end].strip()
                    results.append({
                        "file": str(md_file.relative_to(vault)),
                        "snippet": snippet,
                    })
                    if len(results) >= limit:
                        break
            except Exception:
                continue
    except Exception as e:
        return {"error": str(e), "results": []}
    return {"results": results, "count": len(results)}


def search_obsidian(query: str, vault: Path, limit: int = 3) -> dict[str, Any]:
    """Search main Obsidian vault markdown files for notes matching the query."""
    return _search_vault(query, vault, limit)


_WIKI_SKIP_PREFIXES = ("@AI_", "@Web_", "99_Synced_From_Origin")


def search_lease_wiki(query: str, limit: int = 3) -> dict[str, Any]:
    """Search the lease-wiki-vault for specialized lease domain knowledge.

    The wiki contains: scoring thresholds, asset risk by category, interest rate
    benchmarks, LightGBM model specs, field definitions, design decisions.
    Use this for questions about HOW the scoring system works or WHY a result
    appears — not for searching past cases (use search_cases for that).
    """
    if not _LEASE_WIKI_VAULT.exists():
        return {"error": "lease-wiki-vault が見つかりません", "results": []}

    results: list[dict] = []
    q = query.lower()
    # Prioritize numbered dirs (00_ … 10_), then other files; exclude auto-generated noise
    try:
        all_files = list(_LEASE_WIKI_VAULT.rglob("*.md"))
        def _sort_key(p: Path) -> tuple[int, str]:
            rel = p.relative_to(_LEASE_WIKI_VAULT)
            top = rel.parts[0] if rel.parts else ""
            # numbered dirs first, then everything else
            priority = 0 if top[:2].isdigit() else 1
            return (priority, str(rel))

        for md_file in sorted(all_files, key=_sort_key):
            rel = md_file.relative_to(_LEASE_WIKI_VAULT)
            top = rel.parts[0] if rel.parts else ""
            if any(top.startswith(skip) for skip in _WIKI_SKIP_PREFIXES):
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                if q in text.lower():
                    idx = text.lower().find(q)
                    start = max(0, idx - 120)
                    end = min(len(text), idx + 300)
                    snippet = text[start:end].strip()
                    results.append({
                        "file": str(rel),
                        "snippet": snippet,
                    })
                    if len(results) >= limit:
                        break
            except Exception:
                continue
    except Exception as e:
        return {"error": str(e), "results": []}
    return {"results": results, "count": len(results)}


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
]
