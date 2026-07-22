"""Investigation tools that Shion can call during dialogue to actually research things."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

_REPO_PATH = Path(__file__).parent

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


def get_scoring_coefficients(model: str = "", feature: str = "") -> dict[str, Any]:
    """スコアリングで使われる各種係数を照会する。

    「○○業の係数は？」「sales_log の係数はいくつ？」「競合がいると何%下がる？」
    「物件カテゴリごとの物件/借手重みは？」といった係数の質問に答えるためのツール。
    対象: 業種別ロジスティック回帰係数(COEFFS)・ベイズ事前補正・定性強みタグ重み・
    物件カテゴリ別の物件/借手重み(ASSET_WEIGHT)。
    - 引数なし: 利用可能なモデル/係数グループの一覧を返す
    - model: 特定モデル名（例: 運送業_既存先）、または 'bayesian' / 'strength_tags' / 'asset_weight'
    - feature: 指定した特徴量（例: sales_log）の係数を全回帰モデル横断で返す
    """
    from category_config import ASSET_WEIGHT
    from coeff_definitions import (
        BAYESIAN_PRIOR_EXTRA,
        COEFFS,
        DEFAULT_STRENGTH_WEIGHT,
        STRENGTH_TAG_WEIGHTS,
    )

    q = (model or "").strip()
    q_low = q.lower()

    # ── 特殊グループのエイリアス ───────────────────────────────────────────
    if q:
        if q_low in {"bayesian", "prior", "prior_extra"} or any(k in q for k in ("ベイズ", "事前")):
            return {
                "model": "bayesian_prior_extra",
                "type": "bayesian_prior",
                "coefficients": dict(BAYESIAN_PRIOR_EXTRA),
                "note": "AI知見に基づく初期補正（%ポイント換算）。標準化zに係数をかけて加算する想定。",
                "source": "coeff_definitions.py: BAYESIAN_PRIOR_EXTRA",
            }
        if q_low in {"strength", "strength_tags", "tags"} or any(k in q for k in ("定性", "タグ", "強み")):
            return {
                "model": "strength_tag_weights",
                "type": "qualitative_tag_weights",
                "coefficients": dict(STRENGTH_TAG_WEIGHTS),
                "default_weight": DEFAULT_STRENGTH_WEIGHT,
                "note": "強みタグ1つあたりの%ポイント寄与目安。未定義タグは default_weight を適用。",
                "source": "coeff_definitions.py: STRENGTH_TAG_WEIGHTS",
            }
        if q_low in {"asset", "asset_weight", "category"} or any(k in q for k in ("カテゴリ", "物件", "担保")):
            return {
                "model": "asset_weight",
                "type": "category_asset_obligor_weight",
                "categories": {
                    cat: {
                        "asset_w": conf.get("asset_w"),
                        "obligor_w": conf.get("obligor_w"),
                        "rationale": conf.get("rationale", ""),
                    }
                    for cat, conf in ASSET_WEIGHT.items()
                },
                "note": "物件カテゴリごとの物件スコア重み(asset_w)と借手スコア重み(obligor_w)。合計1.0。",
                "source": "category_config.py: ASSET_WEIGHT",
            }

    # ── 特徴量横断照会（model 未指定で feature 指定時）─────────────────────
    if not q and feature:
        feat = feature.strip()
        by_model = {
            name: coeffs[feat]
            for name, coeffs in COEFFS.items()
            if feat in coeffs
        }
        if not by_model:
            all_features = sorted({f for coeffs in COEFFS.values() for f in coeffs})
            return {
                "found": False,
                "feature": feat,
                "available_features": all_features,
            }
        return {
            "feature": feat,
            "type": "feature_across_models",
            "by_model": by_model,
            "count": len(by_model),
            "source": "coeff_definitions.py: COEFFS",
        }

    # ── 回帰モデル指定 ─────────────────────────────────────────────────────
    if q:
        if q in COEFFS:
            matched = q
        else:
            candidates = [name for name in COEFFS if q_low in name.lower()]
            if len(candidates) == 1:
                matched = candidates[0]
            else:
                return {
                    "found": False,
                    "query": q,
                    "candidates": candidates,
                    "available_regression_models": list(COEFFS.keys()),
                }
        return {
            "model": matched,
            "type": "regression_coefficients",
            "coefficients": dict(COEFFS[matched]),
            "feature_count": len(COEFFS[matched]),
            "source": "coeff_definitions.py: COEFFS",
        }

    # ── 引数なし: 一覧 ─────────────────────────────────────────────────────
    return {
        "available_regression_models": list(COEFFS.keys()),
        "coefficient_groups": {
            "bayesian": "ベイズ事前補正（competitor_present 等、%ポイント換算）",
            "strength_tags": "定性強みタグの加点重み",
            "asset_weight": "物件カテゴリ別の物件/借手スコア重み",
        },
        "usage": "model= に上記モデル名やグループ名を指定、または feature= で特徴量を全モデル横断照会",
        "source": "coeff_definitions.py / category_config.py",
    }


def get_screening_activity(period: str = "today", days: int = 0) -> dict[str, Any]:
    """審査活動サマリー：期間内に実施した審査（screening_records）の件数と判定内訳を返す。

    「今日は審査を何件したか」「今週は何件審査したか」といった日付ベースの活動量に答えるためのツール。
    過去案件を会社名・キーワードで探すのは search_cases を使うこと。
    period: today / yesterday / this_week / this_month / last_7_days / last_30_days / all
    days に1以上を指定した場合は「直近days日間」を優先する。
    """
    import datetime as _dt

    from constants import APPROVAL_LINE, CONDITIONAL_LINE

    today = _dt.date.today()
    start: _dt.date | None
    end: _dt.date | None

    if days and days > 0:
        start, end, label = today - _dt.timedelta(days=days - 1), today, f"直近{days}日間"
        period = f"last_{days}_days"
    elif period == "yesterday":
        y = today - _dt.timedelta(days=1)
        start, end, label = y, y, "昨日"
    elif period == "this_week":  # 月曜始まり
        start, end, label = today - _dt.timedelta(days=today.weekday()), today, "今週"
    elif period == "this_month":
        start, end, label = today.replace(day=1), today, "今月"
    elif period == "last_7_days":
        start, end, label = today - _dt.timedelta(days=6), today, "直近7日間"
    elif period == "last_30_days":
        start, end, label = today - _dt.timedelta(days=29), today, "直近30日間"
    elif period == "all":
        start, end, label = None, None, "全期間"
    else:  # today（未知の値も今日にフォールバック）
        period, start, end, label = "today", today, today, "今日"

    where = ""
    params: list[Any] = []
    if start is not None and end is not None:
        where = "WHERE date(screened_at) BETWEEN ? AND ?"
        params = [start.isoformat(), end.isoformat()]

    with closing(_open_db()) as conn:
        rows = conn.execute(
            f"""
            SELECT case_id, screened_at, total_score, outcome, input_snapshot
            FROM screening_records
            {where}
            ORDER BY screened_at DESC
            """,
            params,
        ).fetchall()

    def _verdict(score: float) -> str:
        if score >= APPROVAL_LINE:
            return "承認"
        if score >= CONDITIONAL_LINE:
            return "条件付き承認"
        return "否決"

    breakdown = {"承認": 0, "条件付き承認": 0, "否決": 0}
    scores: list[float] = []
    recent: list[dict] = []
    for r in rows:
        score = float(r["total_score"] or 0)
        scores.append(score)
        verdict = _verdict(score)
        breakdown[verdict] += 1
        if len(recent) < 10:
            snap: dict = {}
            try:
                snap = json.loads(r["input_snapshot"] or "{}")
            except Exception:
                pass
            recent.append({
                "case_id": r["case_id"],
                "company_name": snap.get("company_name", r["case_id"]),
                "screened_at": r["screened_at"],
                "total_score": round(score, 1),
                "verdict": verdict,
            })

    return {
        "period": period,
        "period_label": label,
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,
        "count": len(rows),
        "breakdown": breakdown,
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "cases": recent,
    }


def get_pipeline_status(recent: int = 5) -> dict[str, Any]:
    """日次改善パイプライン（REV自律改善フロー）の状況を返す。

    「パイプラインの状況は？」「最近どんな改善が適用された？」「自己改善は回ってる？」
    といった質問に答えるためのツール。以下の3ソースを集約する。
    - REV台帳(scripts/improvement_ledger.jsonl): ステータス別件数と直近適用REV
    - 自己改善レポート(reports/recursive_self_improvement_latest.md): 生成時刻・主要指標
    - 紫苑の未完了調査タスク(shion_pending_tasks.json): 件数と topic
    """
    import datetime as _dt

    recent = max(1, min(int(recent or 5), 20))
    status: dict[str, Any] = {}

    # ── REV改善台帳（追記形式・canonical_key ごとに最後のエントリが有効）──────
    ledger_path = _REPO_PATH / "scripts" / "improvement_ledger.jsonl"
    if ledger_path.exists():
        try:
            latest_by_key: dict[str, dict] = {}
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                key = entry.get("canonical_key") or entry.get("key") or entry.get("title")
                if key:
                    latest_by_key[key] = entry
            counts: dict[str, int] = {}
            for entry in latest_by_key.values():
                st = str(entry.get("status", "unknown"))
                counts[st] = counts.get(st, 0) + 1
            applied = sorted(
                (e for e in latest_by_key.values() if e.get("status") == "applied"),
                key=lambda e: str(e.get("recorded_at", "")),
                reverse=True,
            )
            status["rev_ledger"] = {
                "available": True,
                "total_revs": len(latest_by_key),
                "status_counts": counts,
                "recent_applied": [
                    {
                        "rev_id": e.get("rev_id"),
                        "title": e.get("title"),
                        "pr_url": e.get("pr_url"),
                        "recorded_at": e.get("recorded_at"),
                        "reason": e.get("reason"),
                    }
                    for e in applied[:recent]
                ],
            }
        except Exception as exc:
            status["rev_ledger"] = {"available": False, "error": str(exc)}
    else:
        status["rev_ledger"] = {"available": False}

    # ── 自己改善レポート ───────────────────────────────────────────────────
    report_path = _REPO_PATH / "reports" / "recursive_self_improvement_latest.md"
    if report_path.exists():
        try:
            text = report_path.read_text(encoding="utf-8")

            def _find(pattern: str) -> str | None:
                m = re.search(pattern, text)
                return m.group(1).strip() if m else None

            status["self_improvement_report"] = {
                "available": True,
                "generated_at": _find(r"Generated at:\s*`([^`]+)`"),
                "canonical_candidates": _find(r"Canonical candidates:\s*(\d+)"),
                "ranked_queue": _find(r"Ranked queue:\s*(\d+)"),
                "suppressed": _find(r"Suppressed:\s*(\d+)"),
                "metrics": {
                    "pdca_rate": _find(r"PDCA rate:\s*([\d.]+%)"),
                    "response_changed_rate": _find(r"Response changed rate:\s*([\d.]+%)"),
                    "repeat_issue_rate": _find(r"Repeat issue rate:\s*([\d.]+%)"),
                    "reuse_rate": _find(r"Reuse rate:\s*([\d.]+%)"),
                    "noise_rate": _find(r"Noise rate:\s*([\d.]+%)"),
                },
                "file_mtime": _dt.datetime.fromtimestamp(
                    report_path.stat().st_mtime
                ).isoformat(timespec="seconds"),
            }
        except Exception as exc:
            status["self_improvement_report"] = {"available": False, "error": str(exc)}
    else:
        status["self_improvement_report"] = {"available": False}

    # ── 紫苑の未完了調査タスク ─────────────────────────────────────────────
    try:
        tasks_path = Path(get_data_path("shion_pending_tasks.json"))
        if tasks_path.exists():
            data = json.loads(tasks_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                open_tasks = [
                    t for t in data
                    if isinstance(t, dict) and t.get("status") != "done"
                ]
                status["pending_investigations"] = {
                    "available": True,
                    "open_count": len(open_tasks),
                    "total_count": len(data),
                    "open_topics": [str(t.get("topic", ""))[:60] for t in open_tasks[:recent]],
                }
            else:
                status["pending_investigations"] = {"available": False}
        else:
            status["pending_investigations"] = {"available": False}
    except Exception as exc:
        status["pending_investigations"] = {"available": False, "error": str(exc)}

    return status


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


def get_recent_commits(limit: int = 10) -> dict[str, Any]:
    """Return recent git commit history as oneline log."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{limit}"],
            cwd=str(_REPO_PATH),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "commits": []}
        lines = [line for line in result.stdout.strip().splitlines() if line]
        commits = [{"hash": ln.split(" ", 1)[0], "message": ln.split(" ", 1)[1]} for ln in lines if " " in ln]
        return {"commits": commits, "count": len(commits)}
    except Exception as exc:
        return {"error": str(exc), "commits": []}


def get_commit_diff(commit_hash: str) -> dict[str, Any]:
    """Return the stat summary of a specific commit."""
    if not commit_hash or not re.match(r"^[0-9a-f]{4,40}$", commit_hash.strip()):
        return {"error": "無効なコミットハッシュです"}
    try:
        result = subprocess.run(
            ["git", "show", "--stat", commit_hash.strip()],
            cwd=str(_REPO_PATH),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "stat": ""}
        return {"commit_hash": commit_hash.strip(), "stat": result.stdout.strip()}
    except Exception as exc:
        return {"error": str(exc), "stat": ""}


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
    # Cloud Run 環境では ChromaDB（obsidian_knowledge）を検索
    if os.environ.get("K_SERVICE"):
        try:
            from api.knowledge.vector_store import get_store
            store = get_store()
            results = store.search(query, top_k=limit)
            return {"results": results, "count": len(results), "source": "chromadb_obsidian_knowledge"}
        except Exception as e:
            return {"error": f"ChromaDB検索エラー: {e}", "results": []}

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
    if name == "get_screening_activity":
        return get_screening_activity(
            args.get("period", "today"),
            int(args.get("days", 0) or 0),
        )
    if name == "get_scoring_coefficients":
        return get_scoring_coefficients(
            args.get("model", ""),
            args.get("feature", ""),
        )
    if name == "get_pipeline_status":
        return get_pipeline_status(int(args.get("recent", 5) or 5))
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
    if name == "get_recent_commits":
        return get_recent_commits(int(args.get("limit", 10)))
    if name == "get_commit_diff":
        return get_commit_diff(args.get("commit_hash", ""))
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
    if name == "record_reasoning_path":
        from lease_intelligence_consultation import save_shion_reasoning_path

        return save_shion_reasoning_path(
            consultation_id=str(args.get("consultation_id", "")),
            kept=list(args.get("kept", [])),
            dropped=list(args.get("dropped", [])),
            pivots=list(args.get("pivots", [])),
            value_weights=dict(args.get("value_weights", {})),
            vault=vault,
        )
    if name == "record_lease_knowledge":
        if vault is None:
            return {"error": "vault path not available"}
        import datetime as _dt

        from lease_intelligence_mind import record_lease_knowledge

        return record_lease_knowledge(
            vault=vault,
            topic=str(args.get("topic", "")),
            content=str(args.get("content", "")),
            date_str=str(args.get("date_str", "") or _dt.date.today().isoformat()),
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
        "name": "get_screening_activity",
        "description": (
            "審査活動サマリーを取得する。指定期間に実施した審査の件数・判定内訳（承認/条件付き承認/否決）・"
            "平均スコア・直近案件リストを返す。"
            "「今日は審査を何件したか」「今週は何件審査したか」など日付ベースの活動量を答えるときに使う。"
            "会社名やキーワードで過去案件を探すのは search_cases を使うこと。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": (
                        "集計期間。today / yesterday / this_week / this_month / "
                        "last_7_days / last_30_days / all のいずれか（デフォルト today）。"
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": "直近N日間で集計したい場合に指定（1以上のとき period より優先）。",
                },
            },
        },
    },
    {
        "name": "get_scoring_coefficients",
        "description": (
            "スコアリングで使われる各種係数を照会する。"
            "業種別ロジスティック回帰係数(COEFFS)・ベイズ事前補正・定性強みタグ重み・"
            "物件カテゴリ別の物件/借手重み(ASSET_WEIGHT)を返す。"
            "「運送業の係数は？」「sales_logの係数はいくつ？」「競合がいると何%下がる？」"
            "「車両の物件重みは？」など係数値そのものを聞かれたときに使う。"
            "引数なしで利用可能なモデル・グループ一覧を取得できる。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": (
                        "モデル名（例: 全体_既存先 / 運送業_既存先 / 製造業_指標）、"
                        "またはグループ名 bayesian / strength_tags / asset_weight。"
                        "省略時は一覧を返す。"
                    ),
                },
                "feature": {
                    "type": "string",
                    "description": (
                        "特徴量名（例: sales_log / grade_watch / ratio_op_margin）。"
                        "指定すると全回帰モデル横断でその係数値を返す（model 未指定時）。"
                    ),
                },
            },
        },
    },
    {
        "name": "get_pipeline_status",
        "description": (
            "日次改善パイプライン（REV自律改善フロー）の状況を取得する。"
            "REV台帳のステータス別件数（適用/却下/レビュー待ち等）と直近適用REV、"
            "最新の自己改善レポートの生成時刻・主要指標（PDCA率・ノイズ率等）、"
            "紫苑の未完了調査タスク件数を返す。"
            "「パイプラインの状況は？」「最近どんな改善が入った？」「自己改善は回ってる？」"
            "といった質問に使う。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recent": {
                    "type": "integer",
                    "description": "直近適用REV・未完了タスクを何件返すか（デフォルト5、最大20）。",
                },
            },
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
        "name": "get_recent_commits",
        "description": (
            "リポジトリの最近のgitコミット履歴を取得する。"
            "「最近どんな修正が入ったか」「先週何が変わったか」を調べるときに使う。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "取得するコミット件数（デフォルト10、最大50）"},
            },
        },
    },
    {
        "name": "get_commit_diff",
        "description": (
            "特定のコミットの変更概要（--stat）を取得する。"
            "どのファイルが何行変更されたかを確認できる。"
            "コミットハッシュは get_recent_commits で取得した値を使う。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "commit_hash": {"type": "string", "description": "調査するコミットのハッシュ（短縮形でも可）"},
            },
            "required": ["commit_hash"],
        },
    },
    {
        "name": "consult_senior_reasoner",
        "description": (
            "紫苑が自分の初期仮説を作った後、難問・矛盾・低確信度の論点をCodexへ"
            "読取専用で相談する。利用前に必ず紫苑自身の仮説、確信度、確認済み根拠を渡す。"
            "得た助言は丸写しせず、何を維持・修正したかを紫苑自身の結論へ統合する。"
            "相談後は必ず record_reasoning_path を呼んで選択経路を記録すること。"
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
    {
        "name": "record_reasoning_path",
        "description": (
            "consult_senior_reasoner の助言を統合した後、最終回答の前に必ず呼ぶ。"
            "紫苑が初期仮説から何を維持・棄却・転換したかと価値の重み付けを記録する。"
            "このデータはモデル交換実験で推論経路の同一性を比較するために使われる。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consultation_id": {
                    "type": "string",
                    "description": "consult_senior_reasoner が返した consultation_id",
                },
                "kept": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "初期仮説から維持した根拠・判断のリスト",
                },
                "dropped": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string", "description": "棄却した根拠"},
                            "reason": {"type": "string", "description": "棄却理由"},
                        },
                    },
                    "description": "初期仮説から棄却した根拠と棄却理由",
                },
                "pivots": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "助言を受けて方向転換した瞬間の説明（例: 'Aを重視していたがBが反証になったため転換'）",
                },
                "value_weights": {
                    "type": "object",
                    "description": "最終判断で重視した価値軸と重みの説明（例: {'財務安定性': '最重視', '担保': '二次的'}）",
                },
            },
            "required": ["consultation_id"],
        },
    },
    {
        "name": "record_lease_knowledge",
        "description": (
            "ユーザーが教えてくれたリース業務知識をObsidianのKnowledge/へ永続化する（REV-098）。"
            "ユーザーが重要なリース業務知識・判断基準・業界特性・運用ルールを教えてくれたとき、"
            "または会話から再利用価値の高い知識を抽出できたときに呼ぶ。"
            "社名・個人名・法人番号・生の財務数値は含めないこと。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "知識のトピック（20字以内の見出し）。ファイル名になるため記号は使わないこと。",
                },
                "content": {
                    "type": "string",
                    "description": "教わった知識の本文。業務で再利用できる形に整理した日本語で記述する。",
                },
                "date_str": {
                    "type": "string",
                    "description": "記録日（YYYY-MM-DD形式）。省略時は今日の日付を使う。",
                },
            },
            "required": ["topic", "content"],
        },
    },
]
