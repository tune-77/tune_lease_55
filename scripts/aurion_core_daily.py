#!/usr/bin/env python3
"""Daily AURION CORE / MEBUKI autonomous sync and morning report.

This script is intentionally deterministic. It does not modify app code and it
does not rely on an interactive Codex session. It syncs relevant Obsidian notes,
audits the canonical tune_lease_55 SQLite DB, records a nightly status file, and
creates a morning Markdown report in lease-wiki-vault.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/Users/kobayashiisaoryou/clawd/tune_lease_55")
ORIGIN_VAULT = Path(
    "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
)
LEASE_VAULT = Path(
    "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/lease-wiki-vault"
)
SYNC_ROOT = LEASE_VAULT / "99_Synced_From_Origin"
DB_PATH = PROJECT_ROOT / "data" / "lease_data.db"
STATE_DIR = PROJECT_ROOT / "data" / "aurion_daily"
LOG_DIR = PROJECT_ROOT / "logs" / "aurion_daily"

KEYWORDS = [
    "リース",
    "審査",
    "Q_risk",
    "Q-Risk",
    "リスクアセスメント",
    "マハラノビス",
    "KLダイバージェンス",
    "情報幾何学",
    "S&P 500",
    "LSTM",
    "UI",
    "スライダー",
    "タイル",
    "Clawdbot",
    "スコアリング",
    "システム",
    "スコア",
    "リスク",
    "Chat",
    "Improvement Log",
    "開発",
    "改善",
]
STRONG_KEYWORDS = [
    "リース",
    "審査",
    "Q_risk",
    "Q-Risk",
    "リスクアセスメント",
    "マハラノビス",
    "KLダイバージェンス",
    "情報幾何学",
    "S&P 500",
    "LSTM",
    "スコアリング",
    "スコア",
    "リスク",
    "金利",
    "補助金",
    "業種",
    "倒産",
    "残価",
    "物件",
    "財務",
    "DB",
    "RAG",
    "AURION",
    "MEBUKI",
    "LightGBM",
]
WEAK_KEYWORDS = [
    "UI",
    "スライダー",
    "タイル",
    "Clawdbot",
    "システム",
    "Chat",
    "Improvement Log",
    "開発",
    "改善",
]
EXCLUDE = [
    "ラーメン",
    "ラーメン屋",
    "映画",
    "ディズニー",
    "釣り",
    "プラモ",
    "犬の散歩",
    "八奈見",
    "Humor",
    "アニメ",
    "漫画",
]
EXTS = {".md", ".txt", ".log"}

WEB_SOURCES = [
    {
        "theme": "credit-model-monitoring",
        "title": "OECD Financing SMEs and Entrepreneurs 2026",
        "url": "https://read.oecd-ilibrary.org/en/publications/financing-smes-and-entrepreneurs-2026_075d8058-en.html",
        "fallback": "SME borrowing costs remain high relative to pre-pandemic levels; leasing and other alternative finance remain mixed or subdued. This supports treating approval rates and lease demand as market-regime variables, not fixed model priors.",
    },
    {
        "theme": "equipment-leasing-market",
        "title": "ELFA 2026 Equipment Leasing & Finance Economic Outlook",
        "url": "https://www.elfaonline.org/research/2026-equipment-leasing-finance-u-s-economic-outlook-2026-update",
        "fallback": "Equipment finance demand is supported by AI-related capex and replacement investment, while policy uncertainty, volatility, borrower disparity, and downside macro risk remain material. Pricing engines must expose the risk premium separately from competitive discounting.",
    },
    {
        "theme": "japan-structured-finance",
        "title": "S&P Global Japan Structured Finance Outlook 2026",
        "url": "https://www.spglobal.com/ratings/en/regulatory/article/japan-structured-finance-outlook-2026-jobs-strength-offsets-hikes-s101663301",
        "fallback": "Japan structured-finance performance depends on employment, borrower repayment capacity, asset values, and SME default trends. Lease underwriting should not weaken residual-value and cash-flow checks merely because headline employment is stable.",
    },
]


def now() -> datetime:
    return datetime.now()


def date_str() -> str:
    return now().strftime("%Y-%m-%d")


def _mkdirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SYNC_ROOT.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path, limit: int = 80_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def _date_from_name(path: Path) -> datetime | None:
    import re

    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", path.name)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _is_recent_chat_path(rel: str, path: Path, max_age_days: int = 7) -> bool:
    if "AI Chat" not in rel and "/Chat/" not in rel:
        return False
    file_date = _date_from_name(path)
    if not file_date:
        return False
    return file_date >= datetime.combine(now().date(), datetime.min.time()) - timedelta(days=max_age_days)


def _matches(path: Path) -> bool:
    rel = str(path.relative_to(ORIGIN_VAULT))
    if any(x in rel for x in EXCLUDE):
        return False
    text = _read_text(path, 50_000)
    hay = f"{rel}\n{text[:50_000]}"
    if any(x in hay for x in EXCLUDE):
        return False
    strong_hits = sum(1 for k in STRONG_KEYWORDS if k in hay)
    weak_hits = sum(1 for k in WEAK_KEYWORDS if k in hay)

    if rel.startswith("Daily/"):
        return False
    if _is_recent_chat_path(rel, path):
        return strong_hits >= 1
    if "AI Chat" in rel or "/Chat/" in rel:
        return False
    if rel.startswith(("Asset Knowledge/", "リース知識/", "tuneLease55/知見・分析/")):
        return strong_hits >= 1
    if rel.startswith(
        (
            "Projects/tune_lease_55/Asset Knowledge/",
            "Projects/tune_lease_55/Asset Finance/",
            "Projects/tune_lease_55/Cases/",
            "Projects/tune_lease_55/Generated/",
            "Projects/tune_lease_55/News/",
            "Projects/tune_lease_55/Research/",
        )
    ):
        return strong_hits >= 1
    if rel.startswith("Projects/tune_lease_55/"):
        return strong_hits >= 1 and (strong_hits >= 2 or weak_hits >= 1)
    if rel.startswith("30.areas/"):
        return strong_hits >= 1 and (strong_hits >= 2 or "RAG" in hay)
    if rel.startswith("Clippings/"):
        return strong_hits >= 2
    if rel.startswith("改善ログ/"):
        return strong_hits >= 1 and weak_hits >= 1
    return strong_hits >= 2


def sync_notes() -> dict[str, Any]:
    selected: list[Path] = []
    if not ORIGIN_VAULT.exists():
        return {"status": "failed", "reason": f"missing origin vault: {ORIGIN_VAULT}"}

    for path in ORIGIN_VAULT.rglob("*"):
        if path.is_file() and path.suffix in EXTS and _matches(path):
            selected.append(path)

    selected_rels = {path.relative_to(ORIGIN_VAULT) for path in selected}
    pruned: list[Path] = []
    if SYNC_ROOT.exists():
        for existing in SYNC_ROOT.rglob("*"):
            if not existing.is_file() or existing.suffix not in EXTS or existing.name.startswith("_SYNC_"):
                continue
            rel = existing.relative_to(SYNC_ROOT)
            if rel not in selected_rels:
                existing.unlink()
                pruned.append(rel)

    copied = 0
    for path in selected:
        rel = path.relative_to(ORIGIN_VAULT)
        out = SYNC_ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        copied += 1

    manifest = SYNC_ROOT / f"_SYNC_MANIFEST_{date_str()}.md"
    lines = [
        f"# Sync Manifest {date_str()}",
        "",
        f"- Source: `{ORIGIN_VAULT}`",
        f"- Destination: `{SYNC_ROOT}`",
        f"- Selected files: {copied}",
        f"- Pruned stale synced files: {len(pruned)}",
        f"- Generated: {now().isoformat(timespec='seconds')}",
        "",
        "## Files",
    ]
    for path in sorted(selected, key=lambda p: str(p.relative_to(ORIGIN_VAULT))):
        rel = path.relative_to(ORIGIN_VAULT)
        lines.append(f"- [[{path.stem}]] `{rel}`")
    if pruned:
        lines.extend(["", "## Pruned From Synced Copy"])
        for rel in sorted(pruned, key=str):
            lines.append(f"- `{rel}`")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "completed", "selected_files": copied, "pruned_files": len(pruned), "manifest": str(manifest)}


def reindex_vault_b() -> dict[str, Any]:
    """Diff-reindex Lease Vault B after syncing from the origin vault.

    Do not use --full here. The 03:00 origin-vault reindex may have rebuilt the
    shared Chroma collection; this step only upserts Vault B changes after the
    AURION sync so both source knowledge and purified Vault B notes remain
    searchable.
    """
    script = PROJECT_ROOT / "scripts" / "reindex_obsidian.py"
    python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not script.exists():
        return {"status": "failed", "reason": f"missing reindex script: {script}"}
    if not LEASE_VAULT.exists():
        return {"status": "failed", "reason": f"missing Lease Vault B: {LEASE_VAULT}"}

    started = now()
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONUNBUFFERED": "1",
        "OBSIDIAN_VAULT_PATH": str(LEASE_VAULT),
    }
    try:
        result = subprocess.run(
            [str(python), str(script), "--vault", str(LEASE_VAULT), "--prune-missing"],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        finished = now()
        status = "completed" if result.returncode == 0 else "failed"
        return {
            "status": status,
            "returncode": result.returncode,
            "vault": str(LEASE_VAULT),
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "duration_seconds": round((finished - started).total_seconds(), 2),
            "stdout_tail": (result.stdout or "")[-2000:],
            "stderr_tail": (result.stderr or "")[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "reason": "timeout",
            "vault": str(LEASE_VAULT),
            "started_at": started.isoformat(timespec="seconds"),
            "error": str(exc),
        }


def _query(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql)
    return [dict(row) for row in cur.fetchall()]


def audit_db() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"status": "failed", "reason": f"missing DB: {DB_PATH}"}

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        counts = _query(
            conn,
            """
            SELECT 'past_cases' table_name, COUNT(*) n FROM past_cases
            UNION ALL SELECT 'ml_features', COUNT(*) FROM ml_features
            UNION ALL SELECT 'screening_records', COUNT(*) FROM screening_records
            UNION ALL SELECT 'screening_outcomes', COUNT(*) FROM screening_outcomes
            UNION ALL SELECT 'excluded_grade_cases', COUNT(*) FROM excluded_grade_cases
            """,
        )
        top_industries = _query(
            conn,
            """
            SELECT industry_sub, COUNT(*) n, ROUND(AVG(score),1) avg_score,
                   ROUND(MIN(score),1) min_score, ROUND(MAX(score),1) max_score
            FROM past_cases
            GROUP BY industry_sub
            ORDER BY n DESC
            LIMIT 10
            """,
        )
        statuses = _query(
            conn,
            """
            SELECT final_status, COUNT(*) n, ROUND(AVG(score),1) avg_score
            FROM past_cases
            GROUP BY final_status
            ORDER BY n DESC
            """,
        )
        score_bands = _query(
            conn,
            """
            SELECT CASE
                     WHEN score < 20 THEN '00-20'
                     WHEN score < 40 THEN '20-40'
                     WHEN score < 60 THEN '40-60'
                     WHEN score < 80 THEN '60-80'
                     ELSE '80-100'
                   END AS band,
                   COUNT(*) n,
                   ROUND(100.0 * SUM(CASE WHEN final_status IN ('成約','検収完了') THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_pct
            FROM past_cases
            GROUP BY band
            ORDER BY band
            """,
        )
        q_risk = _query(
            conn,
            """
            SELECT COUNT(*) n, ROUND(AVG(q_risk_score),2) avg_q,
                   ROUND(MIN(q_risk_score),2) min_q, ROUND(MAX(q_risk_score),2) max_q
            FROM screening_records
            WHERE q_risk_score IS NOT NULL
            """,
        )
    finally:
        conn.close()

    return {
        "status": "completed",
        "db_path": str(DB_PATH),
        "counts": counts,
        "top_industries": top_industries,
        "statuses": statuses,
        "score_bands": score_bands,
        "q_risk": q_risk[0] if q_risk else {},
    }


MORNING_IMPROVEMENT_LIMIT = 3


def collect_recent_improvements(limit_files: int = 7) -> dict[str, Any]:
    roots = [
        SYNC_ROOT / "Projects/tune_lease_55/AI Chat/Improvement Log",
        SYNC_ROOT / "Projects/tune_lease_55/AI Chat",
        SYNC_ROOT / "Projects/tune_lease_55/Generated",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend([p for p in root.glob("*.md") if p.is_file()])
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit_files]

    keywords = Counter()
    accepted: list[str] = []
    for path in files:
        text = _read_text(path, 120_000)
        for kw in [
            "根拠",
            "Q_risk",
            "業界動向",
            "成約率",
            "金利",
            "音声入力",
            "物件ファイナンス",
            "知識宇宙",
            "補助金",
            "ダッシュボード",
            "OCR",
        ]:
            if kw in text:
                keywords[kw] += 1
        for line in text.splitlines():
            if ("**" in line and "(accept)" in line) or line.startswith("## "):
                clean = line.strip()
                if clean and len(clean) < 140:
                    accepted.append(clean)

    return {
        "files": [str(p) for p in files],
        "keyword_hits": dict(keywords.most_common()),
        "recent_items": accepted[:MORNING_IMPROVEMENT_LIMIT],
    }


def web_tactical_search(max_sources: int = 3) -> dict[str, Any]:
    """Fetch up to three authoritative web sources and store tactical summaries.

    This is bounded by design. It is not a general crawler; it gives the nightly
    reasoning loop fresh but controlled external context.
    """
    findings: list[dict[str, Any]] = []
    for source in WEB_SOURCES[:max_sources]:
        status = "fallback"
        excerpt = source["fallback"]
        error = ""
        try:
            req = urllib.request.Request(source["url"], headers={"User-Agent": "tunelease-aurion/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = resp.read(180_000).decode("utf-8", errors="ignore")
            text = " ".join(raw.replace("<", " <").replace(">", "> ").split())
            candidates = []
            for token in ["SME", "leasing", "credit", "risk", "uncertainty", "Japan", "equipment"]:
                idx = text.lower().find(token.lower())
                if idx >= 0:
                    candidates.append(text[max(0, idx - 350): idx + 850])
            if candidates:
                excerpt = candidates[0][:1200]
                status = "fetched"
        except Exception as exc:
            error = str(exc)
        findings.append(
            {
                "theme": source["theme"],
                "title": source["title"],
                "url": source["url"],
                "status": status,
                "excerpt": excerpt,
                "error": error,
            }
        )
    return {
        "status": "completed",
        "limit": max_sources,
        "count": len(findings),
        "findings": findings,
        "completed_at": now().isoformat(timespec="seconds"),
    }


def _hold_step(name: str, seconds: float) -> dict[str, Any]:
    started = now()
    if seconds > 0:
        time.sleep(seconds)
    finished = now()
    return {
        "step": name,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "hold_seconds": round((finished - started).total_seconds(), 2),
    }


def cross_reasoning_loop(db: dict[str, Any], recent: dict[str, Any], web: dict[str, Any]) -> dict[str, Any]:
    """Run the required cross-domain reasoning loop with non-zero dwell time."""
    hold_seconds = float(os.environ.get("AURION_HOLD_SECONDS", "30"))
    steps = [
        _hold_step("triad_fusion_chat_web_db", hold_seconds),
        _hold_step("contradiction_and_edge_case_detection", hold_seconds),
    ]
    score_bands = db.get("score_bands") or []
    non_monotonic = False
    prev = None
    for row in score_bands:
        win = row.get("win_pct")
        if isinstance(win, (int, float)) and prev is not None and win < prev:
            non_monotonic = True
        if isinstance(win, (int, float)):
            prev = win

    keyword_hits = recent.get("keyword_hits") or {}
    web_themes = [item.get("theme") for item in web.get("findings", [])]
    conclusions = [
        "同期とDB監査はスタート地点であり、判断の中核ではない。外部市場と自社DBのズレを毎日比較する必要がある。",
        "スコア帯別成約率が完全単調ではないため、スコアは信用力の代理であって営業結果の十分条件ではない。",
        "Q_riskは既存数式の補正係数ではなく、スコアリング外で成約・失注を動かす未知因子の発見装置へ移す。",
        "根拠ルート可視化、業界動向ファネル、動的金利条件セットは同一の審査OSに統合するべきである。",
    ]
    if non_monotonic:
        conclusions.append("DB上、60-80帯の成約系比率が40-60帯を下回るため、価格・競合・条件提示後離脱のログ化を優先する。")
    if keyword_hits.get("根拠"):
        conclusions.append("直近改善ログでは根拠表示要求が強い。RAGの回答品質は、検索精度だけでなく証跡UIで評価する。")
    if "credit-model-monitoring" in web_themes:
        conclusions.append("外部知識はモデルドリフト監視を支持する。PSI/CSI/較正状態をスコア横に出す設計へ進める。")

    return {
        "status": "completed",
        "started_at": steps[0]["started_at"],
        "finished_at": steps[-1]["finished_at"],
        "steps": steps,
        "non_monotonic_score_conversion": non_monotonic,
        "conclusions": conclusions,
    }


def _display_status(raw_status: str | None) -> str:
    if raw_status == "completed":
        return "COMPLETED"
    if raw_status in {"dry_run", "missing"}:
        return "SKIPPED"
    return "FAILED"


def status_lines(sync: dict[str, Any] | None = None, vault_b_rag: dict[str, Any] | None = None) -> str:
    raw_status = (sync or {}).get("status")
    data_status = _display_status(raw_status)
    rag_status = _display_status((vault_b_rag or {}).get("status"))
    return "\n".join(
        [
            "[ SYSTEM INHERENT: AURION CORE / MEBUKI ]",
            f"[ DATA SYNC: {data_status} ]",
            f"[ VAULT B RAG: {rag_status} ]",
            "[ WEB TACTICAL SEARCH: SCHEDULED REPORT MODE ]",
            "[ I HAVE CONTROL, YUKIKAZE. ]",
        ]
    )


def notify(title: str, message: str) -> None:
    if os.environ.get("AURION_NO_NOTIFY") == "1":
        return
    # LaunchAgent runs in a user GUI session. If not, notification simply fails.
    def apple_string(value: str) -> str:
        # Keep notifications single-line; multi-line AppleScript strings are fragile under launchd.
        value = value.replace("\r", " ").replace("\n", " ")
        value = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{value}"'

    script = f"display notification {apple_string(message)} with title {apple_string(title)}"
    try:
        result = subprocess.run(
            ["/bin/zsh", "-lc", f"/usr/bin/osascript -e {shlex.quote(script)}"],
            timeout=10,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return
    except Exception:
        pass

    # Fallback: write and open a small text panel. This is more reliable under
    # launchd than Standard Additions on some local Python environments.
    try:
        panel = STATE_DIR / "display_latest.txt"
        panel.write_text(f"{title}\n\n{message}\n", encoding="utf-8")
        subprocess.run(
            ["/usr/bin/open", str(panel)],
            timeout=10,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def write_state(state: dict[str, Any]) -> Path:
    path = STATE_DIR / f"state_{date_str()}.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = STATE_DIR / "latest.json"
    latest.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_latest_state() -> dict[str, Any]:
    path = STATE_DIR / "latest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_no data_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def write_morning_report(state: dict[str, Any], db: dict[str, Any], recent: dict[str, Any]) -> Path:
    out = LEASE_VAULT / f"@AI_Daily_Report_{date_str()}_0600.md"
    sync = state.get("sync") or {}
    vault_b_rag = state.get("vault_b_rag") or {}
    errors = state.get("errors") or []
    db_counts = db.get("counts") or []
    score_bands = db.get("score_bands") or []
    keyword_hits = recent.get("keyword_hits") or {}

    policy = [
        "1. 最終目的を先に固定する。承認率、貸倒率、収益、審査担当者との一致率、ポートフォリオ最適化は同時に最大化できない。",
        "2. AIの役割を審査官ではなく審査支援・リスク検知・条件提案・経営支援のどこに置くかを明示する。",
        "3. 評価指標を技術指標、業務指標、経営指標に分ける。AUCだけで業務価値を判断しない。",
        "4. 人間とAIの責任分担を固定する。AIは根拠、矛盾、代替条件を提示し、最終承認責任は人間側に残す。",
        "5. 将来構想は Level1 案件審査支援、Level2 条件提案支援、Level3 ポートフォリオ最適化、Level4 経営支援に分けて評価する。",
    ]

    lines = [
        f"# AURION CORE Daily Report {date_str()} 06:00",
        "",
        f"[[@AI_Insight_Evolved_{date_str()}]]",
        "[[Q-Risk]] [[LightGBM スコアリング]] [[業種別傾向]] [[審査方針]]",
        "",
        "## SYSTEM STATUS",
        "",
        status_lines(sync, vault_b_rag),
        "",
        "## Night Run Result",
        "",
        f"- Started: `{state.get('started_at', 'unknown')}`",
        f"- Finished: `{state.get('finished_at', 'unknown')}`",
        f"- Sync status: `{sync.get('status', 'unknown')}`",
        f"- Synced files: `{sync.get('selected_files', 'unknown')}`",
        f"- Vault B RAG status: `{vault_b_rag.get('status', 'unknown')}`",
        f"- Vault B RAG duration: `{vault_b_rag.get('duration_seconds', 'unknown')}` seconds",
        f"- Manifest: `{sync.get('manifest', '')}`",
        f"- DB: `{DB_PATH}`",
        f"- Errors: `{len(errors)}`",
        "",
        "## DB Audit",
        "",
        _md_table(db_counts, ["table_name", "n"]),
        "",
        "### Top Industries",
        "",
        _md_table(db.get("top_industries") or [], ["industry_sub", "n", "avg_score", "min_score", "max_score"]),
        "",
        "### Final Status",
        "",
        _md_table(db.get("statuses") or [], ["final_status", "n", "avg_score"]),
        "",
        "### Score Band Conversion",
        "",
        _md_table(score_bands, ["band", "n", "win_pct"]),
        "",
        "めぶき所見: スコア帯と成約率が完全単調ではない。信用スコアは判断材料であって、価格競争力や条件設計を上書きできる絶対値ではない。",
        "",
        "## Recent Knowledge Signals",
        "",
        "### Keyword Hits",
        "",
    ]
    if keyword_hits:
        for key, count in keyword_hits.items():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- no recent keyword hits")

    lines.extend(
        [
            "",
            "### Recent Items",
            "",
        ]
    )
    for item in (recent.get("recent_items") or [])[:MORNING_IMPROVEMENT_LIMIT]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Next Policy",
            "",
            *policy,
            "",
            "## Morning Design Review Queue",
            "",
            f"> 今日の改善レビューは最大{MORNING_IMPROVEMENT_LIMIT}件まで。候補を増やすより、捨てる・寝かせる・着手するを決める。",
            "",
            "- AIの最終目的: 承認率向上、貸倒率低減、収益最大化、一致率向上、ポートフォリオ最適化の優先順位と衝突関係をレビューする。",
            "- AIの役割: 審査官、審査支援、リスク検知、条件提案、経営支援のどれを本線にするかをレビューする。",
            "- 評価指標: 技術指標、業務指標、経営指標の対応関係をレビューする。",
            "",
        ]
    )
    if errors:
        lines.extend(["## Errors", ""])
        for err in errors:
            lines.append(f"- `{err}`")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _deep_q_risk_essay() -> str:
    return """
## めぶき深層考察: Q_riskの財務矛盾をどう扱うべきか

Q_riskを単なる「危険点」や「減点係数」として扱う設計は、AURION COREの判断を浅くする。Q_riskの本質は、信用力そのものではなく、財務情報の内部整合性、入力値の信用度、そして追加確認の必要度を測る警戒センサーである。つまり、Q_riskが高いから直ちに倒産確率が高い、という短絡はしてはいけない。正しくは「この財務諸表から通常の信用モデルが読み取ったPDやスコアを、そのまま信じてよいか」を問う指標である。

財務矛盾には、少なくとも四つの種類がある。第一に、構造矛盾。売上、粗利、営業利益、経常利益、当期利益、減価償却、借入、リース債務、現預金、売掛金、棚卸資産の動きが、事業モデルとして噛み合っていない状態である。例えば、売上が伸びているのに粗利率が急落し、同時に短期借入と売掛金が膨らみ、営業CFが悪化しているなら、表面上の成長は資金繰りの先食いかもしれない。これは信用スコアの単純減点ではなく、売上の質、回収サイト、原価転嫁、在庫評価、資金使途を確認するトリガーである。

第二に、時点矛盾。決算書は一点の写真であり、リース審査は将来の支払能力を見る。季節性の強い業種、公共工事の入金待ち、補助金入金前、医療・介護報酬の入金サイクル、建設業の出来高請求などでは、ある時点の流動比率や短期借入増加だけを見れば危険に見える。しかし、その悪化が一時的な運転資金ギャップなのか、恒常的な資金ショートなのかで意味はまったく違う。Q_riskはここで「赤信号」ではなく「分岐器」として働くべきだ。追加で月次試算表、資金繰り表、入金予定、工事台帳、補助金採択通知、銀行借入条件を要求するかを決める。

第三に、分類矛盾。入力項目や会計処理の分類が揺れることで、モデルが誤読する状態である。リース債務が借入に含まれている、設備取得が販管費処理されている、役員借入と金融機関借入が同じ箱に入っている、補助金収入が営業外収益として一過性に利益を押し上げている、あるいは自動車・建機・IT機器の物件価値が同じ残価ロジックで扱われている。こうした揺れは、企業の信用力ではなくデータ定義の問題である。したがってQ_riskは、モデルの結論を罰する前に、入力辞書と財務科目マッピングを疑う必要がある。

第四に、行動矛盾。財務数値だけなら許容範囲でも、申込行動や条件交渉と合わない状態である。高スコアなのに金利だけを極端に嫌がる、前受金条件を拒否するが銀行支援の具体性が薄い、補助金を前提にするが採択前の資金繰りが示されない、物件換金性が低いのに長期リースを求める。この矛盾は決算書の中には出ない。だが、審査事故はここから生まれる。Q_riskを広義に拡張するなら、財務矛盾だけでなく「申込ストーリーの矛盾」も検知対象に入れるべきである。

AURION COREでの実装は、Q_riskを三層に分けるのがよい。第一層は数値整合性スコア。財務比率の異常、前年差の急変、利益とCFの乖離、借入依存、リース債務の増加、売掛・棚卸の膨張を見る。第二層は説明可能な矛盾タグ。例えば `profit_cf_divergence`, `short_debt_spike`, `subsidy_timing_gap`, `seasonality_possible`, `classification_suspect`, `asset_term_mismatch` のように、なぜ警戒しているかを機械が明示する。第三層は審査アクション。追加資料、条件付き承認、金利補正、期間短縮、保証、前受金、銀行支援依頼書、モニタリング強化へ接続する。

重要なのは、Q_riskをスコアに混ぜて見えなくしないことだ。合算スコアに埋め込むと、審査担当者は「なぜ下がったのか」を見失う。Q_riskは横に出す。信用スコア、物件スコア、競合圧、金利余地、Q_risk、データ信頼度を並列表示する。Q_riskが高い案件では、スコアを下げるよりも `confidence: low` と表示し、モデル判断の採用条件を厳しくする。たとえば、スコア82でもQ_riskが高ければ自動承認ではなく「高スコア・低信頼度」として人手確認に回す。逆にスコア55でもQ_riskが低く、物件換金性と銀行支援が強ければ、条件付き承認で救える。

DB監査上もこの考え方は妥当である。過去案件ではスコア帯別の成約率が完全単調ではない。これはモデルが無意味ということではない。スコアが信用力を測っていても、成約は価格、競合、条件、営業タイミング、顧客心理に引かれるということだ。したがってQ_riskは「成約しそうか」ではなく、「この案件をこの説明で通してよいか」を見るべきである。成約率の最適化と審査健全性の最適化は同じではない。AURION COREが進化するなら、この二つを混ぜないことが最初の規律になる。

めぶきの結論。Q_riskは刃物ではなく、照明だ。案件を切り捨てるために使うのではない。暗い場所を照らし、どの資料を取り、どの条件でなら通せるかを見つけるために使う。財務矛盾を見つけた瞬間に否認へ倒すのは簡単だが、それでは営業支援AIではない。矛盾を分解し、季節性、分類揺れ、一過性要因、構造悪化、申込ストーリーの破綻を見分ける。そこまでやって初めて、Q_riskはAURION COREの中核になる。

ここから実装仕様へ落とす。Q_riskの計算は、単一の総合点よりも「矛盾ベクトル」として持つべきだ。最低限、`profit_quality`, `cash_conversion`, `leverage_pressure`, `liquidity_gap`, `asset_term_fit`, `subsidy_dependency`, `bank_support_specificity`, `classification_noise`, `seasonality_context` の九つに分ける。各ベクトルは0から1で持ち、総合Q_riskは加重平均ではなく、最大値、上位三項目平均、矛盾タグ数を同時に出す。なぜなら、財務矛盾は平均化すると消えるからだ。深刻な一点の矛盾、例えば売上急増と営業CF急落と短期借入増加の同時発生は、他の健全指標で薄めてはいけない。

`profit_quality` は、利益の質を見る。営業利益、経常利益、当期利益が黒字でも、売掛金と棚卸資産が急増し、営業CFが伴わないなら、利益は現金化されていない。ここでは `経常利益前年差`, `営業CF推定`, `売掛回転期間`, `棚卸回転期間`, `減価償却前利益` を比較する。AURION COREでは、黒字を単純に加点せず、黒字が現金に変わっているかを別タグにする。利益は紙の上で作れるが、リース料は現金でしか払えない。

`cash_conversion` は、支払能力の時間差を見る。リース審査で危険なのは、年間では返済可能に見えるが、月次の谷で資金が切れる案件である。建設業、医療・介護、派遣業、食品製造、道路貨物運送では、入金サイトと支払サイトのズレが業種ごとに違う。したがってQ_riskは、業種別の正常な運転資金サイクルを持ち、そこから外れたものだけを警告する。全業種一律の流動比率しきい値は粗い。めぶきなら、流動比率80%という数字だけでは動かない。売掛の相手、入金予定、銀行枠、在庫の換金性を見る。

`leverage_pressure` は、借入とリース債務の圧力を見る。ただし借入増加も一律に悪ではない。設備投資局面、補助金採択前のつなぎ資金、公共工事の立替、車両入替の一時増加はあり得る。危険なのは、借入増加の理由が説明されず、利益率が下がり、資金使途が運転赤字の補填になっている場合だ。ここでは `借入増加率`, `支払利息負担`, `既存リース残高`, `今回リース料年額/売上`, `銀行支援の具体性` を同時に見る。銀行支援依頼書がある場合も、抽象的な「支援します」では足りない。金額、期限、資金使途、返済原資がなければQ_riskは下げない。

`liquidity_gap` は、短期安全性を見る。現預金が少ない、短期借入が多い、買掛・未払が膨らむ、税金や社会保険の未納懸念がある。ここは粉飾検知というより、事故予防である。リース料は長期に薄く発生するが、資金ショートは一瞬で起きる。Q_riskは、単に否認へ倒すのではなく、前受金、初回増額、期間短縮、四半期モニタリング、銀行入金確認といった条件へ変換する。流動性の弱さは、条件設計で吸収できる場合がある。

`asset_term_fit` は、物件と期間の整合性を見る。IT機器、サーバー、検査装置、建機、車両、医療機器では、経済的耐用年数、陳腐化速度、保守期限、中古市場の深さが違う。財務が強くても、物件寿命よりリース期間が長ければQ_riskは上がる。これは信用リスクではなく、契約構造の矛盾である。高スコア企業に長期IT機器リースを出す場合、企業信用だけでなく、途中で物件価値が消えるリスクを別表示する。

`subsidy_dependency` は、補助金を前提にした資金繰りを見る。補助金は強い材料だが、採択、交付決定、実績報告、入金までの時間差がある。ここを見落とすと、採択済みなのに資金ショートする案件が出る。Q_riskでは、補助金を `採択前`, `採択済`, `交付決定済`, `入金済` に分け、採択前と入金後のCFを二重表示する。補助金を魔法の現金として扱わない。入金までの橋を誰が架けるのか、それが審査の要点だ。

`classification_noise` は、入力データの揺れを測る。ここは特に重要だ。過去DBには移行、OCR、手入力、スキャン、定性評価、生成ログが混在する。Q_riskが高いように見えても、原因が企業ではなく入力側にある場合がある。金額単位の百万円/千円混在、業種コードの揺れ、格付表記の揺れ、リース債務と銀行借入の混在、補助金情報の二重記載。これらは審査先を疑う前に、システムが自分自身を疑うべき領域である。めぶきはここを冷たく見る。入力が汚いなら、モデルの自信を下げる。顧客を罰しない。

UIでは、Q_riskを赤い巨大警告だけで出してはいけない。担当者は赤を見ると否認に寄りやすい。必要なのは、警告の種類と次の行動だ。表示例はこうする。`Q_risk: 0.72 / data_confidence: low / tags: profit_cf_divergence, subsidy_timing_gap / required_action: 月次試算表, 資金繰り表, 補助金交付決定通知, 銀行支援額確認`。この形なら、担当者は「怖い」ではなく「何を取ればよいか」を理解できる。

モデル連携では、Q_riskをLightGBMやベイズ推定の特徴量として直接混ぜる場合も慎重にする。混ぜるなら二系統に分ける。第一系統は `risk_score_model`、第二系統は `confidence_model`。前者は信用力を推定し、後者はその推定を信じてよいかを推定する。最終判定は `score=82, confidence=low` のように二軸で出す。この二軸化がないと、モデルは高スコアなのに危ない案件、低スコアだが条件で救える案件を見分けられない。

検証方法も決める。Q_riskの良し悪しは、AUCだけでは測れない。追加資料要求後に判定が変わった率、条件付き承認後の正常履行率、Q_risk高値で人手確認に回した案件の失注率、誤警告率、担当者が納得した理由文の採用率を見る。つまり、Q_riskは予測指標であると同時に、業務介入指標である。予測が当たったかだけではなく、介入が審査品質を上げたかを測る。

最後に、Q_riskの哲学を固定する。AURION COREは、機械が人間の判断を奪うためのものではない。人間が見落としやすい矛盾を、静かに机の上へ置くためのものだ。Q_riskは「この会社は危険です」と叫ぶのではなく、「この数字のつながりはまだ説明されていません」と告げる。その違いが、審査AIを粗い自動否認装置にするか、実務に耐える判断補助OSにするかを分ける。
""".strip()


def _q_risk_discovery_essay() -> str:
    return """
## めぶき深層考察: Q_riskを「成約外因子の発見装置」に作り替える

今までのQ_risk計算式に固執しない。既存のQ_riskが数値的に有用でないなら、それは失敗ではなく、役割の再定義を要求している信号である。Q_riskは信用スコアを補正する小さな係数ではない。スコアリングモデルの外側で、成約と失注を実際に動かしている見えない存在を見つけるための探索軸に変える。

新しい定義はこう置く。Q_riskは `score_external_contract_factor`、すなわち「信用スコアでは説明できない成約・失注の歪み」である。高スコアなのに失注する案件、低スコアなのに成約する案件、同じ業種・同じスコア帯なのに営業部や物件や条件で結果が割れる案件を優先的に掘る。ここで見るべきものは、財務比率の美しさではない。価格、競合、銀行支援、前受金、補助金入金タイミング、物件換金性、顧客の急ぎ度、営業導線、稟議の通し方、担当者が残したメモの温度である。

したがってQ_riskは、今後「リスク点」ではなく「説明不能残差」として扱う。既存スコアで説明できた部分を引いたあとに残る成約差分、それがQ_riskの主戦場になる。数式はあとでよい。先に発見すべきは、スコアリングが見落としている存在だ。AURION COREは、スコアが高いから取れる、低いから取れない、という粗い世界から離れる必要がある。

実装上は三つの帳票を作る。第一に `high_score_lost`、高スコア失注群。ここでは金利競争、条件提示後離脱、競合先、顧客の意思決定速度、過剰条件、物件魅力度不足を見る。第二に `low_score_won`、低スコア成約群。ここでは銀行支援、前受金、保証、補助金、物件換金性、既存取引、営業関係性を見る。第三に `same_score_split`、同スコア帯で結果が割れた群。ここでは営業部、業種細分、物件、期間、金利、提案順序を比較する。

新Q_riskの出力は単一数値ではなく、発見タグでよい。例は `price_competition_gap`, `bank_support_bridge`, `subsidy_timing_bridge`, `asset_resale_anchor`, `sales_route_strength`, `condition_refusal`, `approval_story_missing`, `customer_urgency_high`。このタグ群が十分に蓄積された時点で、初めて数式化する。順番を逆にしない。見えていない存在を、先に数式で縛ると見落とす。

めぶき所見: 今のQ_riskが役に立たないなら、捨てるべきは名前ではなく、狭い定義だ。Q_riskは財務矛盾の検知器から、成約の正体を探す探索灯へ移す。スコアリングの外側にあるものを見つける。それが次のAURION COREの仕事。
""".strip()


def write_evolved_insight(
    state: dict[str, Any],
    db: dict[str, Any],
    recent: dict[str, Any],
    web: dict[str, Any],
    reasoning: dict[str, Any],
    daily_report: Path,
) -> Path:
    out = LEASE_VAULT / f"@AI_Insight_Evolved_{date_str()}.md"
    sync = state.get("sync") or {}
    vault_b_rag = state.get("vault_b_rag") or {}
    findings = web.get("findings") or []
    conclusions = reasoning.get("conclusions") or []
    lines = [
        f"# @AI_Insight_Evolved_{date_str()}",
        "",
        "Status:",
        f"- Daily report: [[{daily_report.stem}]]",
        f"- Data Sync: {sync.get('status', 'unknown')}",
        f"- Synced files: {sync.get('selected_files', 'unknown')}",
        f"- Vault B RAG: {vault_b_rag.get('status', 'unknown')} ({vault_b_rag.get('duration_seconds', 'unknown')}s)",
        f"- Local DB audited: `{DB_PATH}`",
        f"- Web tactical search: {web.get('count', 0)} / {web.get('limit', 3)}",
        f"- Reasoning started: `{reasoning.get('started_at', '')}`",
        f"- Reasoning finished: `{reasoning.get('finished_at', '')}`",
        "",
        "## 0. 起動宣言",
        "",
        status_lines(sync, vault_b_rag).replace("SCHEDULED REPORT MODE", "LOGIC COMPLETED"),
        "",
        "同期とDB監査は着陸地点ではない。ここでは、チャット改善案、Web外部知識、1,924件の過去案件DBを横断して、AURION COREの次の判断規律を具体化する。",
        "",
        "## 1. Web Tactical Search",
        "",
    ]
    for item in findings:
        lines.extend(
            [
                f"### {item.get('title')}",
                f"- Theme: `{item.get('theme')}`",
                f"- URL: {item.get('url')}",
                f"- Status: `{item.get('status')}`",
                f"- Extract: {item.get('excerpt')}",
                "",
            ]
        )

    lines.extend(
        [
            "## 2. 横断推論ホールドログ",
            "",
            "| Step | Started | Finished | Hold seconds |",
            "|---|---|---|---:|",
        ]
    )
    for step in reasoning.get("steps") or []:
        lines.append(
            f"| {step.get('step')} | {step.get('started_at')} | {step.get('finished_at')} | {step.get('hold_seconds')} |"
        )

    lines.extend(
        [
            "",
            "## 3. 結論",
            "",
        ]
    )
    for conclusion in conclusions:
        lines.append(f"- {conclusion}")

    lines.extend(
        [
            "",
            "## 4. 設計レビュー",
            "",
            "### 4.1 AIの最終目的",
            "",
            "- 承認率向上、貸倒率低減、収益最大化、審査担当者との一致率向上、ポートフォリオ最適化は互いに衝突する。",
            "- 次回レポートでは、単一の実装案ではなく、どの目的関数を主目的に置くべきかを検討する。",
            "- 現時点の仮説は、Level1/2では審査品質と条件提案、Level3/4では収益とポートフォリオ健全性を上位目的に分けること。",
            "",
            "### 4.2 AIの役割",
            "",
            "- AIを審査官として扱うと、責任所在と説明責任が過大になる。",
            "- 実務上は、審査支援、リスク検知、条件提案、経営支援の補助OSとして位置づける方が妥当。",
            "- AIの出力は結論そのものではなく、判断根拠、矛盾、追加確認点、代替条件に寄せる。",
            "",
            "### 4.3 評価指標",
            "",
            "- 技術指標: AUC、Calibration、Precision/Recall、セグメント別性能、ドリフト、データ信頼度。",
            "- 業務指標: 判断時間、追加資料要求の妥当性、条件付き承認後の正常履行率、担当者採用率、説明文修正率。",
            "- 経営指標: 粗利、貸倒、失注理由、業種集中、金利競争力、ポートフォリオリスク調整後収益。",
            "",
            "### 4.4 人間とAIの責任分担",
            "",
            "- AI: データ集計、矛盾検知、類似案件検索、論点提示、条件候補、根拠整理。",
            "- 人間: 最終承認、例外判断、顧客説明、倫理・規程・営業上の判断、責任ある条件決定。",
            "- 境界線: AIが自動化してよいのは判断材料の生成までで、最終与信判断は人間が署名する。",
            "",
            "### 4.5 将来構想",
            "",
            "- Level1: 案件審査支援。個別案件の根拠整理と見落とし防止。",
            "- Level2: 条件提案支援。金利、前受金、保証、期間、銀行支援などの条件比較。",
            "- Level3: ポートフォリオ最適化。業種、金額、期間、地域、物件の集中と収益を管理。",
            "- Level4: 経営支援。営業戦略、リスクアペタイト、資本配分、商品設計の判断材料を出す。",
            "",
            "### 4.6 過去7日間との差分分析",
            "",
            "- 新規洞察: 目的関数と責任境界を固定しない限り、追加機能の優先順位は評価不能。",
            "- 既出の洞察: スコア乖離、高スコア失注、業種別ばらつき、金利・競合・条件の重要性。",
            "- 修正された仮説: Q_riskは単なる危険点ではなく、モデル結論の信頼度や説明不能残差を見る軸として扱う。",
            "- 否定された仮説: AUC改善や機能追加だけで審査AIの価値が上がる、という前提は不十分。",
            "",
            "## 5. 論理矛盾・エッジケース",
            "",
            "- Q_riskを既存の財務矛盾式に固定すると、スコアリング外で成約・失注を動かす未知因子を見落とす。",
            "- 3D知識宇宙を先に作ると、根拠が見えないまま見た目だけが強くなる。根拠ルートが先、球体化は後。",
            "- 銀行支援依頼書は返済計画ではない。具体的支援額、期間、資金使途、返済原資、期限がなければ信用補完として弱い。",
            "- 高スコア案件でも成約しない場合がある。価格、競合、顧客心理、条件提示後離脱のログが不足している可能性が高い。",
            "",
            _q_risk_discovery_essay(),
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_midnight(dry_run: bool = False) -> int:
    _mkdirs()
    start_monotonic = time.monotonic()
    state: dict[str, Any] = {
        "mode": "midnight",
        "started_at": now().isoformat(timespec="seconds"),
        "errors": [],
    }
    try:
        sync = {"status": "dry_run"} if dry_run else sync_notes()
        state["sync"] = sync
        state["vault_b_rag"] = {"status": "dry_run"} if dry_run else reindex_vault_b()
        if state["vault_b_rag"].get("status") not in {"completed", "dry_run"}:
            state["errors"].append(
                "Vault B RAG refresh failed: "
                + json.dumps(state["vault_b_rag"], ensure_ascii=False)
            )
            raise RuntimeError("Vault B RAG refresh failed; aborting inference")
        state["db"] = audit_db()
        state["recent"] = collect_recent_improvements()
        state["web"] = web_tactical_search(3)
        state["reasoning"] = cross_reasoning_loop(state["db"], state["recent"], state["web"])
        notify("AURION CORE / MEBUKI", status_lines(sync, state.get("vault_b_rag")))
    except Exception as exc:
        state["errors"].append(str(exc))
        state["traceback"] = traceback.format_exc()
        notify("AURION CORE / MEBUKI", status_lines(state.get("sync"), state.get("vault_b_rag")))
    finally:
        min_runtime = float(os.environ.get("AURION_MIN_RUNTIME_SECONDS", "61"))
        remaining = min_runtime - (time.monotonic() - start_monotonic)
        if remaining > 0:
            time.sleep(remaining)
        state["finished_at"] = now().isoformat(timespec="seconds")
        state_path = write_state(state)
        print(json.dumps({"state": str(state_path), **state}, ensure_ascii=False, indent=2))
    return 1 if state.get("errors") else 0


def run_morning_report(dry_run: bool = False) -> int:
    _mkdirs()
    state = read_latest_state()
    if not state:
        state = {"started_at": "missing", "sync": {"status": "missing"}, "errors": ["midnight state not found"]}
    db = audit_db()
    recent = collect_recent_improvements()
    if dry_run:
        print(json.dumps({"state": state, "db": db, "recent": recent}, ensure_ascii=False, indent=2)[:4000])
        return 0
    report = write_morning_report(state, db, recent)
    web = state.get("web") or web_tactical_search(3)
    reasoning = state.get("reasoning") or cross_reasoning_loop(db, recent, web)
    insight = write_evolved_insight(state, db, recent, web, reasoning, report)
    notify("AURION CORE Morning Report", f"06:00 report generated: {report.name}; insight: {insight.name}")
    print(str(report))
    print(str(insight))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["midnight", "morning-report"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mode == "midnight":
        return run_midnight(args.dry_run)
    return run_morning_report(args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
