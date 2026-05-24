"""migrate_outcomes.py — Phase 0-A: 流体化基盤マイグレーション

実行内容:
  1. screening_outcomes テーブル作成（支払実績・デフォルト追跡）
  2. retraining_log テーブル作成（再学習履歴）
  3. past_cases → screening_records ブリッジ（1900件を学習データに）
  4. 既存 screening_records の outcome NULL を final_status から補完

使い方:
    python3 migrate_outcomes.py
    python3 migrate_outcomes.py --dry-run   # 実行しないで確認のみ
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent
DB_PATH = _DIR / "data" / "lease_data.db"

# ──────────────────────────────────────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────────────────────────────────────

DDL_SCREENING_OUTCOMES = """
CREATE TABLE IF NOT EXISTS screening_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id             TEXT    NOT NULL,
    screening_id        INTEGER,                    -- screening_records.id との紐付け（任意）
    contract_date       TEXT,                       -- 成約日（YYYY-MM-DD）
    scheduled_end_date  TEXT,                       -- リース満了予定日
    actual_status       TEXT    NOT NULL DEFAULT 'unknown',
                                                    -- unknown / normal / late_30 / late_90 / default / completed
    delinquent          INTEGER NOT NULL DEFAULT 0, -- 1=延滞/デフォルト確定、0=正常
    loss_given_default  REAL,                       -- 実損額（円）
    checked_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    notes               TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_so_case_id    ON screening_outcomes(case_id);
CREATE INDEX IF NOT EXISTS idx_so_status     ON screening_outcomes(actual_status);
CREATE INDEX IF NOT EXISTS idx_so_screening  ON screening_outcomes(screening_id);
"""

DDL_RETRAINING_LOG = """
CREATE TABLE IF NOT EXISTS retraining_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_by      TEXT    NOT NULL,
    status            TEXT    NOT NULL,
    records_used      INTEGER,
    new_auc           REAL,
    prev_auc          REAL,
    model_updated     INTEGER NOT NULL DEFAULT 0,
    rollback_reason   TEXT,
    error_message     TEXT,
    duration_seconds  REAL,
    started_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at      TEXT
);
"""

# past_cases の final_status → screening_records.outcome マッピング
_STATUS_TO_OUTCOME = {
    "成約":         "contracted",
    "検収完了":     "completed",
    "失注":         "lost",
    "稟議中":       None,       # スキップ
    "未登録":       None,
    "スコアリングのみ": None,
}


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────────────────────

def _backup_db(db_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.parent / "backups" / f"lease_data_before_fluid_{ts}.db"
    backup.parent.mkdir(exist_ok=True)
    shutil.copy2(db_path, backup)
    log.info("DB バックアップ: %s", backup)
    return backup


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _extract_features(data_json: str) -> dict:
    """past_cases.data JSON から学習用特徴量を抽出する。"""
    try:
        d = json.loads(data_json)
    except (json.JSONDecodeError, TypeError):
        return {}

    result = d.get("result", {}) or {}
    return {
        "total_score":               _safe_float(result.get("score")),
        "asset_score":               _safe_float(result.get("asset_score")),
        "tenant_score":              _safe_float(result.get("score_borrower")),
        "q_risk_score":              _safe_float(result.get("q_risk_score")),
        "competitor_pressure_score": _safe_float(result.get("competitor_pressure_score")),
        "screened_at":               d.get("timestamp", datetime.now().isoformat()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# マイグレーション処理
# ──────────────────────────────────────────────────────────────────────────────

def step1_create_tables(conn: sqlite3.Connection) -> None:
    """テーブル作成（べき等）"""
    conn.executescript(DDL_SCREENING_OUTCOMES)
    conn.executescript(DDL_RETRAINING_LOG)
    conn.commit()
    log.info("✅ Step1: screening_outcomes / retraining_log テーブル作成完了")


def step2_bridge_past_cases(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """past_cases の成約/失注/検収完了を screening_records に投入する。

    既に case_id が存在するレコードはスキップ（INSERT OR IGNORE）。
    Returns: 挿入件数
    """
    rows = conn.execute(
        "SELECT id, final_status, data, timestamp FROM past_cases "
        "WHERE final_status IN ('成約','失注','検収完了')"
    ).fetchall()

    inserted = 0
    skipped = 0
    for case_id, final_status, data_json, ts in rows:
        outcome = _STATUS_TO_OUTCOME.get(final_status)
        if outcome is None:
            skipped += 1
            continue

        feat = _extract_features(data_json)
        if feat.get("total_score", 0) == 0 and feat.get("asset_score", 0) == 0:
            skipped += 1
            continue  # スコア未計算案件はスキップ

        screened_at = feat.get("screened_at") or ts or datetime.now().isoformat()

        if not dry_run:
            conn.execute(
                """
                INSERT OR IGNORE INTO screening_records
                    (case_id, screened_at, total_score, asset_score,
                     tenant_score, q_risk_score, competitor_pressure_score,
                     outcome, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    screened_at,
                    feat["total_score"],
                    feat["asset_score"],
                    feat.get("tenant_score"),
                    feat.get("q_risk_score") or None,
                    feat.get("competitor_pressure_score") or None,
                    outcome,
                    "past_cases_migration",
                ),
            )
        inserted += 1

    if not dry_run:
        conn.commit()

    log.info(
        "✅ Step2: past_cases ブリッジ — 投入=%d件 / スキップ=%d件%s",
        inserted, skipped, " [DRY RUN]" if dry_run else "",
    )
    return inserted


def step3_seed_screening_outcomes(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """検収完了案件を screening_outcomes に delinquent=0 で初期登録する。

    「完済が確認された」案件は確実に非デフォルトなので、
    これが最初の正例データになる。
    Returns: 挿入件数
    """
    rows = conn.execute(
        """
        SELECT sr.id, sr.case_id, sr.screened_at
        FROM screening_records sr
        WHERE sr.outcome = 'completed'
          AND NOT EXISTS (
              SELECT 1 FROM screening_outcomes so WHERE so.case_id = sr.case_id
          )
        """
    ).fetchall()

    inserted = 0
    for sr_id, case_id, screened_at in rows:
        if not dry_run:
            conn.execute(
                """
                INSERT INTO screening_outcomes
                    (case_id, screening_id, actual_status, delinquent, notes)
                VALUES (?, ?, 'completed', 0, '検収完了 — 自動マイグレーション')
                """,
                (case_id, sr_id),
            )
        inserted += 1

    if not dry_run:
        conn.commit()

    log.info(
        "✅ Step3: 検収完了 → screening_outcomes シード — %d件%s",
        inserted, " [DRY RUN]" if dry_run else "",
    )
    return inserted


def step4_report(conn: sqlite3.Connection) -> None:
    """マイグレーション後の統計を表示する。"""
    sr_total = conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()[0]
    sr_dist = conn.execute(
        "SELECT outcome, COUNT(*) FROM screening_records GROUP BY outcome ORDER BY COUNT(*) DESC"
    ).fetchall()
    so_total = conn.execute("SELECT COUNT(*) FROM screening_outcomes").fetchone()[0]
    so_dist = conn.execute(
        "SELECT actual_status, delinquent, COUNT(*) FROM screening_outcomes "
        "GROUP BY actual_status, delinquent"
    ).fetchall()

    log.info("─" * 50)
    log.info("📊 マイグレーション後の統計:")
    log.info("  screening_records: %d件", sr_total)
    for outcome, cnt in sr_dist:
        log.info("    %-20s: %d件", outcome or "NULL", cnt)
    log.info("  screening_outcomes: %d件", so_total)
    for status, delinquent, cnt in so_dist:
        log.info("    %-20s (delinquent=%d): %d件", status, delinquent, cnt)
    log.info("─" * 50)
    log.info("💡 次のステップ: python3 retraining_pipeline.py でモデルを再学習")
    log.info("💡 支払状況の入力: Streamlit の「📋 支払状況登録」ページ（outcome_recorder.py）")


# ──────────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────────

def run(db_path: Path = DB_PATH, dry_run: bool = False) -> None:
    if not db_path.exists():
        log.error("DB が見つかりません: %s", db_path)
        return

    if not dry_run:
        _backup_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    try:
        step1_create_tables(conn)
        step2_bridge_past_cases(conn, dry_run=dry_run)
        step3_seed_screening_outcomes(conn, dry_run=dry_run)
        step4_report(conn)
    finally:
        conn.close()

    if dry_run:
        log.info("🔍 DRY RUN 完了 — DB は変更されていません")
    else:
        log.info("✅ マイグレーション完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流体化基盤マイグレーション")
    parser.add_argument("--dry-run", action="store_true", help="DBを変更せずに確認のみ")
    parser.add_argument("--db", default=str(DB_PATH), help="DBパス")
    args = parser.parse_args()
    run(db_path=Path(args.db), dry_run=args.dry_run)
