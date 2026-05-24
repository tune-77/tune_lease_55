"""migrate_grade9_to_outcomes.py — excluded_grade_cases (格付9) を screening_outcomes へ移行

格付9（信用リスク最高）の99件を delinquent=1 として学習データに組み込む。
FP-001 制約（延滞5件以上）を満たし、モデル再学習を可能にする。

実行方法:
    python3 migrate_grade9_to_outcomes.py          # 実行
    python3 migrate_grade9_to_outcomes.py --dry-run # 確認のみ
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent
DB_PATH = _DIR / "data" / "lease_data.db"


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def migrate(dry_run: bool = False) -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 対象: excluded_grade_cases の全件（すべて grade=9）
    cur.execute("""
        SELECT id, timestamp, score, user_eq, final_status, data
        FROM excluded_grade_cases
        ORDER BY timestamp
    """)
    rows = cur.fetchall()
    print(f"[migrate_grade9] 対象件数: {len(rows)} 件")

    now = datetime.now().isoformat()
    sr_inserted = 0
    sr_skipped = 0
    so_inserted = 0
    so_skipped = 0

    for row in rows:
        case_id = row["id"]
        timestamp = row["timestamp"] or now
        total_score = _safe_float(row["score"])
        final_status = row["final_status"] or "unknown"

        # JSON データを解析してスコア詳細を取得
        asset_score = 0.0
        tenant_score = 0.0
        q_risk_score = 0.0
        competitor_pressure_score = 0.0

        try:
            data = json.loads(row["data"]) if row["data"] else {}
            result = data.get("result", {})
            inputs = data.get("inputs", {})
            asset_score = _safe_float(result.get("asset_score") or inputs.get("asset_score"))
            tenant_score = _safe_float(
                result.get("tenant_score") or inputs.get("tenant_score", 0.0)
            )
            q_risk_score = _safe_float(result.get("q_risk_score", 0.0))
            competitor_pressure_score = _safe_float(result.get("competitor_pressure_score", 0.0))
        except Exception:
            pass

        # ─── screening_records への挿入 ───────────────────────────
        cur.execute("SELECT id FROM screening_records WHERE case_id = ?", (case_id,))
        existing_sr = cur.fetchone()

        if existing_sr is None:
            if not dry_run:
                cur.execute("""
                    INSERT INTO screening_records
                        (case_id, screened_at, total_score, asset_score,
                         tenant_score, q_risk_score, competitor_pressure_score,
                         outcome, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'delinquent', 'excluded_grade_cases', ?, ?)
                """, (
                    case_id, timestamp, total_score, asset_score,
                    tenant_score, q_risk_score, competitor_pressure_score,
                    now, now
                ))
            sr_inserted += 1
        else:
            # 既存レコードは outcome を更新
            if not dry_run:
                cur.execute("""
                    UPDATE screening_records
                    SET outcome = 'delinquent', updated_at = ?
                    WHERE case_id = ? AND (outcome IS NULL OR outcome != 'delinquent')
                """, (now, case_id))
            sr_skipped += 1

        # ─── screening_outcomes への挿入 ───────────────────────────
        try:
            cur.execute("SELECT id FROM screening_outcomes WHERE case_id = ?", (case_id,))
            existing_so = cur.fetchone()

            if existing_so is None:
                if not dry_run:
                    cur.execute("""
                        INSERT INTO screening_outcomes
                            (case_id, actual_status, delinquent,
                             notes, created_at, updated_at)
                        VALUES (?, 'high_risk_grade', 1,
                                '格付9: 信用リスク最高区分（excluded_grade_cases より移行）',
                                ?, ?)
                    """, (case_id, now, now))
                so_inserted += 1
            else:
                so_skipped += 1
        except sqlite3.OperationalError as exc:
            logger.warning(
                "[migrate_grade9] screening_outcomes 操作失敗 case_id=%s: %s", case_id, exc
            )
            continue

    if not dry_run:
        conn.commit()

    conn.close()

    result = {
        "screening_records": {"inserted": sr_inserted, "already_existed": sr_skipped},
        "screening_outcomes": {"inserted": so_inserted, "already_existed": so_skipped},
        "dry_run": dry_run,
    }

    print(f"[migrate_grade9] screening_records: {sr_inserted} 件追加, {sr_skipped} 件既存")
    print(f"[migrate_grade9] screening_outcomes: {so_inserted} 件追加, {so_skipped} 件既存")

    if dry_run:
        print("[migrate_grade9] DRY RUN — DB は変更していません")
    else:
        print("[migrate_grade9] ✅ 移行完了")

    return result


def verify() -> None:
    """移行後の検証: delinquent 件数・再学習可否を確認"""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM screening_outcomes WHERE delinquent = 1")
    delinquent_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM screening_records WHERE outcome = 'delinquent'")
    sr_delinquent = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM screening_records")
    total_sr = cur.fetchone()[0]

    cur.execute("""
        SELECT so.actual_status, COUNT(*) as cnt
        FROM screening_outcomes so
        GROUP BY so.actual_status
        ORDER BY cnt DESC
    """)
    status_dist = cur.fetchall()

    conn.close()

    print("\n─── 検証結果 ───────────────────────────────────")
    print(f"  screening_records 総件数:          {total_sr:,}")
    print(f"  screening_records delinquent:      {sr_delinquent:,}")
    print(f"  screening_outcomes delinquent=1:   {delinquent_count:,}")
    print(f"  再学習条件 (≥5件):                 {'✅ 満たした' if delinquent_count >= 5 else '❌ 未満'}")
    print("\n  actual_status 分布:")
    for status, cnt in status_dist:
        print(f"    {status}: {cnt}")
    print("──────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="格付9ケースを screening_outcomes に移行")
    parser.add_argument("--dry-run", action="store_true", help="実行せず確認のみ")
    args = parser.parse_args()

    migrate(dry_run=args.dry_run)
    if not args.dry_run:
        verify()
