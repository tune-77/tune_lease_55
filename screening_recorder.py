"""screening_recorder.py — 審査結果を screening_records テーブルに記録するサイドカー。

既存スコアリングロジックには一切影響しない。
例外を外部に伝播させない設計（失敗時は success=False を返す）。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

VALID_OUTCOMES = {"contracted", "lost", "delinquent", "completed"}
PII_KEYS = {"name", "address", "phone", "email"}

_DDL = """
CREATE TABLE IF NOT EXISTS screening_records (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id                   TEXT    NOT NULL,
    screened_at               TEXT    NOT NULL,
    total_score               REAL    NOT NULL,
    asset_score               REAL    NOT NULL,
    tenant_score              REAL,
    q_risk_score              REAL,
    competitor_pressure_score REAL,
    outcome                   TEXT,
    input_snapshot            TEXT,
    source                    TEXT    NOT NULL,
    created_at                TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at                TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_screening_records_case_id ON screening_records(case_id);
CREATE INDEX IF NOT EXISTS idx_screening_records_screened_at ON screening_records(screened_at);
CREATE INDEX IF NOT EXISTS idx_screening_records_outcome ON screening_records(outcome);
"""


class ScreeningRecordResult(dict):
    """record_id, success, error を持つ結果辞書。"""


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)


def _redact_pii(snapshot: dict) -> dict:
    """PII キーの値を [REDACTED] に置換した新しい dict を返す。"""
    return {
        k: ("[REDACTED]" if k in PII_KEYS else v)
        for k, v in snapshot.items()
    }


def record_screening_result(
    case_id: str,
    screened_at: str,
    total_score: float,
    asset_score: float,
    source: str,
    tenant_score: Optional[float] = None,
    q_risk_score: Optional[float] = None,
    competitor_pressure_score: Optional[float] = None,
    outcome: Optional[str] = None,
    input_snapshot: Optional[dict] = None,
    db_path: str = "data/lease_data.db",
) -> ScreeningRecordResult:
    """審査結果を screening_records テーブルに記録する。"""
    # BR-402: 必須フィールドのバリデーション
    for field, val in [
        ("case_id", case_id),
        ("screened_at", screened_at),
        ("source", source),
    ]:
        if not val:
            return ScreeningRecordResult(
                record_id=-1,
                success=False,
                error=f"missing required field: {field}",
            )
    try:
        total_score = float(total_score)
        asset_score = float(asset_score)
    except (TypeError, ValueError):
        return ScreeningRecordResult(
            record_id=-1,
            success=False,
            error="missing required field: total_score or asset_score must be numeric",
        )

    # BR-403: スコア範囲チェック
    if not (0.0 <= total_score <= 100.0):
        return ScreeningRecordResult(
            record_id=-1,
            success=False,
            error=f"score out of range: total_score={total_score}",
        )
    if not (0.0 <= asset_score <= 100.0):
        return ScreeningRecordResult(
            record_id=-1,
            success=False,
            error=f"score out of range: asset_score={asset_score}",
        )

    # BR-404: outcome の許容値チェック
    if outcome is not None and outcome not in VALID_OUTCOMES:
        return ScreeningRecordResult(
            record_id=-1,
            success=False,
            error=f"invalid outcome: {outcome}",
        )

    # BR-405: input_snapshot の PII 除去
    snapshot_json: Optional[str] = None
    if input_snapshot is not None:
        try:
            cleaned = _redact_pii(input_snapshot)
            snapshot_json = json.dumps(cleaned, ensure_ascii=False)
        except Exception as e:
            logger.warning("[screening_recorder] JSON serialize failed: %s", e)
            snapshot_json = None

    try:
        # BR-401: DB・ディレクトリ自動作成
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with sqlite3.connect(db_path) as conn:
            _ensure_table(conn)
            cur = conn.execute(
                """
                INSERT INTO screening_records
                    (case_id, screened_at, total_score, asset_score,
                     tenant_score, q_risk_score, competitor_pressure_score,
                     outcome, input_snapshot, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    screened_at,
                    total_score,
                    asset_score,
                    tenant_score,
                    q_risk_score,
                    competitor_pressure_score,
                    outcome,
                    snapshot_json,
                    source,
                ),
            )
            record_id = cur.lastrowid

        logger.info(
            "[screening_recorder] case_id=%s record_id=%s", case_id, record_id
        )
        return ScreeningRecordResult(record_id=record_id, success=True, error=None)

    except Exception as e:
        logger.error("[screening_recorder] ERROR: %s", e)
        return ScreeningRecordResult(record_id=-1, success=False, error=str(e))


def update_screening_outcome(
    case_id: str,
    outcome: str,
    db_path: str = "data/lease_data.db",
) -> ScreeningRecordResult:
    """case_id に紐づく最新レコードの outcome を更新する。"""
    if outcome not in VALID_OUTCOMES:
        return ScreeningRecordResult(
            record_id=-1,
            success=False,
            error=f"invalid outcome: {outcome}",
        )
    try:
        with sqlite3.connect(db_path) as conn:
            _ensure_table(conn)
            conn.execute(
                """
                UPDATE screening_records
                   SET outcome    = ?,
                       updated_at = datetime('now')
                 WHERE id = (
                     SELECT id FROM screening_records
                      WHERE case_id = ?
                      ORDER BY id DESC
                      LIMIT 1
                 )
                """,
                (outcome, case_id),
            )
            # 更新されたレコードの id を取得
            row = conn.execute(
                "SELECT id FROM screening_records WHERE case_id=? ORDER BY id DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            record_id = row[0] if row else -1

        return ScreeningRecordResult(record_id=record_id, success=True, error=None)

    except Exception as e:
        logger.error("[screening_recorder] ERROR in update: %s", e)
        return ScreeningRecordResult(record_id=-1, success=False, error=str(e))
