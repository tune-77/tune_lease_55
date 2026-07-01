#!/usr/bin/env python3
"""Promote approved Cloud Run return data from quarantine DB to the local main DB.

Default mode is dry-run. Use --apply only after reviewing /cloudrun-return-review.
This script intentionally does not infer full case records from demo data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RETURN_DB = PROJECT_ROOT / "data" / "cloudrun_experience_return.db"
DEFAULT_MAIN_DB = PROJECT_ROOT / "data" / "lease_data.db"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "backups"
SUPPORTED_KINDS = ("shion_review", "score_input", "ocr_result")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return bool(row)


def _ensure_quarantine_review_schema(conn: sqlite3.Connection) -> None:
    tables = ("shion_screening_reviews", "cloudrun_score_inputs", "cloudrun_ocr_results")
    for table in tables:
        if not _table_exists(conn, table):
            continue
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        additions = {
            "return_review_status": "TEXT DEFAULT 'candidate'",
            "return_review_note": "TEXT DEFAULT ''",
            "return_reviewed_at": "TEXT DEFAULT ''",
            "return_promoted_at": "TEXT DEFAULT ''",
            "return_promotion_id": "INTEGER",
        }
        for col, ddl in additions.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
    conn.commit()


def _ensure_main_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shion_screening_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            company_name TEXT,
            industry_major TEXT,
            industry_sub TEXT,
            sales_dept TEXT,
            score REAL,
            hantei TEXT,
            q_risk REAL,
            umap_anomaly_score REAL,
            memory_refs INTEGER DEFAULT 0,
            knowledge_refs INTEGER DEFAULT 0,
            identity_used INTEGER DEFAULT 0,
            review_text TEXT NOT NULL,
            prompt_text TEXT,
            form_snapshot TEXT,
            result_snapshot TEXT,
            user_feedback TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cloudrun_return_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_kind TEXT NOT NULL,
            source_local_id INTEGER NOT NULL,
            source_event_id TEXT DEFAULT '',
            target_table TEXT NOT NULL,
            target_id TEXT DEFAULT '',
            promoted_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            note TEXT DEFAULT '',
            UNIQUE(source_kind, source_local_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cloudrun_return_promotions_kind "
        "ON cloudrun_return_promotions(source_kind, source_local_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_shion_screening_reviews_created "
        "ON shion_screening_reviews(created_at)"
    )
    conn.commit()


def _backup_main_db(main_db: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{main_db.stem}_before_cloudrun_return_{stamp}{main_db.suffix}"
    shutil.copy2(main_db, backup_path)
    return backup_path


def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _promotion_exists(main: sqlite3.Connection, kind: str, source_id: int) -> bool:
    row = main.execute(
        """
        SELECT id FROM cloudrun_return_promotions
         WHERE source_kind = ? AND source_local_id = ?
        """,
        (kind, source_id),
    ).fetchone()
    return bool(row)


def _approved_rows(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    if not _table_exists(conn, table):
        return []
    return conn.execute(
        f"""
        SELECT *
          FROM {table}
         WHERE COALESCE(NULLIF(return_review_status, ''), 'candidate') = 'approved'
           AND COALESCE(return_promoted_at, '') = ''
         ORDER BY id ASC
        """
    ).fetchall()


def _insert_promotion_log(
    main: sqlite3.Connection,
    *,
    kind: str,
    source_id: int,
    source_event_id: str,
    target_table: str,
    target_id: str,
    payload: dict[str, Any],
    note: str,
) -> int:
    cur = main.execute(
        """
        INSERT OR IGNORE INTO cloudrun_return_promotions
            (source_kind, source_local_id, source_event_id, target_table, target_id, promoted_at, payload_json, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            kind,
            source_id,
            source_event_id,
            target_table,
            target_id,
            datetime.now(timezone.utc).isoformat(),
            _json_payload(payload),
            note,
        ),
    )
    if cur.lastrowid:
        return int(cur.lastrowid)
    row = main.execute(
        "SELECT id FROM cloudrun_return_promotions WHERE source_kind = ? AND source_local_id = ?",
        (kind, source_id),
    ).fetchone()
    return int(row["id"]) if row else 0


def _mark_promoted(
    quarantine: sqlite3.Connection,
    *,
    table: str,
    source_id: int,
    promotion_id: int,
) -> None:
    quarantine.execute(
        f"""
        UPDATE {table}
           SET return_promoted_at = ?,
               return_promotion_id = ?
         WHERE id = ?
        """,
        (datetime.now(timezone.utc).isoformat(), promotion_id, source_id),
    )


def _promote_shion_review(
    quarantine: sqlite3.Connection,
    main: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    apply: bool,
) -> dict[str, Any]:
    source_id = int(row["id"])
    if _promotion_exists(main, "shion_review", source_id):
        return {"kind": "shion_review", "id": source_id, "action": "skipped_existing"}
    payload = _row_payload(row)
    if not apply:
        return {"kind": "shion_review", "id": source_id, "action": "would_insert"}

    values = (
        row["case_id"],
        row["company_name"],
        row["industry_major"],
        row["industry_sub"],
        row["sales_dept"],
        row["score"],
        row["hantei"],
        row["q_risk"],
        row["umap_anomaly_score"],
        row["memory_refs"] or 0,
        row["knowledge_refs"] or 0,
        row["identity_used"] or 0,
        row["review_text"],
        row["prompt_text"] or "",
        row["form_snapshot"] or "{}",
        row["result_snapshot"] or "{}",
        row["user_feedback"] or "",
    )
    cur = main.execute(
        """
        INSERT INTO shion_screening_reviews (
            case_id, company_name, industry_major, industry_sub, sales_dept, score, hantei,
            q_risk, umap_anomaly_score, memory_refs, knowledge_refs, identity_used,
            review_text, prompt_text, form_snapshot, result_snapshot, user_feedback
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    target_id = str(cur.lastrowid)
    promotion_id = _insert_promotion_log(
        main,
        kind="shion_review",
        source_id=source_id,
        source_event_id=str(row["cloud_event_id"] or row["cloud_review_id"] or ""),
        target_table="shion_screening_reviews",
        target_id=target_id,
        payload=payload,
        note=str(row["return_review_note"] or ""),
    )
    _mark_promoted(
        quarantine,
        table="shion_screening_reviews",
        source_id=source_id,
        promotion_id=promotion_id,
    )
    return {"kind": "shion_review", "id": source_id, "action": "inserted", "target_id": target_id}


def _promote_as_log_only(
    quarantine: sqlite3.Connection,
    main: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    kind: str,
    table: str,
    apply: bool,
) -> dict[str, Any]:
    source_id = int(row["id"])
    if _promotion_exists(main, kind, source_id):
        return {"kind": kind, "id": source_id, "action": "skipped_existing"}
    if not apply:
        return {"kind": kind, "id": source_id, "action": "would_log_only"}
    payload = _row_payload(row)
    promotion_id = _insert_promotion_log(
        main,
        kind=kind,
        source_id=source_id,
        source_event_id=str(row["event_id"] or ""),
        target_table="cloudrun_return_promotions",
        target_id="",
        payload=payload,
        note=str(row["return_review_note"] or "log-only promotion; manual mapping required"),
    )
    _mark_promoted(quarantine, table=table, source_id=source_id, promotion_id=promotion_id)
    return {"kind": kind, "id": source_id, "action": "logged_only"}


def promote_approved_return_data(
    *,
    return_db: Path = DEFAULT_RETURN_DB,
    main_db: Path = DEFAULT_MAIN_DB,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    apply: bool = False,
    backup: bool = True,
    kinds: tuple[str, ...] = SUPPORTED_KINDS,
) -> dict[str, Any]:
    if not return_db.exists():
        raise FileNotFoundError(f"return db not found: {return_db}")
    if apply and not main_db.exists():
        raise FileNotFoundError(f"main db not found: {main_db}")
    unknown = sorted(set(kinds) - set(SUPPORTED_KINDS))
    if unknown:
        raise ValueError(f"unsupported kind: {', '.join(unknown)}")

    backup_path = ""
    if apply and backup:
        backup_path = str(_backup_main_db(main_db, backup_dir))

    results: list[dict[str, Any]] = []
    with _connect(return_db) as quarantine, _connect(main_db) as main:
        _ensure_quarantine_review_schema(quarantine)
        _ensure_main_schema(main)

        if "shion_review" in kinds:
            for row in _approved_rows(quarantine, "shion_screening_reviews"):
                results.append(_promote_shion_review(quarantine, main, row, apply=apply))
        if "score_input" in kinds:
            for row in _approved_rows(quarantine, "cloudrun_score_inputs"):
                results.append(
                    _promote_as_log_only(
                        quarantine,
                        main,
                        row,
                        kind="score_input",
                        table="cloudrun_score_inputs",
                        apply=apply,
                    )
                )
        if "ocr_result" in kinds:
            for row in _approved_rows(quarantine, "cloudrun_ocr_results"):
                results.append(
                    _promote_as_log_only(
                        quarantine,
                        main,
                        row,
                        kind="ocr_result",
                        table="cloudrun_ocr_results",
                        apply=apply,
                    )
                )

        if apply:
            main.commit()
            quarantine.commit()
        else:
            main.rollback()
            quarantine.rollback()

    summary: dict[str, int] = {}
    for item in results:
        key = f"{item['kind']}:{item['action']}"
        summary[key] = summary.get(key, 0) + 1
    return {
        "apply": apply,
        "return_db": str(return_db),
        "main_db": str(main_db),
        "backup_path": backup_path,
        "summary": summary,
        "items": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote approved Cloud Run return data from quarantine DB to local main DB."
    )
    parser.add_argument("--return-db", type=Path, default=DEFAULT_RETURN_DB)
    parser.add_argument("--main-db", type=Path, default=DEFAULT_MAIN_DB)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--kind", choices=SUPPORTED_KINDS + ("all",), default="all")
    parser.add_argument("--apply", action="store_true", help="Actually write to the main DB. Default is dry-run.")
    parser.add_argument("--no-backup", action="store_true", help="Skip main DB backup when --apply is used.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    kinds = SUPPORTED_KINDS if args.kind == "all" else (args.kind,)
    result = promote_approved_return_data(
        return_db=args.return_db,
        main_db=args.main_db,
        backup_dir=args.backup_dir,
        apply=args.apply,
        backup=not args.no_backup,
        kinds=kinds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.apply:
        print("\nDRY-RUN only. Add --apply after reviewing the summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
