#!/usr/bin/env python3
"""Download Cloud Run input event logs from GCS for local nightly processing."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Any


GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_INPUT_PREFIX = os.environ.get("GCS_INPUT_PREFIX", "cloudrun-inputs/")
LOCAL_INPUT_DIR = Path(os.environ.get("LOCAL_CLOUDRUN_INPUT_DIR", "data/cloudrun_inputs"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_LEASE_DB = PROJECT_ROOT / "data" / "lease_data.db"
DEFAULT_RETURN_DB = PROJECT_ROOT / "data" / "cloudrun_experience_return.db"
LOCAL_LEASE_DB = Path(os.environ.get("LOCAL_LEASE_DB_PATH", DEFAULT_RETURN_DB))
CLOUDRUN_EVENT_ARCHIVE_LOG = PROJECT_ROOT / "data" / "cloudrun_experience_events.jsonl"
WIZARD_INPUT_LOG = PROJECT_ROOT / "data" / "wizard_input_log.jsonl"
RAG_FEEDBACK_LOG = PROJECT_ROOT / "data" / "rag_feedback_log.jsonl"
RAG_HIT_LOG = PROJECT_ROOT / "data" / "rag_hit_log.jsonl"
SCREENING_LOOP_FEEDBACK_LOG = PROJECT_ROOT / "data" / "screening_loop_feedback.jsonl"
CLOUDRUN_IMPROVEMENT_LOG = PROJECT_ROOT / "data" / "cloudrun_improvement_log.jsonl"
CLOUDRUN_CHAT_LOG = PROJECT_ROOT / "data" / "cloudrun_chat_log.jsonl"
SHION_MEMORY_USAGE_LOG = PROJECT_ROOT / "data" / "shion_memory_usage_log.jsonl"
SHION_HYPOTHESIS_COLLISION_LOG = PROJECT_ROOT / "data" / "shion_hypothesis_collision_log.jsonl"
USER_PERSONAL_MEMORY_PATH = PROJECT_ROOT / "data" / "user_personal_memory.md"
JUDGMENT_ASSET_EVENT_TYPES = {
    "human_response_feedback",
    "screening_loop_feedback",
    "shion_screening_review_feedback",
    "judgment_feedback_created",
    "lease_news_judgment_change",
}
WIZARD_TRACKED_FIELDS = [
    "company_name",
    "nenshu",
    "op_profit",
    "acquisition_cost",
    "asset_name",
    "passion_text",
    "industry_detail",
    "asset_detail",
    "asset_purpose",
    "asset_location",
]


def _bucket_name() -> str:
    value = (GCS_BUCKET or "").strip()
    if value.startswith("gs://"):
        value = value[5:]
    return value.split("/", 1)[0]


def _date_range(days: int) -> Iterable[date]:
    today = datetime.now(timezone.utc).date()
    count = max(1, int(days))
    for offset in range(count):
        yield today - timedelta(days=offset)


def _event_blob_name(day: date) -> str:
    prefix = GCS_INPUT_PREFIX.strip("/") or "cloudrun-inputs"
    return f"{prefix}/{day.isoformat()}/events.jsonl"


def _download_text_with_gcloud(bucket_name: str, blob_name: str) -> str:
    uri = f"gs://{bucket_name}/{blob_name}"
    proc = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout


# その日の events.jsonl がまだ無いのは正常（イベントゼロ）。認証切れ・gcloud不在などの
# 実エラーと区別し、実エラーは終了コードで検知できるようにする
NOT_FOUND_REASON = "not_found"
_NOT_FOUND_MARKERS = (
    "404",
    "no such object",
    "does not exist",
    "matched no objects",
    "no urls matched",
    "notfound",
)


def _is_not_found_reason(reason: str) -> bool:
    lowered = (reason or "").lower()
    return any(marker in lowered for marker in _NOT_FOUND_MARKERS)


def _download_event_text(bucket: Any | None, bucket_name: str, blob_name: str) -> tuple[str | None, str]:
    if bucket is not None:
        try:
            return bucket.blob(blob_name).download_as_text(), "storage-client"
        except Exception as exc:
            if _is_not_found_reason(str(exc)):
                return None, NOT_FOUND_REASON
    try:
        return _download_text_with_gcloud(bucket_name, blob_name), "gcloud"
    except FileNotFoundError:
        return None, "error: gcloud コマンドが見つかりません（PATH未設定の可能性）"
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if _is_not_found_reason(stderr):
            return None, NOT_FOUND_REASON
        return None, f"error: {stderr[:200] or exc}"
    except Exception as exc:
        return None, f"error: {exc}"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _append_jsonl_dedup(path: Path, rows: list[dict]) -> int:
    existing = _load_jsonl(path)
    merged = _merge_events(existing, rows)
    _write_jsonl(path, merged)
    return max(0, len(merged) - len(existing))


def _merge_events(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for row in existing + incoming:
        event_id = str(row.get("event_id") or "")
        key = event_id or json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    merged.sort(key=lambda row: str(row.get("ts") or ""))
    return merged


def _wizard_entry_from_event(event: dict) -> dict | None:
    if event.get("event_type") not in {"score_calculated", "score_full_calculated"}:
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    empty = [field for field in WIZARD_TRACKED_FIELDS if not inputs.get(field)]
    return {
        "event_id": event.get("event_id"),
        "ts": event.get("ts"),
        "total_fields": len(WIZARD_TRACKED_FIELDS),
        "empty_count": len(empty),
        "empty_fields": empty,
        "surface": f"cloudrun_{event.get('event_type')}",
        "source": "cloudrun_input_writeback",
    }


def _rag_entries_from_event(event: dict) -> tuple[dict | None, dict | None]:
    if event.get("event_type") != "rag_feedback":
        return None, None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if not payload:
        return None, None
    feedback = {
        **payload,
        "event_id": event.get("event_id"),
        "source": "cloudrun_input_writeback",
    }
    hit = {
        "event_id": event.get("event_id"),
        "ts": payload.get("ts") or event.get("ts"),
        "doc_id": payload.get("doc_id"),
        "obsidian_ref": payload.get("obsidian_ref"),
        "rating": payload.get("rating"),
        "surface": payload.get("surface") or event.get("surface"),
        "hit_type": "feedback_confirmed",
        "source": "cloudrun_input_writeback",
    }
    return feedback, hit


def _open_local_db(path: Path | None = None) -> sqlite3.Connection:
    target = path or LOCAL_LEASE_DB
    if target.resolve() == MAIN_LEASE_DB.resolve() and os.environ.get("CLOUDRUN_SYNC_ALLOW_MAIN_DB", "").strip() != "1":
        raise RuntimeError(
            "Refusing to write Cloud Run return events directly into data/lease_data.db. "
            "Set CLOUDRUN_SYNC_ALLOW_MAIN_DB=1 only after backup/review, or use LOCAL_LEASE_DB_PATH."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_local_sync_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cloudrun_score_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            event_type TEXT,
            case_id TEXT,
            surface TEXT,
            score REAL,
            hantei TEXT,
            industry_major TEXT,
            industry_sub TEXT,
            inputs_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cloudrun_ocr_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            doc_type TEXT,
            content_type TEXT,
            result_json TEXT NOT NULL,
            confidence REAL,
            detected_fields TEXT,
            missing_fields TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
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
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(shion_screening_reviews)").fetchall()}
    for col, ddl in {
        "cloud_review_id": "TEXT",
        "cloud_event_id": "TEXT",
    }.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE shion_screening_reviews ADD COLUMN {col} {ddl}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cloudrun_sync_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT,
            local_table TEXT,
            local_id INTEGER,
            cloud_id TEXT,
            synced_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cloudrun_judgment_asset_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            event_type TEXT NOT NULL,
            surface TEXT DEFAULT '',
            asset_type TEXT DEFAULT '',
            title TEXT DEFAULT '',
            signal TEXT DEFAULT '',
            case_id TEXT DEFAULT '',
            score REAL,
            q_risk REAL,
            summary_text TEXT DEFAULT '',
            lesson_text TEXT DEFAULT '',
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shion_screening_reviews_cloud_review_id ON shion_screening_reviews(cloud_review_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shion_screening_reviews_cloud_event_id ON shion_screening_reviews(cloud_event_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cloudrun_score_inputs_created ON cloudrun_score_inputs(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cloudrun_ocr_results_created ON cloudrun_ocr_results(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cloudrun_judgment_asset_event ON cloudrun_judgment_asset_candidates(event_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cloudrun_judgment_asset_created ON cloudrun_judgment_asset_candidates(created_at)")


def _sync_event_seen(conn: sqlite3.Connection, event_id: str) -> bool:
    if not event_id:
        return False
    row = conn.execute("SELECT event_id FROM cloudrun_sync_events WHERE event_id = ?", (event_id,)).fetchone()
    return bool(row)


def _record_sync_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    event_type: str,
    local_table: str,
    local_id: int | None = None,
    cloud_id: str = "",
) -> None:
    if not event_id:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO cloudrun_sync_events
            (event_id, event_type, local_table, local_id, cloud_id, synced_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (event_id, event_type, local_table, local_id, cloud_id, datetime.now(timezone.utc).isoformat()),
    )


def _json_dumps(value: object) -> str:
    return json.dumps(value if isinstance(value, (dict, list)) else {}, ensure_ascii=False, sort_keys=True)


def _insert_shion_review_from_event(conn: sqlite3.Connection, event: dict) -> int:
    if event.get("event_type") != "shion_screening_review":
        return 0
    event_id = str(event.get("event_id") or "")
    if _sync_event_seen(conn, event_id):
        return 0
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    review_text = str(payload.get("review_text") or "").strip()
    if not review_text:
        _record_sync_event(conn, event_id=event_id, event_type="shion_screening_review", local_table="skipped")
        return 0
    cloud_review_id = str(payload.get("cloud_review_id") or payload.get("id") or "")
    existing = None
    if cloud_review_id:
        existing = conn.execute(
            "SELECT id FROM shion_screening_reviews WHERE cloud_review_id = ?",
            (cloud_review_id,),
        ).fetchone()
    if existing:
        local_id = int(existing["id"])
        _record_sync_event(
            conn,
            event_id=event_id,
            event_type="shion_screening_review",
            local_table="shion_screening_reviews",
            local_id=local_id,
            cloud_id=cloud_review_id,
        )
        return 0
    cur = conn.execute(
        """
        INSERT INTO shion_screening_reviews (
            case_id, company_name, industry_major, industry_sub, sales_dept,
            score, hantei, q_risk, umap_anomaly_score,
            memory_refs, knowledge_refs, identity_used, review_text,
            prompt_text, form_snapshot, result_snapshot, user_feedback,
            cloud_review_id, cloud_event_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(payload.get("case_id") or ""),
            str(payload.get("company_name") or ""),
            str(payload.get("industry_major") or ""),
            str(payload.get("industry_sub") or ""),
            str(payload.get("sales_dept") or ""),
            payload.get("score"),
            str(payload.get("hantei") or ""),
            payload.get("q_risk"),
            payload.get("umap_anomaly_score"),
            int(payload.get("memory_refs") or 0),
            int(payload.get("knowledge_refs") or 0),
            1 if payload.get("identity_used") else 0,
            review_text[:8000],
            str(payload.get("prompt_text") or "")[:8000],
            _json_dumps(payload.get("form_snapshot")),
            _json_dumps(payload.get("result_snapshot")),
            str(payload.get("user_feedback") or ""),
            cloud_review_id,
            event_id,
            str(event.get("ts") or payload.get("created_at") or datetime.now(timezone.utc).isoformat()),
        ),
    )
    local_id = int(cur.lastrowid)
    _record_sync_event(
        conn,
        event_id=event_id,
        event_type="shion_screening_review",
        local_table="shion_screening_reviews",
        local_id=local_id,
        cloud_id=cloud_review_id,
    )
    return 1


def _insert_score_input_from_event(conn: sqlite3.Connection, event: dict) -> int:
    if event.get("event_type") not in {"score_calculated", "score_full_calculated"}:
        return 0
    event_id = str(event.get("event_id") or "")
    if _sync_event_seen(conn, event_id):
        return 0
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if not inputs and not result:
        _record_sync_event(conn, event_id=event_id, event_type=str(event.get("event_type") or ""), local_table="skipped")
        return 0
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO cloudrun_score_inputs (
            event_id, event_type, case_id, surface, score, hantei,
            industry_major, industry_sub, inputs_json, result_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            str(event.get("event_type") or ""),
            str(payload.get("case_id") or ""),
            str(event.get("surface") or ""),
            result.get("score_base", result.get("score")),
            str(result.get("hantei") or ""),
            str(result.get("industry_major") or inputs.get("industry_major") or ""),
            str(result.get("industry_sub") or inputs.get("industry_sub") or ""),
            json.dumps(inputs, ensure_ascii=False, sort_keys=True),
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            str(event.get("ts") or datetime.now(timezone.utc).isoformat()),
        ),
    )
    if cur.rowcount:
        local_id = int(cur.lastrowid)
        _record_sync_event(conn, event_id=event_id, event_type=str(event.get("event_type") or ""), local_table="cloudrun_score_inputs", local_id=local_id)
        return 1
    return 0


def _insert_ocr_result_from_event(conn: sqlite3.Connection, event: dict) -> int:
    if event.get("event_type") != "ocr_extracted":
        return 0
    event_id = str(event.get("event_id") or "")
    if _sync_event_seen(conn, event_id):
        return 0
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if not result:
        _record_sync_event(conn, event_id=event_id, event_type="ocr_extracted", local_table="skipped")
        return 0
    detected = result.get("detected_fields") if isinstance(result.get("detected_fields"), list) else []
    missing = result.get("missing_fields") if isinstance(result.get("missing_fields"), list) else []
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO cloudrun_ocr_results (
            event_id, doc_type, content_type, result_json, confidence,
            detected_fields, missing_fields, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            str(payload.get("doc_type") or "financial"),
            str(payload.get("content_type") or ""),
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            result.get("confidence"),
            json.dumps(detected, ensure_ascii=False),
            json.dumps(missing, ensure_ascii=False),
            str(event.get("ts") or datetime.now(timezone.utc).isoformat()),
        ),
    )
    if cur.rowcount:
        local_id = int(cur.lastrowid)
        _record_sync_event(conn, event_id=event_id, event_type="ocr_extracted", local_table="cloudrun_ocr_results", local_id=local_id)
        return 1
    return 0


def _apply_shion_review_feedback_from_event(conn: sqlite3.Connection, event: dict) -> int:
    if event.get("event_type") != "shion_screening_review_feedback":
        return 0
    event_id = str(event.get("event_id") or "")
    if _sync_event_seen(conn, event_id):
        return 0
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    feedback = str(payload.get("user_feedback") or "").strip()
    if feedback not in {"useful", "needs_fix", "wrong"}:
        _record_sync_event(conn, event_id=event_id, event_type="shion_screening_review_feedback", local_table="skipped")
        return 0
    cloud_review_id = str(payload.get("cloud_review_id") or payload.get("id") or "")
    row = None
    if cloud_review_id:
        row = conn.execute(
            "SELECT id FROM shion_screening_reviews WHERE cloud_review_id = ?",
            (cloud_review_id,),
        ).fetchone()
    if not row:
        return 0
    local_id = int(row["id"])
    conn.execute(
        "UPDATE shion_screening_reviews SET user_feedback = ? WHERE id = ?",
        (feedback, local_id),
    )
    _record_sync_event(
        conn,
        event_id=event_id,
        event_type="shion_screening_review_feedback",
        local_table="shion_screening_reviews",
        local_id=local_id,
        cloud_id=cloud_review_id,
    )
    return 1


def _screening_loop_feedback_from_event(event: dict) -> dict | None:
    if event.get("event_type") != "screening_loop_feedback":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if not payload:
        return None
    return {
        **payload,
        "event_id": event.get("event_id"),
        "source": "cloudrun_input_writeback",
    }


def _first_text(*values: Any, limit: int = 900) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text[:limit]
    return ""


def _judgment_asset_candidate_from_event(event: dict) -> dict | None:
    event_type = str(event.get("event_type") or "")
    if event_type not in JUDGMENT_ASSET_EVENT_TYPES:
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if not payload:
        return None

    surface = str(event.get("surface") or payload.get("surface") or "")
    created_at = str(event.get("ts") or payload.get("ts") or datetime.now(timezone.utc).isoformat())
    event_id = str(event.get("event_id") or "")
    case_id = str(payload.get("case_id") or "")
    score = payload.get("score")
    q_risk = payload.get("q_risk") or payload.get("quantum_risk")
    asset_type = "judgment_reaction"
    signal = ""
    title = f"判断資産候補 / {event_type}"
    summary = ""
    lesson = ""

    if event_type == "human_response_feedback":
        rating = str(payload.get("rating") or "")
        route = str(payload.get("route") or surface or "chat")
        asset_type = "response_feedback"
        signal = rating
        title = f"応答feedback / {route} / {rating}".strip(" /")
        summary = _first_text(payload.get("comment"), payload.get("message_preview"), payload.get("response_start"))
        lesson = "人間の良い/悪い反応を、次回応答の冒頭・根拠・具体性の調整材料として残す。"
    elif event_type == "screening_loop_feedback":
        target = str(payload.get("target") or "")
        rating = str(payload.get("rating") or "")
        asset_type = "screening_loop_feedback"
        signal = rating
        title = f"審査ループfeedback / {target} / {rating}".strip(" /")
        summary = _first_text(payload.get("comment"), payload.get("issue_text"), payload.get("ringi_policy_text"))
        lesson = "人間が反応した争点・稟議方針を、次回類似案件の確認観点として残す。"
    elif event_type == "shion_screening_review_feedback":
        feedback = str(payload.get("user_feedback") or "")
        asset_type = "shion_review_feedback"
        signal = feedback
        title = f"紫苑レビュー評価 / {feedback}".strip(" /")
        summary = _first_text(payload.get("review_text"), payload.get("comment"), payload.get("cloud_review_id"), payload.get("id"))
        lesson = "紫苑レビューへの評価を、次回レビューの粒度・観点・表現の調整材料として残す。"
    elif event_type == "judgment_feedback_created":
        model_decision = str(payload.get("model_decision") or "")
        human_decision = str(payload.get("human_decision") or "")
        asset_type = "judgment_difference"
        signal = "decision_changed" if model_decision and human_decision and model_decision != human_decision else "judgment_feedback"
        title = f"判断差分 / {model_decision} -> {human_decision}".strip(" /")
        summary = _first_text(payload.get("reason"), human_decision, model_decision)
        lesson = "モデル判断と人間判断の差分を、レビュー後に判断ルール候補として再利用する。"
    elif event_type == "lease_news_judgment_change":
        final_decision = str(payload.get("final_decision") or "")
        asset_type = "news_judgment_change"
        signal = "news_changed_judgment"
        title = f"ニュース参照後の判断変更 / {final_decision}".strip(" /")
        summary = _first_text(payload.get("reason"), payload.get("news_focus_summary"), payload.get("news_focus_tag_summary"))
        lesson = "外部ニュースで判断が変わった理由を、次回の同種論点確認に使う。"

    if not summary:
        summary = _first_text(title, signal)

    return {
        "event_id": event_id,
        "event_type": event_type,
        "surface": surface,
        "asset_type": asset_type,
        "title": title[:180],
        "signal": signal[:120],
        "case_id": case_id[:160],
        "score": score,
        "q_risk": q_risk,
        "summary_text": summary[:1200],
        "lesson_text": lesson[:1200],
        "evidence_json": json.dumps({"event": event, "payload": payload}, ensure_ascii=False, sort_keys=True),
        "created_at": created_at,
    }


def _insert_judgment_asset_candidate_from_event(conn: sqlite3.Connection, event: dict) -> int:
    candidate = _judgment_asset_candidate_from_event(event)
    if not candidate:
        return 0
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO cloudrun_judgment_asset_candidates (
            event_id, event_type, surface, asset_type, title, signal,
            case_id, score, q_risk, summary_text, lesson_text, evidence_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate["event_id"],
            candidate["event_type"],
            candidate["surface"],
            candidate["asset_type"],
            candidate["title"],
            candidate["signal"],
            candidate["case_id"],
            candidate["score"],
            candidate["q_risk"],
            candidate["summary_text"],
            candidate["lesson_text"],
            candidate["evidence_json"],
            candidate["created_at"],
        ),
    )
    return 1 if cur.rowcount else 0


def _improvement_entry_from_event(event: dict) -> dict | None:
    if event.get("event_type") != "improvement_note":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    title = str(payload.get("title") or "Cloud Run改善メモ").strip()
    body = str(payload.get("body") or "").strip()
    if not body:
        return None
    return {
        "event_id": event.get("event_id"),
        "ts": event.get("ts"),
        "title": title[:120],
        "body": body[:12000],
        "surface": event.get("surface") or "chat_improvement",
        "source": "cloudrun_input_writeback",
    }


def _chat_entry_from_event(event: dict) -> dict | None:
    if event.get("event_type") != "chat_exchange":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    user_message = str(payload.get("user_message") or "").strip()
    assistant_reply = str(payload.get("assistant_reply") or "").strip()
    if not user_message and not assistant_reply:
        return None
    return {
        "event_id": event.get("event_id"),
        "ts": event.get("ts"),
        "surface": event.get("surface") or "unknown",
        "user_id": str(payload.get("user_id") or "default")[:80],
        "category": str(payload.get("category") or "")[:80],
        "response_mode": str(payload.get("response_mode") or "")[:40],
        "user_message": user_message[:1200],
        "assistant_reply": assistant_reply[:1800],
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        "shion_hypothesis": payload.get("shion_hypothesis") if isinstance(payload.get("shion_hypothesis"), dict) else {},
        "source": "cloudrun_input_writeback",
    }


def _shion_memory_usage_from_event(event: dict) -> dict | None:
    if event.get("event_type") != "shion_memory_usage":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    refs = payload.get("refs") if isinstance(payload.get("refs"), list) else []
    return {
        "event_id": event.get("event_id"),
        "ts": payload.get("ts") or event.get("ts"),
        "route": str(payload.get("route") or ""),
        "refs": [str(ref) for ref in refs[:20]],
        "ref_count": len(refs),
        "question": str(payload.get("question") or "")[:120],
        "surface": event.get("surface") or "api_chat",
        "source": "cloudrun_input_writeback",
    }


def _build_hypothesis_collision_rows(chat_rows: list[dict]) -> list[dict]:
    if not chat_rows:
        return []
    try:
        from api.shion_hypothesis_collision import collision_entries_from_chat_rows

        return collision_entries_from_chat_rows(chat_rows)
    except Exception as exc:
        print(f"[ShionHypothesisCollision] materialize skipped: {exc}")
        return []


def _personal_memory_lines_from_event(event: dict) -> tuple[list[str], str]:
    """Return local personal-memory lines and confirmed dog name from an event."""
    event_type = str(event.get("event_type") or "")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    message = ""
    lines: list[str] = []
    dog_name = ""

    if event_type == "personal_memory":
        message = str(payload.get("message") or "").strip()
        dog_name = str(payload.get("dog_name") or "").strip()
        lines = [str(line).strip() for line in payload.get("derived_lines") or [] if str(line).strip()]
    elif event_type == "chat_exchange":
        message = str(payload.get("user_message") or "").strip()
    else:
        return [], ""

    try:
        from api.user_personal_memory import derive_personal_memory_entries

        derived = derive_personal_memory_entries(
            message,
            source=f"cloudrun:{event.get('surface') or 'unknown'}",
            timestamp=str(event.get("ts") or ""),
        )
    except Exception:
        derived = {"captured": False, "lines": [], "dog_name": ""}

    if not dog_name:
        dog_name = str(derived.get("dog_name") or "").strip()
    if not lines:
        lines = [str(line).strip() for line in derived.get("lines") or [] if str(line).strip()]
    if dog_name and not any("Dog name:" in line for line in lines):
        lines.insert(0, f"- [confirmed] Dog name: {dog_name}")
    return lines, dog_name


def _sync_personal_memory_from_events(events: list[dict], path: Path | None = None) -> int:
    if not events:
        return 0
    target_path = path or USER_PERSONAL_MEMORY_PATH

    try:
        from api import user_personal_memory as upm
    except Exception:
        return 0

    extracted: list[tuple[list[str], str]] = []
    for event in sorted(events, key=lambda row: str(row.get("ts") or "")):
        lines, dog_name = _personal_memory_lines_from_event(event)
        if lines or dog_name:
            extracted.append((lines, dog_name))
    if not extracted:
        return 0

    upm._ensure_file(target_path)
    original_text = target_path.read_text(encoding="utf-8", errors="replace")
    file_lines = original_text.splitlines()
    existing = {line.strip() for line in file_lines}
    new_count = 0

    for lines, dog_name in extracted:
        if dog_name:
            before = "\n".join(file_lines)
            file_lines = upm._replace_or_append_dog_name(file_lines, dog_name)
            if "\n".join(file_lines) != before:
                new_count += 1
                existing = {line.strip() for line in file_lines}

        capture_lines = [line for line in lines if line and "Dog name:" not in line]
        if capture_lines and "## Captured Personal Memories" not in file_lines:
            file_lines.extend(["", "## Captured Personal Memories", ""])
            existing = {line.strip() for line in file_lines}
        for line in capture_lines:
            stripped = line.strip()
            if not stripped or stripped in existing:
                continue
            file_lines.append(stripped)
            existing.add(stripped)
            new_count += 1

    next_text = "\n".join(file_lines).rstrip() + "\n"
    if next_text != original_text:
        target_path.write_text(next_text, encoding="utf-8")
    return new_count


def _materialize_local_db(events: list[dict]) -> dict[str, int]:
    with _open_local_db() as conn:
        _ensure_local_sync_schema(conn)
        shion_new = 0
        shion_feedback_updated = 0
        score_inputs_new = 0
        ocr_results_new = 0
        judgment_assets_new = 0
        for event in events:
            score_inputs_new += _insert_score_input_from_event(conn, event)
            ocr_results_new += _insert_ocr_result_from_event(conn, event)
            shion_new += _insert_shion_review_from_event(conn, event)
            judgment_assets_new += _insert_judgment_asset_candidate_from_event(conn, event)
        for event in events:
            shion_feedback_updated += _apply_shion_review_feedback_from_event(conn, event)
        conn.commit()
    return {
        "score_inputs_new": score_inputs_new,
        "ocr_results_new": ocr_results_new,
        "shion_reviews_new": shion_new,
        "shion_review_feedback_updated": shion_feedback_updated,
        "judgment_asset_candidates_new": judgment_assets_new,
    }


def materialize_events(events: list[dict]) -> dict[str, int]:
    all_events_new = _append_jsonl_dedup(CLOUDRUN_EVENT_ARCHIVE_LOG, events) if events else 0
    wizard_rows = [row for event in events if (row := _wizard_entry_from_event(event))]
    screening_loop_rows = [row for event in events if (row := _screening_loop_feedback_from_event(event))]
    improvement_rows = [row for event in events if (row := _improvement_entry_from_event(event))]
    chat_rows = [row for event in events if (row := _chat_entry_from_event(event))]
    hypothesis_collision_rows = _build_hypothesis_collision_rows(chat_rows)
    shion_memory_usage_rows = [row for event in events if (row := _shion_memory_usage_from_event(event))]
    personal_memory_new = _sync_personal_memory_from_events(events) if events else 0
    rag_feedback_rows: list[dict] = []
    rag_hit_rows: list[dict] = []
    for event in events:
        feedback, hit = _rag_entries_from_event(event)
        if feedback:
            rag_feedback_rows.append(feedback)
        if hit:
            rag_hit_rows.append(hit)
    db_result = _materialize_local_db(events) if events else {
        "score_inputs_new": 0,
        "ocr_results_new": 0,
        "shion_reviews_new": 0,
        "shion_review_feedback_updated": 0,
        "judgment_asset_candidates_new": 0,
    }
    return {
        "all_events_new": all_events_new,
        "wizard_new": _append_jsonl_dedup(WIZARD_INPUT_LOG, wizard_rows) if wizard_rows else 0,
        "rag_feedback_new": _append_jsonl_dedup(RAG_FEEDBACK_LOG, rag_feedback_rows) if rag_feedback_rows else 0,
        "rag_hit_new": _append_jsonl_dedup(RAG_HIT_LOG, rag_hit_rows) if rag_hit_rows else 0,
        "screening_loop_feedback_new": _append_jsonl_dedup(SCREENING_LOOP_FEEDBACK_LOG, screening_loop_rows) if screening_loop_rows else 0,
        "improvement_new": _append_jsonl_dedup(CLOUDRUN_IMPROVEMENT_LOG, improvement_rows) if improvement_rows else 0,
        "chat_new": _append_jsonl_dedup(CLOUDRUN_CHAT_LOG, chat_rows) if chat_rows else 0,
        "hypothesis_collision_new": _append_jsonl_dedup(SHION_HYPOTHESIS_COLLISION_LOG, hypothesis_collision_rows) if hypothesis_collision_rows else 0,
        "shion_memory_usage_new": _append_jsonl_dedup(SHION_MEMORY_USAGE_LOG, shion_memory_usage_rows) if shion_memory_usage_rows else 0,
        "personal_memory_new": personal_memory_new,
        **db_result,
    }


def sync_day(bucket: Any | None, day: date, local_dir: Path = LOCAL_INPUT_DIR, bucket_name: str | None = None) -> dict:
    blob_name = _event_blob_name(day)
    resolved_bucket_name = bucket_name or _bucket_name()
    text, source = _download_event_text(bucket, resolved_bucket_name, blob_name)
    if text is None:
        return {"date": day.isoformat(), "downloaded": False, "events": 0, "path": str(local_dir / f"{day.isoformat()}.jsonl"), "reason": source}

    incoming: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            incoming.append(item)

    local_path = local_dir / f"{day.isoformat()}.jsonl"
    existing = _load_jsonl(local_path)
    merged = _merge_events(existing, incoming)
    _write_jsonl(local_path, merged)
    materialized = materialize_events(merged)
    return {
        "date": day.isoformat(),
        "downloaded": True,
        "events": len(merged),
        "new_events": max(0, len(merged) - len(existing)),
        "materialized": materialized,
        "path": str(local_path),
        "source": source,
    }


def main() -> None:
    days = int(os.environ.get("CLOUDRUN_INPUT_SYNC_DAYS", "3"))
    bucket_name = _bucket_name()
    bucket = None
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception:
        bucket = None

    print(f"Cloud Run入力同期開始: gs://{bucket_name}/{GCS_INPUT_PREFIX} → {LOCAL_INPUT_DIR}")
    error_days = 0
    for day in _date_range(days):
        result = sync_day(bucket, day, LOCAL_INPUT_DIR, bucket_name=bucket_name)
        reason = str(result.get("reason") or "")
        is_error = not result["downloaded"] and reason != NOT_FOUND_REASON
        if is_error:
            error_days += 1
        status = "DL" if result["downloaded"] else ("ERR" if is_error else "SKIP")
        detail = result.get("source") or reason
        suffix = f" source={detail}" if detail else ""
        print(f"[{status}] {result['date']} events={result['events']} path={result['path']}{suffix}")
    if error_days:
        print(f"エラー: {error_days} 日分の取得に失敗しました（イベント未取得の可能性）", file=sys.stderr)
        sys.exit(1)
    print("同期完了")


if __name__ == "__main__":
    main()
