"""
tests/spec_phase4/test_P4-001.py — P4-001 Acceptance Criteria テスト (AC-1001〜AC-1010)
"""
from __future__ import annotations

import json
import sqlite3
import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from screening_recorder import record_screening_result, update_screening_outcome

SCREENED_AT = "2026-05-15T00:00:00Z"


def _valid_kwargs(**overrides):
    base = dict(
        case_id="test-case-001",
        screened_at=SCREENED_AT,
        total_score=85.0,
        asset_score=70.0,
        source="streamlit",
    )
    base.update(overrides)
    return base


# AC-1001: 正常 INSERT が成功する
def test_1001_normal_insert_success():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        result = record_screening_result(**_valid_kwargs(db_path=db))
        assert result["success"] is True, result
        assert isinstance(result["record_id"], int)
        assert result["record_id"] > 0
        assert result["error"] is None
    finally:
        os.unlink(db)


# AC-1002: 同一 case_id で複数回 INSERT できる（再審査ユースケース）
def test_1002_duplicate_case_id_allowed():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        record_screening_result(**_valid_kwargs(db_path=db))
        record_screening_result(**_valid_kwargs(db_path=db))
        conn = sqlite3.connect(db)
        count = conn.execute(
            "SELECT COUNT(*) FROM screening_records WHERE case_id='test-case-001'"
        ).fetchone()[0]
        conn.close()
        assert count == 2
    finally:
        os.unlink(db)


# AC-1003: 必須フィールド（case_id）欠損で失敗する
def test_1003_missing_case_id_fails():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        result = record_screening_result(**_valid_kwargs(case_id="", db_path=db))
        assert result["success"] is False
        assert "missing required field" in result["error"]
    finally:
        os.unlink(db)


# AC-1004: スコア範囲外（total_score=101.0）で失敗する
def test_1004_score_out_of_range_fails():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        result = record_screening_result(**_valid_kwargs(total_score=101.0, db_path=db))
        assert result["success"] is False
        # DB に INSERT されていないこと
        conn = sqlite3.connect(db)
        # テーブルが作成されていない可能性もある（バリデーションで早期リターン）
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='screening_records'"
        ).fetchall()
        if tables:
            count = conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()[0]
            assert count == 0
        conn.close()
    finally:
        os.unlink(db)


# AC-1005: 不正な outcome 値で失敗する
def test_1005_invalid_outcome_fails():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        result = record_screening_result(
            **_valid_kwargs(outcome="approved", db_path=db)
        )
        assert result["success"] is False
        assert "invalid outcome" in result["error"]
    finally:
        os.unlink(db)


# AC-1006: input_snapshot の PII が REDACTED される
def test_1006_pii_redacted():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        snapshot = {"name": "山田太郎", "total_score": 85.0}
        result = record_screening_result(
            **_valid_kwargs(input_snapshot=snapshot, db_path=db)
        )
        assert result["success"] is True
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT input_snapshot FROM screening_records WHERE id=?",
            (result["record_id"],),
        ).fetchone()
        conn.close()
        saved = json.loads(row[0])
        assert saved["name"] == "[REDACTED]"
        assert saved["total_score"] == 85.0
    finally:
        os.unlink(db)


# AC-1007: outcome の後付け UPDATE が成功する
def test_1007_update_outcome_success():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        ins = record_screening_result(**_valid_kwargs(db_path=db))
        assert ins["success"] is True

        upd = update_screening_outcome(
            case_id="test-case-001", outcome="contracted", db_path=db
        )
        assert upd["success"] is True

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT outcome FROM screening_records WHERE id=?", (ins["record_id"],)
        ).fetchone()
        conn.close()
        assert row[0] == "contracted"
    finally:
        os.unlink(db)


# AC-1008: DB が存在しない状態でも正常に INSERT できる（自動作成）
def test_1008_auto_create_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = os.path.join(tmpdir, "new_subdir", "auto.db")
        result = record_screening_result(**_valid_kwargs(db_path=db))
        assert result["success"] is True
        assert os.path.exists(db)


# AC-1009: DB 接続失敗でも例外が外部に伝播しない
def test_1009_no_exception_on_permission_error():
    bad_path = "/root/no_permission_p4001.db"
    try:
        result = record_screening_result(**_valid_kwargs(db_path=bad_path))
        # 例外は起きず success=False が返る
        assert result["success"] is False
        assert result["error"] is not None
    except Exception as e:
        raise AssertionError(f"例外が外部に伝播した: {e}") from e


# AC-1010: パフォーマンス要件（50回連続で 5000ms 以内）
def test_1010_performance_50_calls():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        start = time.time()
        for i in range(50):
            record_screening_result(**_valid_kwargs(case_id=f"perf-{i}", db_path=db))
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5000, f"50回の処理に {elapsed_ms:.0f}ms かかった（上限 5000ms）"
    finally:
        os.unlink(db)
