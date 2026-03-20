"""
shinsa_gunshi_db.py — DB操作モジュール（軍師モード）

SQLite を使った案件データの初期化・保存・更新・取得を担う。
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime

# ==============================================================================
# パス設定
# ==============================================================================
_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_DIR)  # lease_logic_sumaho12/
GUNSHI_DB_PATH = os.path.join(_BASE, "data", "lease_data.db")


# ==============================================================================
# DB 初期化
# ==============================================================================

def init_db() -> None:
    """SQLite テーブルを初期化（なければ作成）。マイグレーションも実行。"""
    with closing(sqlite3.connect(GUNSHI_DB_PATH)) as conn:
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS gunshi_cases (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     TEXT    NOT NULL,
                industry       TEXT    NOT NULL,
                score          REAL    NOT NULL,
                pd_pct         REAL    NOT NULL,
                resale         TEXT    NOT NULL,
                repeat_cnt     INTEGER NOT NULL DEFAULT 0,
                subsidy        INTEGER NOT NULL DEFAULT 0,
                bank           INTEGER NOT NULL DEFAULT 0,
                intuition      INTEGER NOT NULL DEFAULT 3,
                prior_prob     REAL    NOT NULL,
                posterior      REAL    NOT NULL,
                result         TEXT    NOT NULL DEFAULT '未登録',
                notes          TEXT    NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS phrase_weights (
                phrase_id   TEXT    PRIMARY KEY,
                industry    TEXT    NOT NULL,
                wins        INTEGER NOT NULL DEFAULT 0,
                total       INTEGER NOT NULL DEFAULT 0
            );
        """)
        # マイグレーション: lease_case_id カラムを追加（既存DBに対応）
        existing_cols = [r[1] for r in cur.execute("PRAGMA table_info(gunshi_cases)").fetchall()]
        if "lease_case_id" not in existing_cols:
            cur.execute("ALTER TABLE gunshi_cases ADD COLUMN lease_case_id TEXT DEFAULT ''")
        conn.commit()


def save_case(data: dict) -> int:
    """案件をDBに保存し、生成された ID を返す。"""
    with closing(sqlite3.connect(GUNSHI_DB_PATH)) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO gunshi_cases
                (created_at, industry, score, pd_pct, resale, repeat_cnt,
                 subsidy, bank, intuition, prior_prob, posterior, result, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            data["industry"], data["score"], data["pd_pct"],
            data["resale"], data["repeat_cnt"], int(data["subsidy"]),
            int(data["bank"]), data["intuition"],
            data["prior_prob"], data["posterior"],
            data.get("result", "未登録"), data.get("notes", ""),
        ))
        row_id = cur.lastrowid
        conn.commit()
    return row_id


def update_result(case_id: int, result: str, notes: str = "") -> None:
    """成約/非成約を登録する。"""
    with closing(sqlite3.connect(GUNSHI_DB_PATH)) as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE gunshi_cases SET result=?, notes=? WHERE id=?",
            (result, notes, case_id)
        )
        conn.commit()


def load_history(limit: int = 200) -> list[dict]:
    """最新 n 件の案件履歴を返す。"""
    with closing(sqlite3.connect(GUNSHI_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM gunshi_cases ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = [dict(r) for r in cur.fetchall()]
    return rows


def get_success_patterns(industry: str) -> dict:
    """同業種の成約率と成功事例 3 件を返す。"""
    with closing(sqlite3.connect(GUNSHI_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM gunshi_cases WHERE industry=? ORDER BY id DESC LIMIT 50",
            (industry,)
        )
        rows = [dict(r) for r in cur.fetchall()]

    total = len(rows)
    wins  = sum(1 for r in rows if r["result"] == "成約")
    success_ratio = wins / total if total else 0.0
    success_samples = [r for r in rows if r["result"] == "成約"][:3]
    fail_samples    = [r for r in rows if r["result"] == "非成約"][:2]
    return {
        "total": total,
        "wins": wins,
        "success_ratio": success_ratio,
        "success_samples": success_samples,
        "fail_samples": fail_samples,
    }
