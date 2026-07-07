"""DB 接続レイヤー（REV-158: Cloud SQL 移行基盤）

環境変数 ``DATABASE_URL`` の有無で接続先を切り替える。

- **未設定（ローカル開発）**: 従来通り SQLite（``runtime_paths.get_db_path()`` を使用）
- **設定済み（Cloud Run）**: ``DATABASE_URL`` で指定した PostgreSQL（Cloud SQL）に接続

使用方法::

    from api.db_connection import get_connection, placeholder

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM past_cases WHERE id = {placeholder()}", (case_id,))
        rows = cur.fetchall()

注意:
- 既存コード（``sqlite3.connect()`` を直接呼んでいる箇所）は今回変更しない。
  新規エンドポイントや以降の PR で段階的に置き換えていく。
- ``get_connection()`` はコンテキストマネージャとして使う。正常終了時 commit、例外時 rollback。
- PostgreSQL 接続時はプールを使わずシンプルに 1 コネクション生成（Cloud Run 1 インスタンス想定）。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, Union

# ── 型エイリアス ───────────────────────────────────────────────────────────────
# sqlite3.Connection と psycopg2.extensions.connection の共通型
AnyConnection = Union[sqlite3.Connection, "psycopg2.extensions.connection"]  # type: ignore[name-defined]


def _is_postgres() -> bool:
    """DATABASE_URL が設定されていれば PostgreSQL モード。"""
    return bool(os.environ.get("DATABASE_URL", "").strip())


def placeholder() -> str:
    """SQL パラメータプレースホルダーを返す。

    - SQLite: ``?``
    - PostgreSQL: ``%s``
    """
    return "%s" if _is_postgres() else "?"


@contextmanager
def get_connection() -> Generator[AnyConnection, None, None]:
    """DB 接続をコンテキストマネージャとして提供する。

    Example::

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
    """
    if _is_postgres():
        with _postgres_connection() as conn:
            yield conn
    else:
        with _sqlite_connection() as conn:
            yield conn


# ── 内部実装 ───────────────────────────────────────────────────────────────────

@contextmanager
def _sqlite_connection() -> Generator[sqlite3.Connection, None, None]:
    """SQLite 接続（ローカル開発用）。"""
    # runtime_paths は REPO ルートにあるため、相対インポートを避けて動的に解決
    import sys
    import os as _os
    _repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from runtime_paths import ensure_cloudrun_demo_db_seeded, get_db_path  # type: ignore[import]

    ensure_cloudrun_demo_db_seeded()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _postgres_connection() -> Generator["psycopg2.extensions.connection", None, None]:  # type: ignore[name-defined]
    """PostgreSQL 接続（Cloud Run / Cloud SQL 用）。

    ``DATABASE_URL`` 形式:
    ``postgresql://USER:PASSWORD@HOST:PORT/DBNAME``

    Cloud SQL（Unix ソケット）の場合:
    ``postgresql://USER:PASSWORD@/DBNAME?host=/cloudsql/PROJECT:REGION:INSTANCE``
    """
    try:
        import psycopg2  # type: ignore[import]
        import psycopg2.extras  # type: ignore[import]
    except ImportError as e:
        raise RuntimeError(
            "psycopg2-binary がインストールされていません。"
            "`pip install psycopg2-binary` を実行してください。"
        ) from e

    database_url = os.environ["DATABASE_URL"]
    # DictCursor は row["col"] / row[0] / dict(row) を両立する。
    # SQLite の sqlite3.Row に近い振る舞いにして、既存APIのタプル前提も壊しにくくする。
    conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.DictCursor, connect_timeout=5)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── ユーティリティ ──────────────────────────────────────────────────────────────

def current_backend() -> str:
    """現在の接続バックエンド名を返す（ログ・デバッグ用）。"""
    return "postgresql" if _is_postgres() else "sqlite"


def ensure_schema() -> None:
    """全コアテーブルを冪等に作成する（Cloud SQL 初回起動時のスキーマ自動初期化）。

    既存テーブルには一切影響しない（CREATE TABLE IF NOT EXISTS）。
    SQLite と PostgreSQL の両方で動作する。
    """
    is_pg = _is_postgres()
    auto_pk = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real_t = "DOUBLE PRECISION" if is_pg else "REAL"
    bool_false = "FALSE" if is_pg else "0"

    _DDL = [
        # past_cases ─────────────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS past_cases (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            industry_sub TEXT,
            score {real_t},
            user_eq {real_t},
            final_status TEXT,
            data TEXT,
            sales_dept TEXT DEFAULT '未設定',
            registration_date TEXT,
            estimate_sent_date TEXT,
            customer_response_date TEXT,
            final_result_date TEXT
        )""",
        # excluded_grade_cases ───────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS excluded_grade_cases (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            industry_sub TEXT,
            score {real_t},
            user_eq {real_t},
            final_status TEXT,
            data TEXT,
            sales_dept TEXT,
            registration_date TEXT,
            estimate_sent_date TEXT,
            customer_response_date TEXT,
            final_result_date TEXT,
            original_grade TEXT,
            excluded_reason TEXT,
            extracted_at TEXT
        )""",
        # screening_records ──────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS screening_records (
            id {auto_pk},
            case_id TEXT NOT NULL,
            screened_at TEXT NOT NULL,
            total_score {real_t} NOT NULL,
            asset_score {real_t} NOT NULL,
            tenant_score {real_t},
            q_risk_score {real_t},
            competitor_pressure_score {real_t},
            outcome TEXT,
            input_snapshot TEXT,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        )""",
        # payment_history ────────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS payment_history (
            id {auto_pk},
            contract_id TEXT NOT NULL,
            check_date TEXT NOT NULL,
            payment_status TEXT NOT NULL,
            overdue_amount INTEGER DEFAULT 0,
            model_version TEXT DEFAULT '',
            screening_score {real_t},
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT ''
        )""",
        # subsidy_master ─────────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS subsidy_master (
            id {auto_pk},
            name TEXT NOT NULL,
            max_amount INTEGER NOT NULL DEFAULT 0,
            industry_codes TEXT NOT NULL DEFAULT '',
            asset_keywords TEXT NOT NULL DEFAULT '',
            deadline TEXT NOT NULL DEFAULT '随時',
            url TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1
        )""",
        # conversation_history ───────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS conversation_history (
            id {auto_pk},
            session_id TEXT NOT NULL,
            company_name TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        # emotion_feedback ───────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS emotion_feedback (
            id {auto_pk},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rating TEXT NOT NULL,
            comment TEXT,
            emotion_category TEXT,
            resolved BOOLEAN DEFAULT {bool_false}
        )""",
        # emotion_history ────────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS emotion_history (
            id {auto_pk},
            recorded_at TEXT NOT NULL,
            hopeful_anxiety {real_t},
            careful_attachment {real_t},
            intellectual_excitement {real_t},
            unrewarded_effort {real_t},
            quiet_loneliness {real_t},
            earned_confidence {real_t},
            protective_frustration {real_t},
            dominant_raw_emotion TEXT,
            notes TEXT
        )""",
        # chat_messages ──────────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS chat_messages (
            id {auto_pk},
            user_id TEXT NOT NULL DEFAULT 'default',
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        # shion_screening_reviews ─────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS shion_screening_reviews (
            id {auto_pk},
            case_id TEXT,
            company_name TEXT,
            industry_major TEXT,
            industry_sub TEXT,
            sales_dept TEXT,
            score {real_t},
            hantei TEXT,
            q_risk {real_t},
            umap_anomaly_score {real_t},
            memory_refs INTEGER DEFAULT 0,
            knowledge_refs INTEGER DEFAULT 0,
            identity_used BOOLEAN DEFAULT {bool_false},
            review_text TEXT NOT NULL,
            prompt_text TEXT,
            form_snapshot TEXT,
            result_snapshot TEXT,
            user_feedback TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        # screening_experience_cases ─────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS screening_experience_cases (
            id {auto_pk},
            demo_case_id TEXT DEFAULT '',
            source_case_id TEXT DEFAULT '',
            company_name TEXT NOT NULL,
            period TEXT DEFAULT '',
            industry_major TEXT DEFAULT '',
            industry_sub TEXT DEFAULT '',
            sales_dept TEXT DEFAULT '',
            score {real_t},
            decision TEXT DEFAULT '',
            outcome TEXT DEFAULT '',
            similarity TEXT DEFAULT '',
            action_taken TEXT DEFAULT '',
            lesson TEXT DEFAULT '',
            difference TEXT DEFAULT '',
            source TEXT DEFAULT 'manual',
            form_snapshot TEXT DEFAULT '',
            result_snapshot TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        # sync_log ───────────────────────────────────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS sync_log (
            id {auto_pk},
            pushed_at TEXT NOT NULL,
            success INTEGER NOT NULL,
            error TEXT
        )""",
    ]
    _IDX = [
        "CREATE INDEX IF NOT EXISTS idx_conv_company ON conversation_history(company_name)",
        "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_emofb_resolved ON emotion_feedback(resolved)",
        "CREATE INDEX IF NOT EXISTS idx_emotion_history_date ON emotion_history(recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_shion_screening_reviews_case_id ON shion_screening_reviews(case_id)",
        "CREATE INDEX IF NOT EXISTS idx_shion_screening_reviews_created ON shion_screening_reviews(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_shion_screening_reviews_industry ON shion_screening_reviews(industry_sub)",
        "CREATE INDEX IF NOT EXISTS idx_screening_experience_demo ON screening_experience_cases(demo_case_id)",
        "CREATE INDEX IF NOT EXISTS idx_screening_experience_industry ON screening_experience_cases(industry_sub)",
        "CREATE INDEX IF NOT EXISTS idx_screening_experience_created ON screening_experience_cases(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_screening_records_case_id ON screening_records(case_id)",
        "CREATE INDEX IF NOT EXISTS idx_screening_records_screened_at ON screening_records(screened_at)",
        "CREATE INDEX IF NOT EXISTS idx_screening_records_outcome ON screening_records(outcome)",
        "CREATE INDEX IF NOT EXISTS idx_ph_contract_id ON payment_history(contract_id)",
        "CREATE INDEX IF NOT EXISTS idx_ph_check_date ON payment_history(check_date)",
        "CREATE INDEX IF NOT EXISTS idx_ph_payment_status ON payment_history(payment_status)",
    ]

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            for stmt in _DDL + _IDX:
                cur.execute(stmt)
            if not is_pg:
                conn.commit()
        print(f"[ensure_schema] スキーマ初期化完了 ({current_backend()})")
    except Exception as e:
        print(f"[ensure_schema] テーブル作成失敗（非致命的）: {e}")
