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
    from runtime_paths import get_db_path  # type: ignore[import]

    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
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
    conn = psycopg2.connect(database_url, cursor_factory=psycopg2.extras.DictCursor)
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
