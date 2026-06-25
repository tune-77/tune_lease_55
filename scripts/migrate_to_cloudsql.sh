#!/usr/bin/env bash
# =============================================================================
# scripts/migrate_to_cloudsql.sh
# REV-158: SQLite → Cloud SQL (PostgreSQL) 手動同期スクリプト
#
# 用途: デモ前に lease_data.db の全テーブルを Cloud SQL に一括流し込む
# 方針: TRUNCATE → INSERT（上書き）
#
# 必要な環境変数（事前に設定してから実行）:
#   DATABASE_URL  例: postgresql://user:pass@/dbname?host=/cloudsql/proj:region:instance
#                     または: postgresql://user:pass@HOST:5432/dbname
#
# 使い方:
#   export DATABASE_URL="postgresql://..."
#   bash scripts/migrate_to_cloudsql.sh
#
#   # ローカル DB パスを明示する場合:
#   SQLITE_DB_PATH=/path/to/lease_data.db bash scripts/migrate_to_cloudsql.sh
# =============================================================================

set -euo pipefail

# ─── 設定 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

SQLITE_DB_PATH="${SQLITE_DB_PATH:-${REPO_ROOT}/data/lease_data.db}"
DATABASE_URL="${DATABASE_URL:-}"

# ─── 前提チェック ──────────────────────────────────────────────────────────────
if [[ -z "${DATABASE_URL}" ]]; then
  echo "❌  DATABASE_URL が設定されていません。"
  echo "    export DATABASE_URL=\"postgresql://user:pass@host:5432/dbname\" を実行してください。"
  exit 1
fi

if [[ ! -f "${SQLITE_DB_PATH}" ]]; then
  echo "❌  SQLite DB が見つかりません: ${SQLITE_DB_PATH}"
  exit 1
fi

# psycopg2 の確認
python3 -c "import psycopg2" 2>/dev/null || {
  echo "❌  psycopg2-binary がインストールされていません。"
  echo "    pip install psycopg2-binary --break-system-packages"
  exit 1
}

echo "============================================================"
echo "  lease_data.db → Cloud SQL 同期スクリプト (REV-158)"
echo "============================================================"
echo "  Source : ${SQLITE_DB_PATH}"
echo "  Target : ${DATABASE_URL%%@*}@***  (URL を一部マスク)"
echo ""
echo "  ⚠️  対象テーブルを TRUNCATE してから INSERT します。"
read -r -p "  続行しますか？ [y/N]: " confirm
[[ "${confirm}" =~ ^[Yy]$ ]] || { echo "中止しました。"; exit 0; }
echo ""

# ─── Python で移行実行 ─────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import os, sys, sqlite3, json
from datetime import datetime

DATABASE_URL = os.environ["DATABASE_URL"]
SQLITE_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lease_data.db")
)

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.extensions
except ImportError:
    print("❌  psycopg2-binary のインポートに失敗しました。")
    sys.exit(1)

# ── テーブル順序（外部キー依存があれば調整） ──────────────────────────────
# sqlite_sequence は PostgreSQL では不要なので除外
SKIP_TABLES = {"sqlite_sequence"}

def get_sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [r[0] for r in cur.fetchall() if r[0] not in SKIP_TABLES]

def get_sqlite_schema(conn: sqlite3.Connection, table: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info(\"{table}\")")
    # cid, name, type, notnull, dflt_value, pk
    return [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in cur.fetchall()]

def sqlite_type_to_pg(sqlite_type: str) -> str:
    """SQLite 型を PostgreSQL 型にマッピング（best-effort）。"""
    t = sqlite_type.upper().strip()
    if t in ("INTEGER", "INT"):
        return "BIGINT"
    if t in ("REAL", "FLOAT", "DOUBLE"):
        return "DOUBLE PRECISION"
    if t in ("BOOLEAN", "BOOL"):
        return "BOOLEAN"
    if t.startswith("TIMESTAMP"):
        return "TEXT"  # 互換性重視で TEXT
    if t in ("BLOB",):
        return "BYTEA"
    return "TEXT"  # TEXT をデフォルト

def ensure_pg_table(pg_cur, table: str, schema: list[dict]):
    """テーブルが存在しなければ CREATE TABLE（存在すれば TRUNCATE）。"""
    col_defs = []
    for col in schema:
        pg_type = sqlite_type_to_pg(col["type"])
        not_null = "NOT NULL" if col["notnull"] else ""
        pk = "PRIMARY KEY" if col["pk"] == 1 else ""
        col_defs.append(f'  "{col["name"]}" {pg_type} {pk} {not_null}'.strip())
    ddl = f'CREATE TABLE IF NOT EXISTS "{table}" (\n' + ",\n".join(col_defs) + "\n);"
    pg_cur.execute(ddl)
    pg_cur.execute(f'TRUNCATE TABLE "{table}"')

def migrate_table(pg_conn, pg_cur, sqlite_conn, table: str):
    schema = get_sqlite_schema(sqlite_conn, table)
    ensure_pg_table(pg_cur, table, schema)

    sqlite_cur = sqlite_conn.cursor()
    sqlite_cur.execute(f'SELECT * FROM "{table}"')
    rows = sqlite_cur.fetchall()

    if not rows:
        print(f"  ✅  {table}: 0 行（スキップ）")
        return

    col_names = [f'"{c["name"]}"' for c in schema]
    placeholders = ", ".join(["%s"] * len(schema))
    insert_sql = (
        f'INSERT INTO "{table}" ({", ".join(col_names)}) VALUES ({placeholders})'
    )

    # NULL・bool変換
    def coerce_row(row):
        result = []
        for val, col in zip(row, schema):
            if val is None:
                result.append(None)
            elif col["type"].upper() in ("BOOLEAN", "BOOL"):
                result.append(bool(val))
            else:
                result.append(val)
        return result

    batch = [coerce_row(r) for r in rows]
    psycopg2.extras.execute_batch(pg_cur, insert_sql, batch, page_size=200)
    pg_conn.commit()
    print(f"  ✅  {table}: {len(batch)} 行を INSERT")

# ─── メイン処理 ────────────────────────────────────────────────────────────────
started_at = datetime.now()
print(f"\n[{started_at.strftime('%Y-%m-%d %H:%M:%S')}] 移行開始\n")

sqlite_conn = sqlite3.connect(SQLITE_DB_PATH, timeout=15)
pg_conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
pg_conn.autocommit = False
pg_cur = pg_conn.cursor()

tables = get_sqlite_tables(sqlite_conn)
print(f"移行対象テーブル: {len(tables)} 件")

errors = []
for table in tables:
    try:
        migrate_table(pg_conn, pg_cur, sqlite_conn, table)
    except Exception as e:
        pg_conn.rollback()
        print(f"  ❌  {table}: エラー → {e}")
        errors.append((table, str(e)))

sqlite_conn.close()
pg_cur.close()
pg_conn.close()

elapsed = (datetime.now() - started_at).total_seconds()
print(f"\n[完了] {elapsed:.1f} 秒")
if errors:
    print(f"\n⚠️  {len(errors)} テーブルでエラーが発生しました:")
    for t, msg in errors:
        print(f"   - {t}: {msg}")
    sys.exit(1)
else:
    print("✅  全テーブルの移行が完了しました。")
PYEOF

echo ""
echo "============================================================"
echo "  同期完了"
echo "============================================================"
