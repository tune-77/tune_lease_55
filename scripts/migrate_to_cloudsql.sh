#!/usr/bin/env bash
# =============================================================================
# scripts/migrate_to_cloudsql.sh
# Legacy-compatible wrapper for scripts/migrate_to_cloud_sql.py.
#
# The old implementation duplicated migration logic and committed per table.
# Keep this entrypoint for existing docs/operators, but delegate to the safer
# single-transaction Python migration script.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_MIGRATOR="${SCRIPT_DIR}/migrate_to_cloud_sql.py"

if [[ ! -f "${PY_MIGRATOR}" ]]; then
  echo "❌  Migration script not found: ${PY_MIGRATOR}" >&2
  exit 1
fi

echo "============================================================"
echo "  SQLite → Cloud SQL migration wrapper"
echo "============================================================"
echo "  Delegating to: ${PY_MIGRATOR}"
if [[ -n "${DATABASE_URL:-}" ]]; then
  echo "  DATABASE_URL : set"
else
  echo "  DATABASE_URL : unset"
fi
echo ""

exec python3 "${PY_MIGRATOR}" "$@"
