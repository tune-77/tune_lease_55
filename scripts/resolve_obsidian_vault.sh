# shellcheck shell=bash
#
# Vault パスを runtime_paths（唯一の解決窓口）から取得するための共有ヘルパー。
#
# シェル側に既定パスを直書きすると、Python 側の解決結果とずれて
# 「書き込み先」と「RAG 索引先」が分裂する。必ずこの関数を使う。
#
# 使い方:
#     source "$(dirname "${BASH_SOURCE[0]}")/resolve_obsidian_vault.sh"
#     VAULT="$(resolve_obsidian_vault)"
#
# 解決できない場合は空文字を返す（呼び出し側でフォールバックすること）。
# 呼び出し側が `set -e` でも落とさないよう、失敗は握って空文字にする。

resolve_obsidian_vault() {
  local root py
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  py="${PYTHON_BIN:-${root}/.venv/bin/python}"
  if [ ! -x "$py" ]; then
    py="$(command -v python3 || true)"
  fi
  [ -n "$py" ] || return 0

  "$py" - "$root" <<'PYEOF' 2>/dev/null || true
import sys

sys.path.insert(0, sys.argv[1])
from runtime_paths import resolve_obsidian_vault

print(resolve_obsidian_vault())
PYEOF
}
