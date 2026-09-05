#!/bin/bash
# Claude Code on the web / Dispatch のリモートコンテナ向けセットアップ。
#
# 背景: リモートコンテナはリポジトリを clone するだけで依存を持たない。
# そのため CLAUDE.md が PR 前に必須としている `npx tsc --noEmit` も
# `make test` も最初の1コマンド目で落ち、セッションが成果物なしで終わっていた。
# ここで CI (.github/workflows/pr-checks.yml) と同じ依存集合を入れておく。
#
# ローカル Mac は .venv + launchd 運用のため対象外（CLAUDE_CODE_REMOTE で分岐）。
#
# 方針:
#   - 冪等（導入済みならスキップ。ただし「一部だけ入っている」は未導入として扱う）
#   - 非対話
#   - 途中で失敗してもセッション起動自体は止めない（常に exit 0）

set -uo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 0

# ── Python: pr-checks.yml の pytest-core + preflight-guard と同じ集合 ──
# pip 名と import 名は一致しないものがあるため両方を持つ。片方を足したら
# もう片方も必ず更新すること。
PY_PIP_PACKAGES=(
  pytest fastapi uvicorn pydantic slowapi httpx python-multipart
  requests numpy pandas matplotlib seaborn scipy scikit-learn==1.7.2 joblib
  python-dateutil pyyaml beautifulsoup4 filelock bcrypt flask flask-cors
  google-api-core lxml toml chromadb google-adk apscheduler pyflakes
)
PY_IMPORT_NAMES="pytest fastapi uvicorn pydantic slowapi httpx multipart \
requests numpy pandas matplotlib seaborn scipy sklearn joblib \
dateutil yaml bs4 filelock bcrypt flask flask_cors \
google.api_core lxml toml chromadb google.adk apscheduler pyflakes"

# 代表2つではなく全依存を確認する。一部だけ入っている環境で pytest や
# api の import が落ちるのを防ぐ。find_spec は実行を伴わないため約20ms。
py_deps_complete() {
  python3 -c "
import importlib.util, sys
try:
    ok = all(importlib.util.find_spec(m) is not None for m in '''${PY_IMPORT_NAMES}'''.split())
except Exception:
    ok = False
sys.exit(0 if ok else 1)
" >/dev/null 2>&1
}

if ! py_deps_complete; then
  echo "[session-start] Python 依存をインストール中..."
  # --ignore-installed PyYAML: Debian の apt 版 PyYAML は RECORD が無く pip が
  # アンインストールできないため、これを指定しないとインストール全体が中断する。
  python3 -m pip install -q --disable-pip-version-check --ignore-installed PyYAML \
    "${PY_PIP_PACKAGES[@]}" 2>&1
  if py_deps_complete; then
    echo "[session-start] Python 依存: OK"
  else
    echo "[session-start] ⚠️ Python 依存が揃いませんでした。pytest / preflight_pr_guard.py は動きません。"
  fi
fi

# ── Node: frontend の tsc --noEmit / eslint ──
# npm install ではなく npm ci。npm install は package-lock.json の peer 情報を
# 書き換えてしまい、毎セッション作業ツリーが汚れて生成物混入ガード
# (scripts/check_pr_change_risk.py) の誤爆源になる。lock が壊れている時だけ
# npm install にフォールバックする。
#
# 完了判定は node_modules/ の有無ではなく専用の目印にする。レジストリ障害等で
# 中途半端な node_modules/ が残ると、ディレクトリ判定では以降のセッションが
# 永久にインストールをスキップして復旧できなくなるため。
# 目印は node_modules/ の中に置くので、node_modules/ を消せば一緒に消える。
NODE_MARKER="frontend/node_modules/.session-start-complete"

if [ ! -f "$NODE_MARKER" ]; then
  echo "[session-start] frontend 依存をインストール中..."
  if (cd frontend && { npm ci --no-audit --no-fund --silent \
        || npm install --no-audit --no-fund --silent; }) 2>&1; then
    : > "$NODE_MARKER" 2>/dev/null
    echo "[session-start] frontend 依存: OK"
  else
    echo "[session-start] ⚠️ frontend 依存のインストールに失敗。npx tsc --noEmit / npm run lint は動きません（次回セッションで再試行します）。"
  fi
fi

# ── graft: 常時適用スキル・.mcp.json の graft サーバが依存するCLI (@nanonets/graft) ──
# ローカルMacはグローバルインストール済み想定（.claude/helpers/graft-hooks.cjs の
# BAKED 定数参照）。リモートコンテナは毎回まっさらなので、ここで揃えないと
# .mcp.json の graft サーバが「Executable not found in $PATH」で接続失敗する。
if ! command -v graft >/dev/null 2>&1; then
  echo "[session-start] graft CLI をインストール中..."
  if npm install -g @nanonets/graft --no-audit --no-fund 2>&1; then
    echo "[session-start] graft CLI: OK"
  else
    echo "[session-start] ⚠️ graft CLI のインストールに失敗。graft skill/MCPはrg/Readへフォールバックします（次回セッションで再試行します）。"
  fi
fi

# リポジトリ直下の scoring_core.py 等を import するテスト・スクリプトのため固定する。
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PYTHONPATH=\"${PROJECT_DIR}:\${PYTHONPATH:-}\"" >> "$CLAUDE_ENV_FILE"
fi

exit 0
