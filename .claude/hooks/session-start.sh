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
#   - 冪等（インストール済みならスキップ）
#   - 非対話
#   - 途中で失敗してもセッション起動自体は止めない（常に exit 0）

set -uo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 0

# ── Python: pr-checks.yml の pytest-core + preflight-guard と同じ集合 ──
# 判定は代表2パッケージ。両方入っていれば再インストールしない。
if ! python3 -c "import pytest, fastapi" >/dev/null 2>&1; then
  echo "[session-start] Python 依存をインストール中..."
  # --ignore-installed PyYAML: Debian の apt 版 PyYAML は RECORD が無く pip が
  # アンインストールできないため、これを指定しないとインストール全体が中断する。
  if python3 -m pip install -q --disable-pip-version-check --ignore-installed PyYAML \
      pytest fastapi uvicorn pydantic slowapi httpx python-multipart \
      requests numpy pandas matplotlib seaborn scipy scikit-learn==1.7.2 joblib \
      python-dateutil pyyaml beautifulsoup4 filelock bcrypt flask flask-cors \
      google-api-core lxml toml chromadb google-adk apscheduler pyflakes 2>&1; then
    echo "[session-start] Python 依存: OK"
  else
    echo "[session-start] ⚠️ Python 依存のインストールに失敗。pytest / preflight_pr_guard.py は動きません。"
  fi
fi

# ── Node: frontend の tsc --noEmit / eslint ──
# npm install ではなく npm ci。npm install は package-lock.json の peer 情報を
# 書き換えてしまい、毎セッション作業ツリーが汚れて生成物混入ガード
# (scripts/check_pr_change_risk.py) の誤爆源になる。lock が壊れている時だけ
# npm install にフォールバックする。
if [ ! -d frontend/node_modules ]; then
  echo "[session-start] frontend 依存をインストール中..."
  if (cd frontend && { npm ci --no-audit --no-fund --silent \
        || npm install --no-audit --no-fund --silent; }) 2>&1; then
    echo "[session-start] frontend 依存: OK"
  else
    echo "[session-start] ⚠️ npm install に失敗。npx tsc --noEmit / npm run lint は動きません。"
  fi
fi

# リポジトリ直下の scoring_core.py 等を import するテスト・スクリプトのため固定する。
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PYTHONPATH=\"${PROJECT_DIR}:\${PYTHONPATH:-}\"" >> "$CLAUDE_ENV_FILE"
fi

exit 0
