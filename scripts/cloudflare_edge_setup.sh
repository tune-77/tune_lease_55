#!/bin/bash
# shion.tune77.com の Cloudflare エッジ保護（Access / 回数制限 / 休止ページ Worker）を設定する。
#   bash scripts/cloudflare_edge_setup.sh            # ドライラン（差分表示のみ）
#   bash scripts/cloudflare_edge_setup.sh --apply    # 反映
#   bash scripts/cloudflare_edge_setup.sh --verify   # 反映後の確認
# API トークンはキーチェーン（サービス cloudflare-api-token / アカウント tune-lease-55）から読み、
# 環境変数でだけ Python に渡す。表示・ファイル保存はしない。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if ! CLOUDFLARE_API_TOKEN="$(security find-generic-password -s cloudflare-api-token -a tune-lease-55 -w 2>/dev/null)"; then
  echo "キーチェーンに Cloudflare API トークンがありません（サービス cloudflare-api-token / アカウント tune-lease-55）。" >&2
  exit 2
fi
export CLOUDFLARE_API_TOKEN

PYTHON="$ROOT_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"
exec "$PYTHON" "$ROOT_DIR/scripts/cloudflare_edge_setup.py" "$@"
