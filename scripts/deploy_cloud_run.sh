#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/scripts/deploy_cloud_run_api.sh"
"$ROOT_DIR/scripts/deploy_cloud_run_web.sh"
