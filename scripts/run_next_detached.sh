#!/bin/bash
# Start the persistent Next/FastAPI/Cloudflare launcher via macOS launchd.
#
# Codex-managed shells may clean up nohup/background children when the command
# finishes. launchd is the reliable detached owner for this local app.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/next

bash scripts/install_next_launchagent.sh
