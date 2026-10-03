#!/bin/bash
# Install/reload the dedicated 05:30 Vertex credit daily LaunchAgent.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PLIST="${PROJECT_ROOT}/launchd/com.tunelease.vertex-credit-daily.plist"
TARGET_DIR="${HOME}/Library/LaunchAgents"
TARGET_PLIST="${TARGET_DIR}/com.tunelease.vertex-credit-daily.plist"
DOMAIN="gui/$(id -u)"
SERVICE="${DOMAIN}/com.tunelease.vertex-credit-daily"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
BASE_PATH="${PROJECT_ROOT}/.venv/bin:${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
if command -v gcloud >/dev/null 2>&1; then
  GCLOUD_BIN_DIR="$(dirname "$(command -v gcloud)")"
elif [ -x "${HOME}/google-cloud-sdk/bin/gcloud" ]; then
  GCLOUD_BIN_DIR="${HOME}/google-cloud-sdk/bin"
else
  GCLOUD_BIN_DIR=""
fi
LAUNCH_PATH="${GCLOUD_BIN_DIR:+${GCLOUD_BIN_DIR}:}${BASE_PATH}"

source "${PROJECT_ROOT}/scripts/resolve_obsidian_vault.sh"
VAULT="$(resolve_obsidian_vault)"
if [ -z "${VAULT}" ]; then
  echo "Obsidian Vault path could not be resolved." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}" "${HOME}/Library/Logs"
cp "${SOURCE_PLIST}" "${TARGET_PLIST}"

# The checked-in plist documents the production shape; resolve machine-local
# paths at install time so a fresh clone does not retain another user's paths.
plutil -replace ProgramArguments.1 -string "${PROJECT_ROOT}/scripts/run_vertex_credit_daily.sh" "${TARGET_PLIST}"
plutil -replace WorkingDirectory -string "${PROJECT_ROOT}" "${TARGET_PLIST}"
plutil -replace EnvironmentVariables.PYTHONPATH -string "${PROJECT_ROOT}" "${TARGET_PLIST}"
plutil -replace EnvironmentVariables.PYTHON -string "${PYTHON}" "${TARGET_PLIST}"
plutil -replace EnvironmentVariables.OBSIDIAN_VAULT_PATH -string "${VAULT}" "${TARGET_PLIST}"
plutil -replace EnvironmentVariables.OBSIDIAN_VAULT -string "${VAULT}" "${TARGET_PLIST}"
plutil -replace EnvironmentVariables.PATH -string "${LAUNCH_PATH}" "${TARGET_PLIST}"
plutil -replace StandardOutPath -string "${HOME}/Library/Logs/tune_lease_55_vertex_credit_daily.out.log" "${TARGET_PLIST}"
plutil -replace StandardErrorPath -string "${HOME}/Library/Logs/tune_lease_55_vertex_credit_daily.err.log" "${TARGET_PLIST}"
plutil -lint "${TARGET_PLIST}" >/dev/null

launchctl bootout "${SERVICE}" 2>/dev/null || true
launchctl bootstrap "${DOMAIN}" "${TARGET_PLIST}"
launchctl enable "${SERVICE}"

echo "Installed: ${SERVICE}"
echo "Schedule: daily at 05:30 (not started immediately)"
echo "Plist: ${TARGET_PLIST}"
