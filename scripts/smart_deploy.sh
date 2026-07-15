#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGION="${REGION:-asia-northeast1}"
API_SERVICE_NAME="${API_SERVICE_NAME:-tune-lease-55-api}"
WEB_SERVICE_NAME="${WEB_SERVICE_NAME:-tune-lease-55-web}"
CHECK_TIMEOUT="${CHECK_TIMEOUT:-30}"

FORCE_API=0
FORCE_WEB=0
DRY_RUN=0
SKIP_CHECKS=0

usage() {
  cat <<'USAGE'
Usage: scripts/smart_deploy.sh [options]

Deploy only the Cloud Run services that need updating.

Options:
  --api          Deploy API regardless of detected changes
  --web          Deploy Web regardless of detected changes
  --all          Deploy API and Web
  --force        Same as --all
  --dry-run      Show the decision without deploying
  --skip-checks  Deploy without post-deploy curl checks
  -h, --help     Show this help

Decision sources:
  - Compare current git HEAD with the image tag currently serving on Cloud Run.
  - Include staged/unstaged local changes in the path classification.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api)
      FORCE_API=1
      ;;
    --web)
      FORCE_WEB=1
      ;;
    --all|--force)
      FORCE_API=1
      FORCE_WEB=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-checks)
      SKIP_CHECKS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$ROOT_DIR"

CURRENT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"

service_image_tag() {
  local service_name="$1"
  local image
  image="$(gcloud run services describe "$service_name" \
    --region "$REGION" \
    --format='value(spec.template.spec.containers[0].image)' 2>/dev/null || true)"
  if [[ "$image" == *:* ]]; then
    printf '%s\n' "${image##*:}"
  fi
}

changed_files_between() {
  local deployed_sha="$1"
  if [[ -n "$deployed_sha" ]] && git rev-parse --verify --quiet "$deployed_sha^{commit}" >/dev/null; then
    git diff --name-only "$deployed_sha..HEAD"
  else
    git diff --name-only HEAD
  fi
  git diff --name-only --cached
  git diff --name-only
}

path_impacts_api() {
  case "$1" in
    api/*|data/*|mobile_app/*|runtime_paths.py|requirements*.txt|Dockerfile.api|cloudbuild.api.yaml)
      return 0
      ;;
    scripts/deploy_cloud_run_api.sh|scripts/package_cloud_run_bundle.sh|scripts/check_cloudrun_demo_readiness.py)
      return 0
      ;;
    scripts/build_shion_memory_index.py|scripts/gcs_vault_loader.py|scripts/cloud_init.py)
      return 0
      ;;
  esac
  return 1
}

path_impacts_web() {
  case "$1" in
    frontend/*|Dockerfile.web|cloudbuild.web.yaml|scripts/deploy_cloud_run_web.sh)
      return 0
      ;;
  esac
  return 1
}

classify_changes() {
  local deployed_api_sha="$1"
  local deployed_web_sha="$2"
  local file

  API_NEEDED=0
  WEB_NEEDED=0
  API_REASONS=()
  WEB_REASONS=()
  OTHER_REASONS=()

  API_CHANGED_FILES=()
  WEB_CHANGED_FILES=()

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    API_CHANGED_FILES+=("$file")
    if path_impacts_api "$file"; then
      API_NEEDED=1
      API_REASONS+=("$file")
    fi
  done < <(changed_files_between "$deployed_api_sha" | sort -u)

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    WEB_CHANGED_FILES+=("$file")
    if path_impacts_web "$file"; then
      WEB_NEEDED=1
      WEB_REASONS+=("$file")
    fi
  done < <(changed_files_between "$deployed_web_sha" | sort -u)

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    if ! path_impacts_api "$file" && ! path_impacts_web "$file"; then
      OTHER_REASONS+=("$file")
    fi
  done < <(
    {
      if [[ "${#API_CHANGED_FILES[@]}" -gt 0 ]]; then
        printf '%s\n' "${API_CHANGED_FILES[@]}"
      fi
      if [[ "${#WEB_CHANGED_FILES[@]}" -gt 0 ]]; then
        printf '%s\n' "${WEB_CHANGED_FILES[@]}"
      fi
    } | sort -u
  )

  if [[ "$deployed_api_sha" != "$CURRENT_SHA" && -z "$deployed_api_sha" ]]; then
    API_NEEDED=1
    API_REASONS+=("deployed API image tag is unknown")
  fi
  if [[ "$deployed_web_sha" != "$CURRENT_SHA" && -z "$deployed_web_sha" ]]; then
    WEB_NEEDED=1
    WEB_REASONS+=("deployed Web image tag is unknown")
  fi

  if [[ "$FORCE_API" -eq 1 ]]; then
    API_NEEDED=1
    API_REASONS+=("forced by option")
  fi
  if [[ "$FORCE_WEB" -eq 1 ]]; then
    WEB_NEEDED=1
    WEB_REASONS+=("forced by option")
  fi
}

print_reasons() {
  local title="$1"
  shift
  local items=("$@")
  echo "$title"
  if [[ "${#items[@]}" -eq 0 ]]; then
    echo "  - none"
    return
  fi
  local item
  for item in "${items[@]}"; do
    echo "  - $item"
  done
}

check_api() {
  local url="$1"
  local status_code
  echo "[check] API cloud-status"
  curl --max-time "$CHECK_TIMEOUT" -sS "$url/api/system/cloud-status"
  echo
  echo "[check] API score/full"
  status_code="$(curl --max-time "$CHECK_TIMEOUT" -sS \
    -o /tmp/tune_lease_smart_deploy_score.json \
    -w '%{http_code}' \
    "$url/api/score/full" \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"company_name":"テスト株式会社","industry":"サービス業","sales":1000,"operating_profit":100,"ordinary_profit":100,"total_assets":2000,"equity":500,"current_assets":800,"current_liabilities":400,"long_term_debt":300,"cash":200,"trade_receivables":100,"inventories":50,"fixed_assets":1200,"short_term_borrowings":100,"long_term_borrowings":200,"interest_expense":10,"depreciation":20,"num_employees":50,"established_year":2010,"lease_term_months":36,"lease_amount":100,"existing_relationship":"既存","payment_delay_history":"なし","bankruptcy_history":"なし","deal_source_bank":"本店","risk_notes":"","customer_new":false}')"
  echo "score/full HTTP $status_code"
  if [[ "$status_code" != "200" ]]; then
    echo "score/full check failed" >&2
    return 1
  fi
}

check_web() {
  local url="$1"
  local status_code
  echo "[check] Web /home"
  status_code="$(curl --max-time "$CHECK_TIMEOUT" -sS -o /tmp/tune_lease_smart_deploy_web.html -w '%{http_code}' "$url/home")"
  echo "web /home HTTP $status_code"
  if [[ "$status_code" != "200" && "$status_code" != "307" && "$status_code" != "308" ]]; then
    echo "web check failed" >&2
    return 1
  fi
}

DEPLOYED_API_SHA="$(service_image_tag "$API_SERVICE_NAME")"
DEPLOYED_WEB_SHA="$(service_image_tag "$WEB_SERVICE_NAME")"

classify_changes "$DEPLOYED_API_SHA" "$DEPLOYED_WEB_SHA"

echo "Current HEAD: ${CURRENT_SHA}"
echo "Deployed API: ${DEPLOYED_API_SHA:-unknown}"
echo "Deployed Web: ${DEPLOYED_WEB_SHA:-unknown}"
if [[ "${#API_REASONS[@]}" -gt 0 ]]; then
  print_reasons "API-impacting changes:" "${API_REASONS[@]}"
else
  print_reasons "API-impacting changes:"
fi
if [[ "${#WEB_REASONS[@]}" -gt 0 ]]; then
  print_reasons "Web-impacting changes:" "${WEB_REASONS[@]}"
else
  print_reasons "Web-impacting changes:"
fi
if [[ "${#OTHER_REASONS[@]}" -gt 0 ]]; then
  print_reasons "Non-deploy changes:" "${OTHER_REASONS[@]}"
fi

if [[ "$API_NEEDED" -eq 0 && "$WEB_NEEDED" -eq 0 ]]; then
  echo "No Cloud Run deploy needed."
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: API_NEEDED=$API_NEEDED WEB_NEEDED=$WEB_NEEDED"
  exit 0
fi

if [[ "$API_NEEDED" -eq 1 ]]; then
  echo "[deploy] API"
  ALLOW_UNAUTHENTICATED=1 "$ROOT_DIR/scripts/deploy_cloud_run_api.sh"
  if [[ "$SKIP_CHECKS" -eq 0 ]]; then
    API_URL="$(gcloud run services describe "$API_SERVICE_NAME" --region "$REGION" --format='value(status.url)')"
    check_api "$API_URL"
  fi
fi

if [[ "$WEB_NEEDED" -eq 1 ]]; then
  echo "[deploy] Web"
  ALLOW_UNAUTHENTICATED=1 "$ROOT_DIR/scripts/deploy_cloud_run_web.sh"
  if [[ "$SKIP_CHECKS" -eq 0 ]]; then
    WEB_URL="$(gcloud run services describe "$WEB_SERVICE_NAME" --region "$REGION" --format='value(status.url)')"
    check_web "$WEB_URL"
  fi
fi

echo "Smart deploy finished."
