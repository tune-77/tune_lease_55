#!/usr/bin/env bash

# Cloud Run の DB 接続引数を一か所で組み立てる。
# DATABASE_URL が Cloud SQL の Unix socket を指す場合、インスタンス接続を
# 同時に付けないと API は起動しても全DB集計が空になるため、必ず対で扱う。
configure_cloud_run_database_deploy_args() {
  if [[ "${CLOUDRUN_DATA_MODE}" == "demo" ]]; then
    echo "Demo mode: DATABASE_URL/Cloud SQL is intentionally not attached."
    remove_stale_cloud_run_database_secret
    deploy_args+=(--clear-cloudsql-instances)
    return 0
  fi

  if [[ -z "${CLOUDSQL_INSTANCE}" ]]; then
    echo "Production SQLite/GCS mode: CLOUDSQL_INSTANCE is unset; DATABASE_URL/Cloud SQL is detached."
    remove_stale_cloud_run_database_secret
    deploy_args+=(--clear-cloudsql-instances)
    return 0
  fi

  if ! gcloud secrets describe "${DATABASE_SECRET_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    echo "ERROR: Cloud SQL instance ${CLOUDSQL_INSTANCE} was requested, but Secret Manager secret ${DATABASE_SECRET_NAME} was not found." >&2
    return 1
  fi

  deploy_args+=(
    --set-secrets "DATABASE_URL=${DATABASE_SECRET_NAME}:latest"
    --add-cloudsql-instances "${CLOUDSQL_INSTANCE}"
  )
}

# gcloud run deploy は --set-secrets と --remove-secrets を同時指定できない。
# DATABASE_URL が実際に残っている時だけ先に除去し、通常のデプロイでは
# --set-secrets と競合しない --clear-cloudsql-instances だけを渡す。
remove_stale_cloud_run_database_secret() {
  local current_env_names
  current_env_names="$(
    gcloud run services describe "${SERVICE_NAME}" \
      --project "${PROJECT_ID}" \
      --region "${REGION}" \
      --format='value(spec.template.spec.containers[0].env[].name)' \
      2>/dev/null || true
  )"
  if [[ "${current_env_names}" != *"DATABASE_URL"* ]]; then
    return 0
  fi

  echo "Removing stale DATABASE_URL binding before deploy."
  gcloud run services update "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --remove-secrets=DATABASE_URL \
    --clear-cloudsql-instances \
    --quiet
}
