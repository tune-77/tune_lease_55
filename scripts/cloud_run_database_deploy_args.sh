#!/usr/bin/env bash

# Cloud Run の DB 接続引数を一か所で組み立てる。
# DATABASE_URL が Cloud SQL の Unix socket を指す場合、インスタンス接続を
# 同時に付けないと API は起動しても全DB集計が空になるため、必ず対で扱う。
configure_cloud_run_database_deploy_args() {
  if [[ "${CLOUDRUN_DATA_MODE}" == "demo" ]]; then
    echo "Demo mode: DATABASE_URL/Cloud SQL is intentionally not attached."
    deploy_args+=(--remove-secrets=DATABASE_URL --clear-cloudsql-instances)
    return 0
  fi

  if [[ -z "${CLOUDSQL_INSTANCE}" ]]; then
    echo "Production SQLite/GCS mode: CLOUDSQL_INSTANCE is unset; DATABASE_URL/Cloud SQL is detached."
    deploy_args+=(--remove-secrets=DATABASE_URL --clear-cloudsql-instances)
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
