# Cloud Run deployment

## Recommended initial settings

- Region: `asia-northeast1`
- Container port: `8080`
- Memory: `4Gi`
- CPU: `2`
- Request timeout: `900` seconds
- Concurrency: `1`
- Minimum instances: `0`
- Maximum instances: `1`
- Execution environment: second generation

Concurrency and maximum instances are intentionally limited because the current
scoring engine uses shared process state, SQLite, and a physical result bridge.
Increase these only after moving state to external services.

## Secrets

Never put real values in `.env`, source files, Docker build arguments, or
Cloud Build substitutions. Store them in Secret Manager.

```bash
gcloud secrets create GEMINI_API_KEY \
  --replication-policy=automatic \
  --project gen-lang-client-0420497423

read -rs GEMINI_VALUE
printf '%s' "$GEMINI_VALUE" | gcloud secrets versions add GEMINI_API_KEY \
  --data-file=- \
  --project gen-lang-client-0420497423
unset GEMINI_VALUE
```

Grant the Cloud Run runtime service account
`roles/secretmanager.secretAccessor` for only the secrets it needs.

The Gemini REST calls use the `x-goog-api-key` header so API keys are not
included in HTTP URLs or normal request logs.

## Deploy

The script uses the currently configured Google Cloud project by default.
Deployment is private unless `ALLOW_UNAUTHENTICATED=1` is explicitly set.
Before deploying, it packages the current SQLite snapshots and selected Obsidian
notes into `.cloudrun_bundle/` and bakes that bundle into the container image.

```bash
PROJECT_ID=gen-lang-client-0420497423 \
REGION=asia-northeast1 \
SERVICE_NAME=tune-lease-55 \
./scripts/deploy_cloud_run.sh
```

For a public service:

```bash
ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run.sh
```

## Important persistence limitation

Cloud Run's writable container filesystem is temporary and consumes instance
memory. The current deployment uses a bundled snapshot of the SQLite database
and selected Obsidian folders, then copies that snapshot into the runtime
filesystem on boot. It is still ephemeral and will be lost when the instance
restarts.

Before production use:

1. Treat the bundled SQLite/Obsidian content as read-only master data.
2. Move case and screening data from SQLite to Cloud SQL or another managed DB
   if you need durable edits.
3. Store generated files and news notes in Cloud Storage or a database if they
   must survive restarts.
4. Keep maximum instances at `1` until the physical scoring bridge and shared
   session state are removed.

## Verification

```bash
gcloud run services describe tune-lease-55 \
  --region asia-northeast1 \
  --project gen-lang-client-0420497423

gcloud run services logs read tune-lease-55 \
  --region asia-northeast1 \
  --project gen-lang-client-0420497423 \
  --limit 100
```
