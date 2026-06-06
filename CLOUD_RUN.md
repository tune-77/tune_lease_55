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

The current setup is split into two Cloud Run services:

- API service: `tune-lease-55-api`
- Web service: `tune-lease-55-web`

Role split:

- API service handles FastAPI, SQLite, Obsidian-backed analysis, and all `/api/*`
  requests.
- Web service handles Next.js pages and UI, and proxies `/api/*` to the API
  service at build time.

Before deploying the API, it packages the current SQLite snapshots and selected
Obsidian notes into `.cloudrun_bundle/` and bakes that bundle into the API
container image.

```bash
ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run_api.sh
ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run_web.sh
```

The legacy wrapper still exists:

```bash
ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run.sh
```

Use the API script when only backend logic changed. Use the Web script when
only frontend/Next.js changed. Run the wrapper only when both changed.

Update rule:

- If you touched `api/`, `data/`, `runtime_paths.py`, or bundle logic, deploy
  the API service.
- If you touched `frontend/src/` or `frontend/next.config.ts`, deploy the Web
  service.

## Important persistence limitation

Cloud Run's writable container filesystem is temporary and consumes instance
memory. The API deployment uses a bundled snapshot of the SQLite database and
selected Obsidian folders, then copies that snapshot into the runtime
filesystem on boot. It is still ephemeral and will be lost when the instance
restarts.

Before production use:

1. Treat the bundled SQLite/Obsidian content as read-only master data.
2. Move case and screening data from SQLite to Cloud SQL or another managed DB
   if you need durable edits.
3. Store generated files and news notes in Cloud Storage or a database if they
   must survive restarts.
4. Keep maximum instances at `1` for the API service until the physical
   scoring bridge and shared session state are removed.
5. The Web service can scale separately from the API service.

## Verification

```bash
gcloud run services describe tune-lease-55-api \
  --region asia-northeast1 \
  --project gen-lang-client-0420497423

gcloud run services logs read tune-lease-55-api \
  --region asia-northeast1 \
  --project gen-lang-client-0420497423 \
  --limit 100
```
