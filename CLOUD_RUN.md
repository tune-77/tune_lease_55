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
./scripts/deploy_cloud_run_api.sh
./scripts/deploy_cloud_run_web.sh
```

The legacy wrapper still exists:

```bash
./scripts/deploy_cloud_run.sh
```

Use the API script when only backend logic changed. Use the Web script when
only frontend/Next.js changed. Run the wrapper only when both changed.

## セキュリティ: アクセス制御（重要）

**デプロイの既定値は認証必須（`ALLOW_UNAUTHENTICATED=0`）です。** これは安全側の
デフォルトで、`--no-allow-unauthenticated` としてデプロイされます。以前の手順は
既定で無認証公開であり、FastAPI 側に認証が無いため、**API の URL を知る誰でも
`DELETE /api/cases/operation/clear-all`（全案件削除）や審査データ・会話履歴の
読み取り、サーバー側 Gemini キーでの LLM 実行が可能な状態でした。**

FastAPI は175の `/api/*` エンドポイントを持ち、それ自体はアプリ層の認証を持ちません。
以下のいずれか（できれば両方）で保護してください。

### 1. Cloud Run IAM（推奨・主防御）

API サービスを非公開のまま維持し、Web サービスのサービスアカウントにのみ
`roles/run.invoker` を付与する。Web → API 呼び出しには ID トークンが必要になるため、
**Next.js の `rewrites` プロキシ（`next.config.ts`）を Route Handler 化して
`Authorization: Bearer <ID token>` を server-side で付与する必要があります**
（`rewrites` はヘッダを追加できません）。

```bash
# API は非公開（既定）
./scripts/deploy_cloud_run_api.sh
# Web だけ公開する場合
ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run_web.sh
```

### 2. 共有シークレット（多層防御 / IAM が難しい場合）

同じ値の `API_ACCESS_KEY` を **API サービスと Web サービスの両方**に設定します。

- **API 側**: FastAPI の `ApiKeyAuthMiddleware`（`api/api_key_auth.py`）が有効化され、
  `/api/*` へのリクエストに一致する `X-API-Key`（または `Authorization: Bearer <key>`）を
  要求します（`/`, `/healthz`, `/docs` は免除）。未設定時は無効なので、ローカル開発・
  テスト・既存構成は一切壊れません。
- **Web 側**: `frontend/src/proxy.ts`（Next.js 16 の proxy 規約）が `/api/*` に
  `X-API-Key` を server-side で自動注入し、`next.config.ts` の `rewrites` が
  FastAPI へ転送します（`rewrites` はヘッダを付与できないため proxy で足す。
  proxy が設定した request header は rewrite destination へ届く）。SSE
  ストリーミング・OCR の multipart アップロードは従来どおり `rewrites` が透過
  処理するため影響しません。個別 Route Handler
  （`api/lease-intelligence/dialogue`, `api/research-organ/run`）は自前 fetch のため
  `internalApiAuthHeaders()`（`frontend/src/lib/apiAuth.ts`）でキーを注入します。

**キーはブラウザへ露出させないこと**: `API_ACCESS_KEY` は server-only 環境変数として
設定し、`NEXT_PUBLIC_` を付けないでください（付けるとバンドルに焼き込まれ意味を失う）。

```bash
# 例: API と Web の両サービスに同じキーを設定して公開する場合
API_KEY="$(openssl rand -hex 32)"
API_ACCESS_KEY="$API_KEY" ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run_api.sh
API_ACCESS_KEY="$API_KEY" ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run_web.sh
```

いずれの手段も未適用のまま `ALLOW_UNAUTHENTICATED=1` で公開すると、上記の
無防備な状態に戻ります。公開が本当に必要な場合のみ明示的に指定してください。

### 残課題: Web フロントエンド自体の保護（別タスク）

上記はいずれも「API の直叩き」を塞ぐものです。Web サービスを
`ALLOW_UNAUTHENTICATED=1` で公開している限り、**公開 URL を知る第三者は Web UI を
開いて proxy 経由で API を操作できます**（キーは server-side 注入されるため UI からは
操作可能）。1 ユーザー運用でフロント自体も秘匿するには、次のいずれかを別途導入します:

- Next.js にログイン（認証必須ページ）を追加する
- Web サービスも `--no-allow-unauthenticated` にし、`gcloud run services proxy` や
  IAP 経由でアクセスする

この対応は本 PR のスコープ外です（`WHY_USER_COUNT_MATTERS.md` /
`IMPLEMENTATION_DECISION_FOR_1USER.md` の方針と併せて別途計画）。

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
