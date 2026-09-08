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

## Revision cleanup

Each deploy creates a new Cloud Run revision; old revisions with 0% traffic
accumulate over time. `scripts/cleanup_cloud_run_revisions.py` lists and
optionally deletes old, unreferenced revisions for `tune-lease-55-api` and
`tune-lease-55-web`.

1. Install the gcloud CLI: https://cloud.google.com/sdk/docs/install
2. Authenticate:
   ```bash
   gcloud auth activate-service-account --key-file=/path/to/key.json
   # or Application Default Credentials:
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```
3. Review the plan, then apply:
   ```bash
   python3 scripts/cleanup_cloud_run_revisions.py            # dry run: prints what would be deleted
   python3 scripts/cleanup_cloud_run_revisions.py --apply    # deletes
   ```

Revisions currently referenced by the service's traffic spec (serving
traffic or tagged at 0%) are never deleted. Among the rest, the most recent
`--keep` revisions per service (default 5) are kept for rollback headroom;
only older, untagged, 0%-traffic revisions beyond that are removed.

## Artifact Registry cleanup

Cloud Run revision cleanup does not delete container images. The shared
`cloud-run-source-deploy` repository therefore uses
`config/artifact_registry_cleanup_policy.json` to delete versions older than
one day while retaining the 15 newest Artifact Registry versions for each API
and Web package. A Docker push currently produces approximately three registry
versions (the runnable image plus index/metadata records), so this preserves
roughly five deployable build generations for rollback.

Apply the policy in dry-run mode first, then enable deletion after reviewing
the repository details:

```bash
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
  --project gen-lang-client-0420497423 \
  --location asia-northeast1 \
  --policy config/artifact_registry_cleanup_policy.json \
  --dry-run

gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
  --project gen-lang-client-0420497423 \
  --location asia-northeast1 \
  --policy config/artifact_registry_cleanup_policy.json \
  --no-dry-run
```

Cleanup runs asynchronously and can take approximately one day. The policy
also lets obsolete combined/local-demo packages expire because only the active
API and Web package prefixes receive the 15-version keep rule.

## セキュリティ: アクセス制御（重要）

実データ用Webは `--no-allow-unauthenticated --invoker-iam-check` でデプロイします。
WebがAPIキーを自動付与する前にCloud Run IAMで利用者を認証します。
Webの公開可否は接続先APIの `CLOUDRUN_DATA_MODE` から決め、未設定は実データ扱いです。
APIが明示的に `demo` の場合だけWebを公開します。

API単体は引き続きアプリ層の共有キーで保護します。Web→APIの経路は従来どおりです。
実データ用ではAPI・Web・一体型のいずれも `API_ACCESS_KEY` がSecret Managerに
存在しなければデプロイを中止します。一体型も実データ用ではIAMで入口を保護し、
デモだけキー未設定を許可します。

### 1. 実データ用Webへのアクセス

利用するGoogleアカウントに対象Webサービスの `roles/run.invoker` が必要です。
認証済みのローカルプロキシを通してブラウザから利用できます。

```bash
gcloud run services proxy tune-lease-55-web --region asia-northeast1 --port 8080
# ブラウザで http://localhost:8080 を開く
```

デプロイ後、実データ用Webの直接URLには匿名アクセスできなくなります。
通常のブラウザログインを導入する場合はIAP等を別途構成してください。
共有キー方式は利用者別の権限分離ではありません。IAMで許可された利用者は
既存の業務操作を利用できます。プロジェクト全体などで広いInvoker権限を付与して
いないことも運用側で確認してください。

参考: [Cloud Runの公開アクセスとIAMチェック](https://docs.cloud.google.com/run/docs/authenticating/public)

### 2. 共有シークレット（既定の保護方式）

同じ値の `API_ACCESS_KEY` を **API サービスと Web サービスの両方**に設定します。

- **API 側**: FastAPI の `ApiKeyAuthMiddleware`（`api/api_key_auth.py`）が有効化され、
  `/api/*` へのリクエストに一致する `X-API-Key`（または `Authorization: Bearer <key>`）を
  要求します（`/`, `/health`, `/healthz`, `/docs` は免除）。Cloud Run の公開ヘルスチェックは
  予約パスとの衝突を避けるため `/health` を使います。未設定時は無効なので、ローカル開発・
  テスト・既存構成は一切壊れません。`REQUIRE_API_ACCESS_KEY`（既定: demoモードは`0`、
  非demoモードは`1`）が`1`かつキー未設定の場合、`/api/*` は503を返します。
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

デプロイスクリプトは Secret Manager の `API_ACCESS_KEY` シークレットを両サービスへ
自動配線します。事前に一度だけ登録してください。

```bash
gcloud secrets create API_ACCESS_KEY \
  --replication-policy=automatic \
  --project gen-lang-client-0420497423

openssl rand -hex 32 | gcloud secrets versions add API_ACCESS_KEY \
  --data-file=- \
  --project gen-lang-client-0420497423

# 登録後は通常通りデプロイするだけでよい
./scripts/deploy_cloud_run_api.sh   # 非demoモードでは未登録だと exit 1 で止まる
./scripts/deploy_cloud_run_web.sh
```

`CLOUDRUN_DATA_MODE=demo` を明示すれば、`API_ACCESS_KEY` 未登録でも従来通り
無防備な公開デモとしてデプロイできます（意図的な選択のみ許可）。

### 公開デモの削除保護（DEMO_READONLY）

ハッカソン等で URL を不特定多数に見せる場合、来場者が
`DELETE /api/cases/operation/clear-all`（全案件削除）等でデモデータを破壊できないよう、
環境変数 `DEMO_READONLY=1` で `/api/*` への **DELETE を 403 で拒否**します
（`api/demo_guard.py`）。スコアリング・チャット・討論・案件登録などの試用は許可し、
削除操作のみ塞ぎます。

APIデプロイの既定は実データモード、`DEMO_READONLY=0` です。
公開デモの削除を保護するときは両方を明示してください。

```bash
# 公開デモ（削除保護ON）
CLOUDRUN_DATA_MODE=demo DEMO_READONLY=1 ./scripts/deploy_cloud_run_api.sh
# 削除も許可する公開デモ
CLOUDRUN_DATA_MODE=demo DEMO_READONLY=0 ./scripts/deploy_cloud_run_api.sh
```

UI 側の削除ボタンは押下時に 403 となりエラー表示されます（データは保全）。デモ体験を
更に磨くならボタン自体の非表示は任意の追加対応です。

### DB復元と起動確認

実データ用では、一体型・API単体とも同梱DBを配置する前にGCSスナップショットを復元します。
既存バケットにスナップショットが存在しない初回だけ、同梱DBで初期化します。
通信障害・権限エラー・バケット欠落・依存ライブラリ不足では起動を中止します。
古い同梱DBが後続の定期保存で最新スナップショットを上書きすることを防ぎます。
復旧後に起動を再試行してください。

一体型の起動確認は `/api/health/auth` にAPIキーを付けて行います。
認証対象外の `/docs` だけで起動成功と判定しません。

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
