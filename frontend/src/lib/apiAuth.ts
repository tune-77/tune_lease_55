// サーバー専用: FastAPI への内部呼び出しに付与する共有シークレットヘッダ。
// API_ACCESS_KEY は Web サービスの server-only 環境変数（NEXT_PUBLIC_ を付けない＝
// ブラウザバンドルへ露出させない）。未設定時は空オブジェクトを返すため、
// ローカル開発・API 認証無効時は一切影響しない。
// 対応する検証側は api/api_key_auth.py（ApiKeyAuthMiddleware）。
export function internalApiAuthHeaders(): Record<string, string> {
  const key = process.env.API_ACCESS_KEY;
  return key ? { "x-api-key": key } : {};
}
