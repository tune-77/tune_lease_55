import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Next.js 16 の proxy 規約（旧 middleware）。
// /api/* リクエストに server-only の共有シークレット X-API-Key を注入し、
// next.config.ts の rewrites が FastAPI へ転送する際に一緒に届ける。
// rewrites 自体はヘッダを付与できないためこの proxy で足す（proxy が設定した
// request header は rewrite destination へ転送される。Next docs: proxy.md「Setting Headers」）。
// これにより SSE ストリーミング・multipart アップロードは従来どおり rewrites が
// 透過処理し、認証ヘッダだけが追加される。
//
// API_ACCESS_KEY 未設定時は何も変更しない（ローカル開発・API 認証無効時に無影響）。
// 検証側は api/api_key_auth.py（ApiKeyAuthMiddleware）。

export function proxy(request: NextRequest) {
  const key = process.env.API_ACCESS_KEY;
  if (!key) {
    return NextResponse.next();
  }
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-api-key", key);
  return NextResponse.next({
    request: { headers: requestHeaders },
  });
}

export const config = {
  matcher: "/api/:path*",
};
