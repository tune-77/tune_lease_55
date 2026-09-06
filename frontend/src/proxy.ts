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
// PUBLIC_TUNNEL=1 の外部リクエストは、利用者が明示設定した専用パスワードで
// Web境界も保護する。ローカルhostへの直接アクセスは開発用として対象外。
// API_ACCESS_KEY 未設定時はAPIヘッダを変更しない（ローカル開発時に無影響）。
// 検証側は api/api_key_auth.py（ApiKeyAuthMiddleware）。

const constantTimeEqual = (left: string, right: string) => {
  const length = Math.max(left.length, right.length);
  let difference = left.length ^ right.length;
  for (let index = 0; index < length; index += 1) {
    difference |= (left.charCodeAt(index) || 0) ^ (right.charCodeAt(index) || 0);
  }
  return difference === 0;
};

const hasValidTunnelCredentials = (request: NextRequest, password: string) => {
  const authorization = request.headers.get("authorization") || "";
  if (!authorization.startsWith("Basic ")) return false;
  try {
    const decoded = atob(authorization.slice(6));
    const separator = decoded.indexOf(":");
    if (separator < 0) return false;
    return constantTimeEqual(decoded.slice(0, separator), "lease")
      && constantTimeEqual(decoded.slice(separator + 1), password);
  } catch {
    return false;
  }
};

export function proxy(request: NextRequest) {
  const key = process.env.API_ACCESS_KEY;
  const tunnelPassword = process.env.PUBLIC_TUNNEL_AUTH;
  const hostname = request.nextUrl.hostname;
  const isLocalHost = hostname === "localhost" || hostname === "127.0.0.1";
  const isTunnelRequest = process.env.PUBLIC_TUNNEL === "1"
    && (request.headers.has("cf-connecting-ip") || !isLocalHost);
  if (isTunnelRequest && (!tunnelPassword || !hasValidTunnelCredentials(request, tunnelPassword))) {
    return new NextResponse("Authentication required", {
      status: 401,
      headers: { "WWW-Authenticate": 'Basic realm="Tune Lease 55"' },
    });
  }
  if (!request.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.next();
  }
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
  matcher: "/:path*",
};
