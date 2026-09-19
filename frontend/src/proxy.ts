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

const TUNNEL_SESSION_COOKIE = "tune_lease_session";
const TUNNEL_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30;

const tunnelSessionToken = async (password: string) => {
  const payload = new TextEncoder().encode(`tune-lease-session-v1:${password}`);
  const digest = await crypto.subtle.digest("SHA-256", payload);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
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

const hasValidSyncProbeToken = (request: NextRequest, token: string) => {
  const provided = request.headers.get("x-sync-probe-key") || "";
  return provided.length > 0 && constantTimeEqual(provided, token);
};

const hasValidDashboardHealthProbeToken = (request: NextRequest, token: string) => {
  const provided = request.headers.get("x-dashboard-health-probe-key") || "";
  return provided.length > 0 && constantTimeEqual(provided, token);
};

export async function proxy(request: NextRequest) {
  const key = process.env.API_ACCESS_KEY;
  const tunnelPassword = process.env.PUBLIC_TUNNEL_AUTH;
  const syncProbeToken = process.env.KNOWLEDGE_SYNC_PROBE_TOKEN;
  const dashboardHealthProbeToken = process.env.DASHBOARD_HEALTH_PROBE_TOKEN;
  const hostname = request.nextUrl.hostname;
  const isLocalHost = hostname === "localhost" || hostname === "127.0.0.1";
  // GitHub Actions runs one read-only sync-health probe on a schedule. It used
  // to skip Basic auth entirely for this path, but this proxy still attaches
  // the privileged x-api-key afterwards, so an external caller that just kept
  // requesting the path anonymously could reach FastAPI (Vault scan + Chroma
  // count) for free and keep the scale-to-zero, concurrency=1 API billable or
  // monopolized. It now needs its own dedicated secret (X-Sync-Probe-Key)
  // instead of the browser Basic-auth password, so the probe stays
  // authenticated without depending on copying that password into GitHub
  // Secrets. The full cloud-status response still requires Basic auth.
  const isKnowledgeSyncProbe = request.nextUrl.pathname
    === "/api/system/knowledge-sync-health";
  // Same reasoning applies to the hourly dashboard-data-health probe: it used
  // to send no credentials at all, so once PUBLIC_TUNNEL_AUTH started gating
  // every /api/* path (2026-09-08) it began failing with 401 on every run.
  // It gets its own dedicated secret (X-Dashboard-Health-Probe-Key) rather
  // than the browser Basic-auth password.
  const isDashboardHealthProbe = request.nextUrl.pathname
    === "/api/dashboard/data-health";
  const isTunnelRequest = process.env.PUBLIC_TUNNEL === "1"
    && (request.headers.has("cf-connecting-ip") || !isLocalHost);
  let shouldCreateTunnelSession = false;
  if (isTunnelRequest) {
    let isAuthorized = false;
    if (isKnowledgeSyncProbe) {
      isAuthorized = !!syncProbeToken
        && hasValidSyncProbeToken(request, syncProbeToken);
    } else if (isDashboardHealthProbe) {
      // 機械アクセス想定のprobeなのでセッションCookieは発行しない。
      isAuthorized = !!dashboardHealthProbeToken
        && hasValidDashboardHealthProbeToken(request, dashboardHealthProbeToken);
    } else if (tunnelPassword) {
      const expectedSession = await tunnelSessionToken(tunnelPassword);
      const providedSession = request.cookies.get(TUNNEL_SESSION_COOKIE)?.value || "";
      const hasValidSession = providedSession.length > 0
        && constantTimeEqual(providedSession, expectedSession);
      const hasValidBasicAuth = hasValidTunnelCredentials(request, tunnelPassword);
      isAuthorized = hasValidSession || hasValidBasicAuth;
      // Basic認証のみ成功した初回だけCookieを発行し、以降はセッションで通す。
      shouldCreateTunnelSession = hasValidBasicAuth && !hasValidSession;
    }
    if (!isAuthorized) {
      return new NextResponse("Authentication required", {
        status: 401,
        headers: { "WWW-Authenticate": 'Basic realm="Tune Lease 55"' },
      });
    }
  }
  let response: NextResponse;
  if (request.nextUrl.pathname.startsWith("/api/") && key) {
    const requestHeaders = new Headers(request.headers);
    requestHeaders.set("x-api-key", key);
    response = NextResponse.next({
      request: { headers: requestHeaders },
    });
  } else {
    response = NextResponse.next();
  }
  if (shouldCreateTunnelSession && tunnelPassword) {
    response.cookies.set({
      name: TUNNEL_SESSION_COOKIE,
      value: await tunnelSessionToken(tunnelPassword),
      httpOnly: true,
      secure: true,
      sameSite: "strict",
      path: "/",
      maxAge: TUNNEL_SESSION_MAX_AGE_SECONDS,
    });
  }
  return response;
}

export const config = {
  matcher: "/:path*",
};
