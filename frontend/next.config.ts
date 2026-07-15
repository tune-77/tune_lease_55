import type { NextConfig } from "next";
// @ts-expect-error next-pwa v5 has no official TS types.
import withPWAInit from "next-pwa";

const API_URL = process.env.FASTAPI_URL || "http://127.0.0.1:8000";
const BROWSER_FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_BASE_URL || "";
const connectSrc = ["'self'", "http://127.0.0.1:8000", "http://localhost:8000", BROWSER_FASTAPI_URL]
  .filter(Boolean)
  .join(" ");

const securityHeaders = [
  { key: "X-DNS-Prefetch-Control", value: "on" },
  { key: "X-Frame-Options", value: "SAMEORIGIN" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  {
    key: "Strict-Transport-Security",
    value: "max-age=63072000; includeSubDomains; preload",
  },
  {
    key: "Content-Security-Policy",
    value: [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self'",
      `connect-src ${connectSrc}`,
      "frame-ancestors 'none'",
    ].join("; "),
  },
];

const nextConfig: NextConfig = {
  output: "standalone",
  allowedDevOrigins: ["*.trycloudflare.com"],
  experimental: {
    // rewrites 経由の FastAPI 呼び出しはデフォルト30秒（30000ms）で切断される。
    // マルチエージェント討論審査（/api/multi-agent-screening）は30〜90秒かかるため延長する。
    proxyTimeout: 300_000,
  },
  async headers() {
    return [
      {
        source: "/multi-shion-demo",
        headers: [
          ...securityHeaders,
          { key: "Cache-Control", value: "no-store, max-age=0" },
        ],
      },
      {
        source: "/screening",
        headers: [
          ...securityHeaders,
          { key: "Cache-Control", value: "no-store, max-age=0, must-revalidate" },
        ],
      },
      { source: "/(.*)", headers: securityHeaders },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/api/:path*`,
      },
    ];
  },
};

const withPWA = withPWAInit({
  dest: "public",
  disable: true,
  register: false,
  skipWaiting: true,
});

export default withPWA(nextConfig);
