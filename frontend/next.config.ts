import type { NextConfig } from "next";
// @ts-expect-error next-pwa v5 has no official TS types.
import withPWAInit from "next-pwa";

const API_URL = process.env.FASTAPI_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
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
  disable: process.env.NODE_ENV === "development",
  register: true,
  skipWaiting: true,
});

export default withPWA(nextConfig);
