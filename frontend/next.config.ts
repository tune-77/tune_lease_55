import type { NextConfig } from "next";
// @ts-ignore — next-pwa v5 has no official TS types
const withPWA = require("next-pwa");

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

export default withPWA({
  dest: "public",
  disable: process.env.NODE_ENV === "development",
  register: true,
  skipWaiting: true,
})(nextConfig);
