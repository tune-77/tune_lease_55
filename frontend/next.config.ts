import type { NextConfig } from "next";
// @ts-ignore — next-pwa v5 has no official TS types
const withPWA = require("next-pwa");

const nextConfig: NextConfig = {
  /* config options here */
};

export default withPWA({
  dest: "public",
  disable: process.env.NODE_ENV === "development",
  register: true,
  skipWaiting: true,
})(nextConfig);
