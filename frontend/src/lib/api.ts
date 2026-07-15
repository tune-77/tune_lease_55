import axios from "axios";

// 既定は同一オリジン（Next.js rewrites 経由）。
// NEXT_PUBLIC_FASTAPI_BASE_URL にブラウザから到達可能な FastAPI URL を設定した場合だけ、
// Next の proxy を介さず FastAPI を直接呼ぶ。
const configuredApiBase = (process.env.NEXT_PUBLIC_FASTAPI_BASE_URL || "").replace(/\/$/, "");

const localFastApiBase = () => {
  if (typeof window === "undefined") return "";
  const host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1") {
    return "http://127.0.0.1:8000";
  }
  return "";
};

export const API_BASE = configuredApiBase || localFastApiBase();

export const apiClient = axios.create({ baseURL: API_BASE });
