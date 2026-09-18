import axios from "axios";

// 既定は同一オリジン（Next.js rewrites 経由）。
// NEXT_PUBLIC_FASTAPI_BASE_URL にブラウザから到達可能な FastAPI URL を設定した場合だけ、
// Next の proxy を介さず FastAPI を直接呼ぶ。
const configuredApiBase = (process.env.NEXT_PUBLIC_FASTAPI_BASE_URL || "").replace(/\/$/, "");

export const API_BASE = configuredApiBase;

export const apiClient = axios.create({ baseURL: API_BASE });

export const getApiErrorDetail = (error: unknown, fallback: string): string => {
  if (typeof error !== "object" || error === null) return fallback;
  const candidate = error as {
    message?: unknown;
    response?: { data?: { detail?: unknown } };
  };
  const detail = candidate.response?.data?.detail;
  if (Array.isArray(detail)) return JSON.stringify(detail);
  if (typeof detail === "string" && detail) return detail;
  return typeof candidate.message === "string" && candidate.message
    ? candidate.message
    : fallback;
};
