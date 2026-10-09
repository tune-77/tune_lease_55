import axios from "axios";

// 既定は同一オリジン（Next.js rewrites 経由）。
// NEXT_PUBLIC_FASTAPI_BASE_URL にブラウザから到達可能な FastAPI URL を設定した場合だけ、
// Next の proxy を介さず FastAPI を直接呼ぶ。
const configuredApiBase = (process.env.NEXT_PUBLIC_FASTAPI_BASE_URL || "").replace(/\/$/, "");

export const API_BASE = configuredApiBase;

export const apiClient = axios.create({ baseURL: API_BASE });

// REV-591: ブラウザから検証する時は localStorage に shion_verification=1 を置くと、
// 会話に検証の印が付き、紫苑の記憶・内省の材料にならない（本人の通常利用では何も付かない）。
apiClient.interceptors.request.use((config) => {
  try {
    if (typeof window !== "undefined" && window.localStorage.getItem("shion_verification") === "1") {
      config.headers.set("X-Shion-Verification", "1");
    }
  } catch {
    // localStorage が使えない環境では印を付けない
  }
  return config;
});

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
