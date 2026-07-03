import axios from "axios";

// 既定は同一オリジン（Next.js rewrites 経由）。
// NEXT_PUBLIC_FASTAPI_BASE_URL にブラウザから到達可能な FastAPI URL を設定した場合だけ、
// Next の proxy を介さず FastAPI を直接呼ぶ。
export const API_BASE = (process.env.NEXT_PUBLIC_FASTAPI_BASE_URL || "").replace(/\/$/, "");

export const apiClient = axios.create({ baseURL: API_BASE });
