import axios from "axios";

// Next.js rewrites が /api/* → FastAPI へプロキシするため、同一オリジンへ向ける
export const API_BASE = "";

export const apiClient = axios.create({ baseURL: API_BASE });
