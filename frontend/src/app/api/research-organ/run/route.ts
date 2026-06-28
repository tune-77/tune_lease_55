import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = process.env.FASTAPI_URL || "http://127.0.0.1:8000";
const RESEARCH_TIMEOUT_MS = 180_000;

export const runtime = "nodejs";
export const maxDuration = 180;

export async function POST(request: NextRequest) {
  let body: string;
  try {
    body = await request.text();
  } catch {
    return NextResponse.json({ detail: "調査リクエストの読み取りに失敗しました。" }, { status: 400 });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), RESEARCH_TIMEOUT_MS);

  try {
    const upstream = await fetch(`${FASTAPI_URL}/api/research-organ/run`, {
      method: "POST",
      headers: {
        "content-type": request.headers.get("content-type") || "application/json",
      },
      body,
      signal: controller.signal,
      cache: "no-store",
    });
    const text = await upstream.text();
    return new NextResponse(text || null, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json",
        "cache-control": "no-store",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const detail =
      message === "This operation was aborted"
        ? "外部調査器官の実行がタイムアウトしました。テーマを短くするか、保存なし確認から再実行してください。"
        : `外部調査器官APIへの接続に失敗しました: ${message}`;
    return NextResponse.json({ detail }, { status: 504 });
  } finally {
    clearTimeout(timeout);
  }
}
