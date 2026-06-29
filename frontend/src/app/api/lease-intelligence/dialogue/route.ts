const FASTAPI_URL = process.env.FASTAPI_URL || "http://127.0.0.1:8000";
const DIALOGUE_TIMEOUT_MS = 180_000;

async function proxyToFastApi(request: Request, method: "POST" | "DELETE") {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DIALOGUE_TIMEOUT_MS);

  try {
    const body = method === "POST" ? await request.text() : undefined;
    const upstream = await fetch(`${FASTAPI_URL}/api/lease-intelligence/dialogue`, {
      method,
      headers: method === "POST" ? { "Content-Type": "application/json" } : undefined,
      body,
      signal: controller.signal,
      cache: "no-store",
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") || "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const timedOut = error instanceof Error && error.name === "AbortError";
    return Response.json(
      {
        detail: timedOut
          ? "対話AIの応答がタイムアウトしました。入力を少し短くするか、時間を置いて再送してください。"
          : `FastAPI対話エンドポイントへ接続できません: ${message}`,
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}

export async function POST(request: Request) {
  return proxyToFastApi(request, "POST");
}

export async function DELETE(request: Request) {
  return proxyToFastApi(request, "DELETE");
}
