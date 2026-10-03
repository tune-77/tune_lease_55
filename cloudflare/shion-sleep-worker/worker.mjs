// shion.tune77.com/* に載せる Worker。Mac（オリジン）に届かない時だけ「お休み中」ページを返し、
// それ以外は素通しする。本文・Cookie・ヘッダはどこにも記録しない（console.* を使わない）。
// 配備は scripts/cloudflare_edge_setup.sh（Cloudflare API 経由）。

// cloudflared 停止中の 530/1033、cloudflared は生きているが Next.js が落ちている 502 など、
// Cloudflare 側が生成するオリジン不達系のステータス。アプリ自身の 500/503 は素通しする。
const OFFLINE_STATUSES = new Set([502, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530]);

// 休止ページの表示確認用（Access を通過したリクエストだけが届く）。
export const SIMULATE_HEADER = "x-shion-sleep-test";

const HTML = `<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>紫苑はお休み中です</title>
<style>
  body { margin: 0; min-height: 100vh; display: grid; place-items: center;
         font-family: -apple-system, "Hiragino Sans", sans-serif; background: #f6f4fb; color: #3b3450; }
  main { padding: 2rem 1.5rem; max-width: 26rem; text-align: center; }
  h1 { font-size: 1.4rem; margin: 0 0 1rem; }
  p { line-height: 1.7; margin: 0 0 .75rem; }
  small { color: #7a7390; }
</style>
</head>
<body>
<main>
  <h1>紫苑は今お休み中です</h1>
  <p>サーバー（Mac）が停止しているか、接続が切れています。</p>
  <p>しばらくしてから、もう一度開いてください。</p>
  <small>shion.tune77.com</small>
</main>
</body>
</html>`;

export function sleepResponse(request) {
  const headers = {
    "cache-control": "no-store",
    "retry-after": "300",
    "x-shion-offline": "1",
  };
  const accept = request.headers.get("accept") || "";
  if (!accept.includes("text/html")) {
    // API 呼び出し（fetch/axios）には JSON で返し、画面側のエラー処理に任せる。
    return new Response(
      JSON.stringify({ error: "shion_offline", message: "紫苑は今お休み中です" }),
      { status: 503, headers: { ...headers, "content-type": "application/json; charset=utf-8" } },
    );
  }
  return new Response(HTML, {
    status: 503,
    headers: { ...headers, "content-type": "text/html; charset=utf-8" },
  });
}

export default {
  async fetch(request) {
    if (request.headers.get(SIMULATE_HEADER) === "1") {
      return sleepResponse(request);
    }
    let response;
    try {
      response = await fetch(request);
    } catch {
      return sleepResponse(request);
    }
    if (OFFLINE_STATUSES.has(response.status)) {
      return sleepResponse(request);
    }
    return response;
  },
};
