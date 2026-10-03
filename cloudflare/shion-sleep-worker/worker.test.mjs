import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import worker, { SIMULATE_HEADER } from "./worker.mjs";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function page(headers = {}) {
  return new Request("https://shion.tune77.com/chat", { headers: { accept: "text/html", ...headers } });
}

test("正常時はオリジンの応答をそのまま返す", async () => {
  const origin = new Response("ok", { status: 200, headers: { "set-cookie": "a=b" } });
  globalThis.fetch = async () => origin;
  const res = await worker.fetch(page());
  assert.equal(res, origin);
});

test("アプリ自身の 500 / 503 は素通しする", async () => {
  for (const status of [500, 503]) {
    globalThis.fetch = async () => new Response("app error", { status });
    const res = await worker.fetch(page());
    assert.equal(res.status, status);
    assert.equal(await res.text(), "app error");
  }
});

test("トンネル切れ(530)・オリジン不達(502)ではお休みページを返す", async () => {
  for (const status of [502, 530]) {
    globalThis.fetch = async () => new Response("cf error", { status });
    const res = await worker.fetch(page());
    assert.equal(res.status, 503);
    assert.equal(res.headers.get("x-shion-offline"), "1");
    assert.equal(res.headers.get("cache-control"), "no-store");
    assert.match(await res.text(), /紫苑は今お休み中です/);
  }
});

test("fetch 自体が失敗してもお休みページを返す", async () => {
  globalThis.fetch = async () => {
    throw new Error("network");
  };
  const res = await worker.fetch(page());
  assert.equal(res.status, 503);
});

test("API 呼び出しには JSON で返す", async () => {
  globalThis.fetch = async () => new Response("", { status: 530 });
  const req = new Request("https://shion.tune77.com/api/chat", {
    method: "POST",
    headers: { accept: "application/json" },
    body: "{}",
  });
  const res = await worker.fetch(req);
  assert.equal(res.status, 503);
  assert.equal((await res.json()).error, "shion_offline");
});

test("確認用ヘッダではオリジンに行かずにお休みページを返す", async () => {
  globalThis.fetch = async () => {
    throw new Error("must not reach origin");
  };
  const res = await worker.fetch(page({ [SIMULATE_HEADER]: "1" }));
  assert.equal(res.status, 503);
  assert.match(await res.text(), /お休み中/);
});
