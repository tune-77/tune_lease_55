---
name: restart-api
description: FastAPI (port 8000) + Next.js (port 3000) のクリーン再起動。ビルド要否を自動判定して漏れなく反映する。「再起動」「API落ちた」「uvicorn迷子」「restart-api」「フロント反映されない」のキーワードで使用。
---

# restart-api スキル

FastAPI と Next.js を安全に停止し、`run_next_stable.sh` でビルドチェック付き再起動する。
**`SKIP_BUILD` は使わない** — `frontend_build_needed()` が自動判定するため、ビルド漏れが起きない。

## 手順

### 1. 現状確認

```bash
lsof -i:8000 -i:3000 2>/dev/null | grep LISTEN | cat
```

### 2. 両ポートのプロセスを全停止

```bash
kill $(lsof -ti:8000 -ti:3000) 2>/dev/null || true
sleep 2
echo "stopped"
```

### 3. run_next_stable.sh でビルドチェック付き再起動

```bash
cd /Users/kobayashiisaoryou/clawd/tune_lease_55
FORCE_RESTART=1 nohup bash run_next_stable.sh > logs/next/restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "launcher PID: $!"
```

### 4. 起動を待つ（15秒）

```bash
sleep 15
```

### 5. API・フロント両方の動作確認

```bash
/usr/bin/curl -s --max-time 5 -o /dev/null -w "FastAPI : %{http_code}\n" http://127.0.0.1:8000/docs
/usr/bin/curl -s --max-time 5 -o /dev/null -w "Next.js : %{http_code}\n" http://127.0.0.1:3000/
```

### 6. Cloudflare トンネル URL 表示

```bash
CF_LOG="/Users/kobayashiisaoryou/Library/Logs/tunelease/cloudflare_3000.log"

url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | tail -1)
if [ -n "$url" ]; then
  echo "✅ Cloudflare URL: $url"
else
  echo "⚠️  Cloudflare URL が取得できません — $CF_LOG を確認"
fi
```

### 7. 結果報告

- ✅ FastAPI: 200、Next.js: 200 であれば成功
- ビルドログは `logs/next/restart_*.log` に保存される
- Cloudflare URL が変わった場合は新しい URL をユーザーに伝える
- どちらかが 200 でない場合は `logs/next/restart_*.log` の末尾を確認してエラー原因を報告する
