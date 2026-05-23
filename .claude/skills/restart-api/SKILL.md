---
name: restart-api
description: FastAPI (port 8000) のクリーン再起動。迷子uvicorn(0.0.0.0:8000)を親ごと終了してから run_next_stable.sh のsupervisorに再起動させる。「再起動」「API落ちた」「uvicorn迷子」「restart-api」のキーワードで使用。
---

# restart-api スキル

FastAPI の迷子プロセスを安全に整理してクリーン再起動する。

## 手順

### 1. 現状確認

```bash
lsof -ti:8000 | xargs ps -o pid,ppid,args -p 2>/dev/null | cat
```

### 2. 迷子プロセス（0.0.0.0:8000）の親ごとkill

```bash
# 0.0.0.0バインドのuvicornとその親supervisorをkill
rogue_pids=$(lsof -ti:8000 2>/dev/null | xargs -I{} sh -c 'ps -o pid=,args= -p {} 2>/dev/null' | grep '0\.0\.0\.0' | awk '{print $1}')
if [ -n "$rogue_pids" ]; then
  # 親PIDも取得してkill
  for pid in $rogue_pids; do
    ppid=$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')
    echo "Killing rogue PID=$pid PPID=$ppid"
    kill $ppid $pid 2>/dev/null || true
  done
fi
```

### 3. 127.0.0.1側も再起動（新コード読み込みのため）

```bash
kill $(lsof -ti:8000) 2>/dev/null || true
```

### 4. supervisor再起動を待つ（10秒）

```bash
sleep 10
```

### 5. 動作確認

```bash
curl -s -o /dev/null -w "FastAPI: %{http_code}\n" http://127.0.0.1:8000/
lsof -ti:8000 | xargs ps -o pid,args -p 2>/dev/null | cat
```

### 6. Cloudflare トンネル起動確認・起動・URL表示

フロントエンド（Next.js port 3000）向けのトンネルを起動する。

```bash
CF_LOG="/tmp/cloudflared_3000.log"

show_cf_url() {
  for i in $(seq 1 15); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1)
    if [ -n "$url" ]; then
      echo "✅ Cloudflare URL: $url"
      return 0
    fi
    sleep 1
  done
  echo "⚠️  URL取得タイムアウト — $CF_LOG を確認"
}

if ps aux | grep -v grep | grep 'cloudflared' | grep '3000' > /dev/null 2>&1; then
  echo "Cloudflare tunnel for :3000 already running"
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1)
  if [ -n "$url" ]; then
    echo "✅ Cloudflare URL: $url"
  else
    echo "(既存トンネルのURL: $CF_LOG を確認)"
  fi
else
  echo "Cloudflare tunnel for :3000 not running — starting..."
  > "$CF_LOG"
  nohup cloudflared tunnel --url http://127.0.0.1:3000 > "$CF_LOG" 2>&1 &
  show_cf_url
fi
```

### 7. 結果報告

- ✅ FastAPI: HTTP 200 が返れば成功
- ポート8000に `127.0.0.1` バインドのプロセスが1つだけ残っていればOK
- `0.0.0.0` バインドが残っていたら手順2を繰り返す
- ✅ Cloudflare URL が表示されれば完了（既存・新規ともに URL を出力）
