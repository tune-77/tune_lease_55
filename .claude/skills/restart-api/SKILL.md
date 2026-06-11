---
name: restart-api
description: FastAPI (port 8000) + Next.js (port 3000) のクリーン再起動。ビルド要否を自動判定して漏れなく反映する。「再起動」「API落ちた」「uvicorn迷子」「restart-api」「フロント反映されない」のキーワードで使用。
---

# restart-api スキル

FastAPI と Next.js を `run_next_stable.sh` の FORCE_RESTART で再起動する。
**`SKIP_BUILD` は使わない** — `frontend_build_needed()` が自動判定するため、ビルド漏れが起きない。

## 重要な前提（やり直し多発の防止）

- **ポートを手で kill しない。** 旧ランチャーの supervisor ループが1秒後にプロセスを
  蘇らせ、新旧プロセスがポートを奪い合って起動失敗を繰り返す。停止も含めて
  `FORCE_RESTART=1 bash run_next_stable.sh` 一本に任せること。
- **Cloudflare トンネルは再起動の対象外。** ランチャーは既存の cloudflared を再利用する
  ため quick tunnel の URL は変わらない。トンネルだけ再起動したい場合のみ
  `RESTART_SCOPE=tunnel bash run_next_stable.sh`（この場合は URL が変わる）。
- **FastAPI は起動に2〜4分かかる**（AIモデル等のインポート）。さらにフロント変更が
  あればビルドに約3分。短い sleep + 1回の curl で判定せず、必ずポーリングで待つ。
- launchd ジョブ `com.tunelease.next` は FORCE_RESTART=0 + KeepAlive(SuccessfulExit=false)
  + AbandonProcessGroup=true 構成で、手動再起動と喧嘩しない。launchctl は触らなくてよい。

## 手順

### 1. 再起動（停止も含めてこれ一発）

```bash
cd /Users/kobayashiisaoryou/clawd/tune_lease_55
FORCE_RESTART=1 PUBLIC_TUNNEL=1 nohup bash run_next_stable.sh > logs/next/restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "launcher PID: $!"
```

### 2. 起動完了をポーリングで待つ

`run_in_background: true` の until ループで待つ（タイムアウト目安: 8分）:

```bash
until [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/docs 2>/dev/null)" = "200" ] \
   && [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/ 2>/dev/null)" = "200" ]; do
  sleep 10
done
echo READY
```

5分以上上がらないときはログを確認する:

```bash
tail -20 "$(ls -t logs/next/api_*.log | head -1)"     # FastAPI 起動エラー（トレースバック）
tail -20 "$(ls -t logs/next/build_*.log | head -1)"   # フロントエンドのビルドエラー
```

「FastAPI exited; restarting」が短間隔で連続していたら起動時例外でクラッシュループ中。
トレースバックを読んで原因（import エラー・依存不足など）を修正してから再実行する。

### 3. ステータスとトンネル URL の確認

```bash
RESTART_SCOPE=status bash run_next_stable.sh
```

API / Next / Tunnel URL をまとめて表示する。URL は最新の `logs/next/tunnel_*.log` から
取得される。旧パス `~/Library/Logs/tunelease/cloudflare_3000.log` は**使わない**
（古い URL を返すため）。

### 4. 結果報告

- API: OK / Next: OK なら成功。トンネル URL は通常**変わらない**。
- URL が変わった場合（トンネル自体が落ちて再生成された場合）のみ新 URL をユーザーに伝える。
- ビルドログは `logs/next/build_*.log`、再起動ログは `logs/next/restart_*.log` に残る。
