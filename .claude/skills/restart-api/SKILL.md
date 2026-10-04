---
name: restart-api
description: FastAPI (port 8000) + Next.js (port 3000) のクリーン再起動。ビルド要否を自動判定して漏れなく反映する。「再起動」「API落ちた」「uvicorn迷子」「restart-api」「フロント反映されない」のキーワードで使用。
---

# restart-api スキル

FastAPI と Next.js を `run_next_stable.sh` の FORCE_RESTART で再起動する。
**`SKIP_BUILD` は使わない** — `frontend_build_needed()` が自動判定するため、ビルド漏れが起きない。

## 重要な前提（やり直し多発の防止）

- **2026-10 からこの Mac が本番（Cloudflare 版）。** 再起動中の数分間は本番も止まり、外からは
  休止ページ Worker の「紫苑は今お休み中です」が見える。Cloud Run は停止中の予備で、デプロイ先ではない。
  Mac のスリープ・蓋閉じでも本番が止まるので、電源接続時はスリープしない設定にしておく。

- **ポートを手で kill しない。** 旧ランチャーの supervisor ループが1秒後にプロセスを
  蘇らせ、新旧プロセスがポートを奪い合って起動失敗を繰り返す。停止も含めて
  `FORCE_RESTART=1 bash run_next_stable.sh` 一本に任せること。
- **公開URLは固定: https://shion.tune77.com**（Cloudflare named tunnel `tune-lease-55`、
  設定 `~/.cloudflared/tune-lease-55.yml`）。再起動しても URL は変わらない。
- **Cloudflare トンネルは再起動の対象外。** ランチャーは既存の cloudflared を再利用する。
  トンネルだけ再接続したい場合のみ `RESTART_SCOPE=tunnel bash run_next_stable.sh`
  （named tunnel なので URL は同じ。数秒だけ公開URLが落ちる）。
- **FastAPI は起動に2〜4分かかる**（AIモデル等のインポート）。さらにフロント変更が
  あればビルドに約3分。短い sleep + 1回の curl で判定せず、必ずポーリングで待つ。
- launchd ジョブ `com.tunelease.next`（KeepAlive）がランチャーを常駐させ、ランチャー内の
  監視ループが FastAPI / Next.js / cloudflared をそれぞれ落ちたら自動で起こし直す。

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
until [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/healthz 2>/dev/null)" = "200" ] \
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

API / Next / Tunnel URL をまとめて表示する。`/docs` は公開運用では無効（404）なので
起動判定には `/healthz` を使う。外からの疎通は次で確認する（Cloudflare Access 導入前は 200、
導入後は未ログインなのでログイン画面への 302 が正常。Mac 側まで届いているかは
`bash scripts/cloudflare_edge_setup.sh --verify` が service token で確認する）:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://shion.tune77.com/chat
```

### 4. 結果報告

- API: OK / Next: OK / https://shion.tune77.com/chat が 200（Access 導入後は 302）なら成功。URL は固定なので伝え直す必要はない。
- ビルドログは `logs/next/build_*.log`、再起動ログは `logs/next/restart_*.log` に残る。
