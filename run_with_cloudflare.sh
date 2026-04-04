#!/bin/bash
# リース審査アプリを起動し、Cloudflare Tunnel で外部からアクセスできるようにする
# ngrok の代替手段として無料・制限なしで利用可能
#
# インストール方法:
#   Mac:   brew install cloudflare/cloudflare/cloudflared
#   Linux: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
#
# 使い方:
#   ./run_with_cloudflare.sh
#   → 起動後に表示される https://xxxx.trycloudflare.com がアクセス URL

cd "$(dirname "$0")"

echo "Streamlit を起動しています..."
python3 -m streamlit run lease_logic_sumaho12.py \
  --server.address 0.0.0.0 \
  --server.port 8501 &
STREAMLIT_PID=$!

# 終了時に Streamlit を止める
cleanup() {
  echo ""
  echo "終了します..."
  kill $STREAMLIT_PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM

# Streamlit の起動を待つ
sleep 4
if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
  echo "Streamlit の起動に失敗しました。"
  exit 1
fi

echo "Streamlit 起動済み (port 8501)。Cloudflare Tunnel で公開します..."
echo "表示された https://xxxx.trycloudflare.com にアクセスしてください。"
echo ""

# cloudflared でトンネル開始（アカウント不要・無料）
cloudflared tunnel --url http://localhost:8501

# cloudflared が終了したら Streamlit も止める
kill $STREAMLIT_PID 2>/dev/null
