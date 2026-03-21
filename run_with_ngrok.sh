#!/bin/bash
# リース審査アプリを起動し、ngrok で外部からアクセスできるようにする
cd "$(dirname "$0")"

echo "🚀 Streamlit を起動しています..."
python3 -m streamlit run lease_logic_sumaho8.py --server.address 0.0.0.0 --server.port 8501 &
STREAMLIT_PID=$!

# 終了時に Streamlit を止める
cleanup() {
  echo ""
  echo "🛑 終了します..."
  kill $STREAMLIT_PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM

sleep 4
if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
  echo "❌ Streamlit の起動に失敗しました。"
  exit 1
fi

echo "✅ Streamlit 起動済み (port 8501)。ngrok で公開します..."
echo "   ※ ngrok が未インストールの場合は: brew install ngrok または https://ngrok.com から取得"
echo ""
ngrok http 8501

# ngrok が終了したら Streamlit も止める
kill $STREAMLIT_PID 2>/dev/null
