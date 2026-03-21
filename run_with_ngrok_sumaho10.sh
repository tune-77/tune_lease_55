#!/bin/bash
# リース審査アプリ (sumaho10) を起動し、ngrok で外部からアクセスできるようにする
cd "$(dirname "$0")"

echo "🚀 Streamlit を起動しています..."
python3 -m streamlit run lease_logic_sumaho10/lease_logic_sumaho10.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
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
  echo "❌ Streamlit の起動に失敗しました。ログを確認してください (streamlit.log)"
  exit 1
fi

echo "✅ Streamlit 起動済み (port 8501)。ngrok で公開します..."
echo ""

# ngrok http --domain lora-gyrational-trebly.ngrok-free.dev 8501 > ngrok.log 2>&1 &
# ngrok http --url=lora-gyrational-trebly.ngrok-free.dev 8501 > ngrok.log 2>&1 &
ngrok http 8501 > ngrok.log 2>&1 &
NGROK_PID=$!

sleep 5
echo "🌐 ngrok URL:"
# ngrok.log から URL を抽出する (json 形式ではない場合があるが、Freeプランでは stdout に url=... が出るか、APIで見れる)
# jq があれば API を叩くのが確実だが、ここでは簡易的に log を cat する
cat ngrok.log
# curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*'

# 待機 (Ctrl+C で終了するまで)
wait $NGROK_PID

# ngrok が終了したら Streamlit も止める
kill $STREAMLIT_PID 2>/dev/null
