#!/bin/bash
# lease_logic_sumaho10 + Flask 起動スクリプト
# 毎回これを実行すれば確実に起動できます

cd "$(dirname "$0")"

echo "=== 古いプロセスを停止 ==="
lsof -ti:8505 | xargs kill -9 2>/dev/null && echo "Streamlit (8505) 停止" || true
lsof -ti:5050 | xargs kill -9 2>/dev/null && echo "Flask (5050) 停止" || true
sleep 1

echo ""
echo "=== Flask 起動（ポート5050）==="
python lease_logic_sumaho10/web/app.py &
FLASK_PID=$!
echo "Flask PID: $FLASK_PID"
sleep 2

echo ""
echo "=== Streamlit 起動（ポート8505）==="
streamlit run lease_logic_sumaho10/lease_logic_sumaho10.py --server.port 8505 &
STREAMLIT_PID=$!
echo "Streamlit PID: $STREAMLIT_PID"

echo ""
echo "==================================="
echo "✅ 起動完了"
echo "  審査アプリ  : http://localhost:8505"
echo "  簡易審査    : http://localhost:5050"
echo "==================================="
echo ""
echo "終了するには Ctrl+C を押してください"

# 両プロセスを待機（Ctrl+C で両方終了）
trap "kill $FLASK_PID $STREAMLIT_PID 2>/dev/null; echo '停止しました'; exit" INT TERM
wait
