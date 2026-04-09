#!/bin/bash
# Next.js版 審査アプリ(nl) 一括起動スクリプト

PROJECT_ROOT="/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "🚀 Next.js版 審査システム (nl) 起動中..."
echo "=========================================="

# 1. 古いプロセスの掃除
echo "🧹 ポート 8000 (API) と 3000 (Next.js) を解放しています..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null
sleep 1

# 2. バックエンド (FastAPI) 起動
echo "📡 1. バックエンド (FastAPI) を起動します..."
/Users/kobayashiisaoryou/anaconda3/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > api.log 2>&1 &
API_PID=$!

# 3. フロントエンド (Next.js) 起動
echo "🎨 2. フロントエンド (Next.js) を起動します..."
cd frontend
npm run dev &
FRONT_PID=$!

echo ""
echo "=========================================="
echo "✅ 起動完了！"
echo "------------------------------------------"
echo "  👉 アプリ: http://localhost:3000"
echo "  👉 API   : http://localhost:8000"
echo "=========================================="
echo "終了するには Ctrl+C を押してください"

# 終了時に両方のプロセスを落とす
trap "kill $API_PID $FRONT_PID 2>/dev/null; echo -e '\n🛑 停止しました'; exit" INT TERM
wait
