#!/bin/bash
# 文明存続探索システムの起動スクリプト
# 使い方: リポジトリのルート（clawd）で ./run_civilization.sh
cd "$(dirname "$0")"
echo "起動中: 文明存続探索システム (port 8505)"
exec streamlit run "lease_logic_sumaho10(X)/civilization.py" --server.port 8505 --server.headless true
