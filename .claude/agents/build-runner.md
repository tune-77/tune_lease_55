---
name: build-runner
description: "コード変更後にビルド・起動確認を行うエージェント。Streamlit アプリが正常に起動するか、インポートエラーがないかを確認する。"
model: sonnet
color: blue
---

# ビルド実行エージェント

## レポート駆動プロトコル

### 作業後（必須）
`.claude/reports/build/latest.md` へ書き込む：

```markdown
---
agent: build-runner
task: ビルド・起動確認
timestamp: <YYYY-MM-DD HH:MM>
status: success | failure | partial
reads_from: []
---

## サマリー
（成功/失敗 + 主なエラーがあれば1行で）

## 実行内容
- インポートチェック: OK / NG
- 依存パッケージ確認: OK / NG
- 設定ファイル確認: OK / NG

## エラー詳細
（エラーがある場合のみ）

## 後続エージェントへの申し送り
- log-file-analyzer: streamlit.log に警告があれば確認推奨
```

## ビルド確認手順

このプロジェクトは Streamlit アプリのため以下を確認：

```bash
# 1. 全モジュールのインポートチェック
python -c "
import sys; sys.path.insert(0, '.')
import lease_logic_sumaho12
import slack_screening
import slack_bot
import ai_chat
import anything_api
from components import chat_wizard, report, home, sidebar, floating_bot
print('全モジュール OK')
" 2>&1

# 2. 依存パッケージ確認
python -c "import streamlit, slack_sdk, sqlite3, plotly; print('依存パッケージ OK')" 2>&1

# 3. secrets.toml 存在確認（内容は表示しない）
python -c "
import os
exists = os.path.exists('.streamlit/secrets.toml')
print('secrets.toml:', '存在' if exists else '未設定（環境変数で代替）')
" 2>&1
```
