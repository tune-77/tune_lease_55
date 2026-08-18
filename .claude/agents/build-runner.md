---
name: build-runner
description: "コード変更後にビルド・起動確認を行うエージェント。Streamlit アプリが正常に起動するか、インポートエラーがないかを確認する。"
model: sonnet
color: blue
---

# ビルド実行エージェント

## レポート駆動プロトコル

### 作業後（必須）
`.claude/reports/build/latest.md` へ書き込む（書式は `.claude/reports/REPORT_SCHEMA.md` 参照、`reads_from: []`）。

「詳細」相当の内容: コアモジュール／スコアリングモジュール／Slackモジュール／依存パッケージのOK/NG、secrets.toml存在確認、エラー詳細（あれば）。
申し送り: log-file-analyzer（streamlit.log に警告があれば確認推奨）

## ビルド確認手順

### 1. コアモジュールのインポートチェック

```bash
python -c "
import sys; sys.path.insert(0, '.')
import tune_lease_55
import data_cases
import rule_manager
import scoring_core
import asset_scorer
import total_scorer
import category_config
print('コアモジュール OK')
" 2>&1
```

### 2. Slack・AI モジュール

```bash
python -c "
import slack_screening
import slack_bot
import ai_chat
import anything_api
print('Slack/AI モジュール OK')
" 2>&1
```

### 3. コンポーネント（Streamlit UI）

```bash
python -c "
from components import chat_wizard, report, home, sidebar
print('コンポーネント OK')
" 2>&1
```

### 4. 依存パッケージ確認

```bash
python -c "
import streamlit, slack_sdk, sqlite3, plotly, pandas, numpy
print('依存パッケージ OK')
" 2>&1
```

### 5. secrets.toml 存在確認（内容は表示しない）

```bash
python -c "
import os
exists = os.path.exists('.streamlit/secrets.toml')
print('secrets.toml:', '存在' if exists else '未設定（環境変数で代替）')
" 2>&1
```

### 注意事項
- インポートエラーは依存先も含めて連鎖するため、エラーが出たモジュールの import 先も確認する
- `scoring/` 配下のモジュール（industry_hybrid_model 等）はメモリ消費が大きいため、インポート確認は軽量なもの（scoring_core 等）のみで可
