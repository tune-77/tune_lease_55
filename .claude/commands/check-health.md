# /check-health — 全依存サービス接続確認

## 使い方
```
/check-health [--quick]
```
- `--quick`: Gemini/Ollama のみ（30秒以内）
- 引数なし: 全サービスをチェック（1〜2分）

## 処理手順

1. **環境確認**
   - `.streamlit/secrets.toml` が存在するか確認（内容は読まない）
   - 環境変数 `GEMINI_API_KEY`, `SLACK_BOT_TOKEN`, `OLLAMA_HOST` の存在確認（値は表示しない）

2. **各サービスをチェック（並列で実行）**

   **Gemini API:**
   ```bash
   python3 -c "
   import os, urllib.request, json
   key = os.environ.get('GEMINI_API_KEY', '')
   if not key:
       try:
           import tomllib
           with open('.streamlit/secrets.toml', 'rb') as f:
               key = tomllib.load(f).get('GEMINI_API_KEY', '')
       except: pass
   if not key:
       print('❌ Gemini: APIキー未設定')
   else:
       try:
           url = f'https://generativelanguage.googleapis.com/v1beta/models?key={key}'
           req = urllib.request.urlopen(url, timeout=5)
           print('✅ Gemini API: 正常')
       except Exception as e:
           print(f'❌ Gemini API: {e}')
   "
   ```

   **Ollama:**
   ```bash
   python3 -c "
   import os, urllib.request
   host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
   try:
       req = urllib.request.urlopen(f'{host}/api/tags', timeout=3)
       print(f'✅ Ollama ({host}): 正常')
   except Exception as e:
       print(f'❌ Ollama: {e}')
   "
   ```

   **SQLite lease_data.db:**
   ```bash
   python3 -c "
   import sqlite3, os
   from contextlib import closing
   db = 'data/lease_data.db'
   if not os.path.exists(db):
       print('⚠️ SQLite: lease_data.db が存在しない（初回起動前）')
   else:
       size = os.path.getsize(db) / 1024 / 1024
       with closing(sqlite3.connect(db)) as conn:
           tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
       print(f'✅ SQLite: 正常 ({size:.1f}MiB, {len(tables)}テーブル)')
   "
   ```

3. **結果サマリーを表示**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 🩺 ヘルスチェック結果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Gemini API        — 正常 (152ms)
✅ Ollama            — 正常 (llama3 利用可)
❌ Slack Bot Token   — 未設定（Slack機能は無効）
⚠️ e-Stat API       — タイムアウト（キャッシュで代替可）
✅ SQLite            — 正常 (12.3MiB, 5テーブル)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
正常: 3 / 警告: 1 / 異常: 1
```

4. **異常がある場合の推奨対処を表示**
   - Gemini 停止 → `GEMINI_API_KEY` の確認、Ollama へのフォールバック案内
   - Ollama 停止 → `ollama serve` の実行方法を案内
   - Slack 未設定 → `secrets.toml` への追加方法を案内

## 注意事項
- APIキー・トークンの値をコンソールに表示しない
- `api-health-checker` エージェントを起動して詳細レポートを生成する場合は `Agent` ツールを使う
