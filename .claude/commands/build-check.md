# /build-check — ビルド・起動確認

## 使い方
```
/build-check
```
全モジュールのインポートチェックと依存パッケージ確認を実行する。

## 処理手順

1. **コアモジュールのインポートチェック**
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, '.')
   results = {}

   modules = [
       ('コア', ['lease_logic_sumaho12', 'data_cases', 'rule_manager']),
       ('スコアリング', ['scoring_core', 'asset_scorer', 'total_scorer', 'category_config']),
       ('Slack/AI', ['slack_screening', 'slack_bot', 'ai_chat', 'anything_api']),
   ]

   for group, mods in modules:
       for m in mods:
           try:
               __import__(m)
               results[m] = '✅'
           except Exception as e:
               results[m] = f'❌ {e}'
       statuses = [results[m] for m in mods]
       ok = all(s == '✅' for s in statuses)
       print(f'{'✅' if ok else '❌'} {group}: {\" / \".join(m for m in mods)}')
       if not ok:
           for m in mods:
               if results[m] != '✅':
                   print(f'  {results[m]} ({m})')
   " 2>&1
   ```

2. **コンポーネント（Streamlit UI）**
   ```bash
   python3 -c "
   try:
       from components import chat_wizard, report, home, sidebar
       print('✅ コンポーネント: OK')
   except Exception as e:
       print(f'❌ コンポーネント: {e}')
   " 2>&1
   ```

3. **依存パッケージ**
   ```bash
   python3 -c "
   pkgs = ['streamlit', 'slack_sdk', 'sqlite3', 'plotly', 'pandas', 'numpy']
   for p in pkgs:
       try:
           __import__(p)
           print(f'✅ {p}')
       except ImportError:
           print(f'❌ {p} — pip install {p}')
   " 2>&1
   ```

4. **secrets.toml 確認**
   ```bash
   python3 -c "
   import os
   print('secrets.toml:', '存在' if os.path.exists('.streamlit/secrets.toml') else '未設定（環境変数で代替）')
   " 2>&1
   ```

5. **結果サマリーを表示**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 🔨 ビルドチェック結果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ コアモジュール     — OK
✅ スコアリング       — OK
✅ Slack/AI          — OK
✅ コンポーネント     — OK
✅ 依存パッケージ     — OK
⚠️ secrets.toml     — 未設定
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

6. **結果を `.claude/reports/build/latest.md` へ書き込む**

## 注意事項
- このコマンドはコードを変更しない（Read-only）
- インポートエラーは依存先も連鎖するため、エラーメッセージの import 元をたどること
- `scoring/industry_hybrid_model` 等の大型モジュールはインポート対象外（起動時間短縮のため）
