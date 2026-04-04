# /analyze-logs — ログ分析

## 使い方
```
/analyze-logs [--path <ログファイルパス>]
```
- 引数なし: デフォルトのログパスを自動検索して分析
- `--path logs/streamlit.log`: 指定ファイルのみ分析

## 処理手順

1. **ログファイルを探索**
   ```bash
   python3 -c "
   import os
   candidates = [
       'logs/streamlit.log', 'streamlit.log',
       'logs/slack_bot.log', 'logs/app.log',
   ]
   found = [p for p in candidates if os.path.exists(p)]
   if found:
       for p in found:
           size = os.path.getsize(p) / 1024
           print(f'  {p} ({size:.1f}KB)')
   else:
       print('ログファイルが見つかりません')
   "
   ```

2. **エラー・警告を抽出**
   ```bash
   python3 -c "
   import re, os

   log_file = '<ログパス>'  # 見つかったファイル
   if not os.path.exists(log_file):
       print('ログファイルが存在しません')
       exit()

   with open(log_file, encoding='utf-8', errors='replace') as f:
       lines = f.readlines()

   patterns = {
       'ERROR': re.compile(r'error|exception|traceback', re.I),
       'WARNING': re.compile(r'warning|warn', re.I),
       'CRITICAL': re.compile(r'critical|fatal', re.I),
   }

   counts = {k: 0 for k in patterns}
   samples = {k: [] for k in patterns}

   for line in lines:
       for level, pat in patterns.items():
           if pat.search(line):
               counts[level] += 1
               if len(samples[level]) < 3:
                   samples[level].append(line.strip()[:120])

   print(f'総行数: {len(lines)}')
   for level in ['CRITICAL', 'ERROR', 'WARNING']:
       print(f'{level}: {counts[level]}件')
       for s in samples[level]:
           print(f'  {s}')
   "
   ```

3. **既知エラーパターンの照合**

   | パターン | 原因 | 対処 |
   |---------|------|------|
   | `ollama.connect` 失敗 | Ollama 未起動 | `ollama serve` |
   | `AnythingLLM` 接続エラー | ポート 3001 未起動 | AnythingLLM 起動 |
   | `SLACK_BOT_TOKEN` 未設定 | secrets.toml 未設定 | トークン設定 |
   | `slack_sessions.json` JSON エラー | ファイル破損 | ファイル削除 |
   | `sqlite3.OperationalError: locked` | DB 多重アクセス | プロセス確認 |

4. **結果サマリーを表示して `.claude/reports/log-analysis/latest.md` へ書き込む**

## 注意事項
- ログファイルが存在しない場合は「ログなし」として記録する
- 大きいログファイル（10MB 超）は末尾 1000 行のみを対象とする
- 個人情報・APIキーが含まれる行はレポートに含めない
