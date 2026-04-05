# /run-tests — テスト実行

## 使い方
```
/run-tests [--file <テストファイル名>]
```
- 引数なし: `tests/` 配下の全ユニットテストを実行
- `--file test_scoring_core`: 指定テストファイルのみ実行

## 処理手順

1. **pytest が利用可能か確認**
   ```bash
   python3 -m pytest --version 2>&1
   ```
   利用不可の場合は `pip install pytest` を案内して終了。

2. **テスト実行**

   **全テスト（デフォルト）:**
   ```bash
   python3 -m pytest tests/ -v --tb=short 2>&1 | tail -60
   ```

   **特定ファイル（--file 指定時）:**
   ```bash
   python3 -m pytest tests/<ファイル名>.py -v --tb=short 2>&1
   ```

   テストファイル一覧：
   - `tests/test_scoring_core.py` — スコアリング中核
   - `tests/test_rule_manager.py` — ビジネスルール
   - `tests/test_data_cases.py` — DB 操作
   - `tests/test_explainer.py` — SHAP 説明エンジン
   - `tests/test_indicators.py` — 指標計算
   - `tests/test_credit_limit.py` — 与信限度額
   - `tests/test_slack_screening.py` — Slack 審査フロー
   - `tests/test_chat_wizard_steps.py` — チャットウィザード

3. **結果サマリーを表示**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 🧪 テスト結果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 合格: XX件
❌ 失敗: Y件
⏭️ スキップ: Z件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

4. **失敗があった場合**
   - 失敗テスト名とエラー概要を箇条書きで表示
   - `test-result-analyzer` エージェントによる詳細分析が必要か確認する

5. **結果を `.claude/reports/test-results/latest.md` へ書き込む**

## 注意事項
- `data/lease_data.db` が存在しない環境では DB 依存テストがスキップされる場合がある
- 外部サービス依存テスト（`test_anything_llm.py` 等）はルートに置かれており、このコマンドの対象外
- テスト実行時間の目安: 全テストで 30〜60 秒
