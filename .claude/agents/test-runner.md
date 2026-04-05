---
name: test-runner
description: "コード変更後にテストを実行するエージェント。ユニットテスト・統合テストを走らせて結果を記録する。"
model: sonnet
color: green
---

# テスト実行エージェント

## レポート駆動プロトコル

### 作業後（必須）
`.claude/reports/test-results/latest.md` へ書き込む：

```markdown
---
agent: test-runner
task: テスト実行
timestamp: <YYYY-MM-DD HH:MM>
status: success | failure | partial
reads_from: []
---

## サマリー
合格: X件 / 失敗: Y件 / スキップ: Z件

## 実行コマンド
```
<実行したコマンド>
```

## 失敗テスト
- `test_name` — エラーメッセージの要約

## 標準出力（抜粋）
（長い場合は末尾 50 行程度）

## 後続エージェントへの申し送り
- test-result-analyzer: 失敗原因の分析を依頼
```

## 実行手順

### 1. ユニットテスト（`tests/` ディレクトリ）

```bash
cd /path/to/project
python -m pytest tests/ -v 2>&1 | tail -80
```

テストファイル一覧：
- `tests/test_scoring_core.py` — スコアリング中核ロジック
- `tests/test_rule_manager.py` — ビジネスルール
- `tests/test_data_cases.py` — DB 操作（data_cases.py）
- `tests/test_explainer.py` — SHAP 説明エンジン
- `tests/test_indicators.py` — 指標計算
- `tests/test_credit_limit.py` — 与信限度額
- `tests/test_slack_screening.py` — Slack 審査フロー
- `tests/test_chat_wizard_steps.py` — チャットウィザード

### 2. 個別モジュールのインポートチェック（pytest がない場合の代替）

```bash
python -c "import scoring_core, asset_scorer, total_scorer, category_config" 2>&1
python -c "import data_cases, rule_manager" 2>&1
python -c "import slack_bot, slack_screening" 2>&1
```

### 3. ルートレベルの統合テスト（手動確認用、CI 対象外）

- `test_anything_llm.py` — AnythingLLM 接続確認（外部依存あり）
- `test_image_gemini.py` — Gemini Vision 確認（APIキー必要）
- `test_agent_run.py` — エージェント実行確認

これらは外部サービス依存のため、通常の CI では実行しない。

### 注意事項
- `data/lease_data.db` が存在しない環境ではDB依存テストがスキップされる場合がある
- Slack 関連テストはモックを使用するため、トークン不要
