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

このプロジェクトには pytest ベースのテストがある場合、以下の順で実行：

```bash
# 1. スコアリングモジュールのテスト
python -m pytest scoring/ -v 2>&1 | head -100

# 2. 個別モジュールの構文チェック
python -c "import lease_logic_sumaho12" 2>&1
python -c "import slack_screening" 2>&1
python -c "import slack_bot" 2>&1
python -c "from components import chat_wizard, report, home, sidebar" 2>&1

# 3. テストファイルがある場合
python -m pytest tests/ -v 2>&1 | head -100
```

テストがない場合は構文チェックとインポートチェックを実施し、その結果をレポートする。
