# エージェント間レポートスキーマ

エージェントはタスク完了後に以下の形式でレポートを書く。
後続エージェントはこのレポートを読んでから作業を開始する。

## ファイル配置

```
.claude/reports/
  file-searcher/latest.md      ← 対象ファイル一覧
  impact-analysis/latest.md   ← ビジネス影響分析
  code-review/latest.md        ← コード品質レビュー
  security/latest.md           ← セキュリティ検査
  build/latest.md              ← ビルド結果
  test-results/latest.md       ← テスト実行結果
  log-analysis/latest.md       ← ログ分析結果
```

## レポートフォーマット

```markdown
---
agent: <エージェント名>
task: <実施したタスクの概要>
timestamp: <YYYY-MM-DD HH:MM>
status: success | failure | partial
reads_from: [読み込んだ上流レポート一覧]
---

## サマリー
1〜3行で結果を要約。

## 詳細
箇条書きで具体的な発見・結果。

## 課題・リスク
問題点や懸念事項があれば記載。なければ「なし」。

## 後続エージェントへの申し送り
次に動くべきエージェントと、伝えておくべき情報。
```

## エージェント依存グラフ

```
file-searcher
    ├── impact-analysis   (file-searcher のレポートを読む)
    ├── code-reviewer     (file-searcher のレポートを読む)
    │       └── security-checker  (file-searcher + code-review を読む)
    └── (直接 build/test へも連携可)

build-runner
    └── (独立実行 or file-searcher 後)

test-runner
    └── test-result-analyzer  (test-runner のレポートを読む)

log-file-analyzer
    └── (独立実行 or test/build 後)
```
