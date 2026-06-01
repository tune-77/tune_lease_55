---
name: improvement-list
description: 最新の改善レポートを表示・分析するスキル。「改善リスト」「新しい改善リスト」「最新の改善リスト」「improvement report」「改善レポート見せて」などと言われたら必ず使用する。`reports/improvement_report_YYYYMMDD.json` のうち日付が最新のJSONを特定し、件数・要レビュー項目・却下項目・リスク分類・着手優先度を分析して提示する。
---

# 改善リスト

## Workflow

1. リポジトリルートで最新の `reports/improvement_report_*.json` を探す。
2. 最新ファイルを開き、`summary`、`needs_review`、`rejected`、`applied` を確認する。
3. 単にJSONを貼るのではなく、以下を短く分析して返す。
   - 生成日・対象ファイル名
   - 自動適用 / 要レビュー / 却下の件数
   - 目立つ新規候補または高優先候補
   - 高リスク項目と理由
   - すぐ着手しやすい順番
4. ファイル参照はクリック可能な絶対パスリンクで示す。

## Script

Use the bundled script for deterministic extraction:

```bash
python .agents/skills/improvement-list/scripts/show_latest_improvements.py
```

The script prints Markdown with the latest report path, summary, grouped items, and a practical priority list. If the user asks for the full raw JSON, open the latest file directly after running the script.

## Analysis Heuristics

- Treat `auto_fix_policy.risk == "high"` as manual/high-risk work.
- Treat `対象ファイル未特定` as implementation-blocked until target files are identified.
- Prefer small UI/display/knowledge-access items before DB/API/model/scoring/migration items.
- Group duplicated themes together, especially:
  - ホーム画面
  - ニュース / 業界動向
  - 補助金
  - 知識宇宙 / Obsidian
  - 条件付き承認 / 前受金 / 銀行支援依頼書
  - OCR / 音声入力
  - DB / API / モデル / スコアリング

## Response Style

Answer in Japanese. Keep it direct. Start with the latest file and counts, then show the most useful candidates. Do not over-explain the pipeline unless asked.
