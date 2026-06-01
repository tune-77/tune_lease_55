# Report Sync Notes

## 対象ファイル

- `reports/improvement_report_YYYYMMDD.json`
- `reports/latest.json`

## 反映ルール

- `applied` は `id`, `file`, `title`, `pr_url` の最小構成で扱う。
- `needs_review` から移した ID は削除する。
- `summary.applied_count` と `summary.needs_review_count` を更新する。
- `reports/latest.json` では `applied_improvements` と `items[].status` も同期する。
- `status` は `applied_count > 0` なら `COMPLETED`、それ以外は `NO_APPLIED` を基本にする。

## 実装メモ

- 既存の順序は維持する。
- 重複 ID は無視する。
- 見つからない ID は警告してスキップする。
