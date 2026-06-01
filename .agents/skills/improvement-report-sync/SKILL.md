---
name: improvement-report-sync
description: 改善済みの項目を reports/improvement_report_YYYYMMDD.json と reports/latest.json に反映し、applied / needs_review / 件数 / status を同期する。改善実装後に「済にして」「改善済み登録」「レポートへ反映」などと言われたときに使う。
---

# Improvement Report Sync

## 目的

改善が実装済みになった項目を、レポート側の `applied` へ移し、`needs_review` から外し、`reports/latest.json` も同じ状態に保つ。

## 手順

1. 対象の改善 ID を確認する。
2. 最新の `reports/improvement_report_YYYYMMDD.json` と `reports/latest.json` を開く。
3. `scripts/sync_improvement_reports.py` で `needs_review` から `applied` へ移す。
4. JSON を検証し、件数と `status` が一致するか確認する。
5. 4:00 の自動生成パイプラインでは、`scripts/run_daily_improvement_pipeline.sh` の最後で同じ同期処理を自動実行する。

## 実行方針

- 既に `applied` にある項目は重複追加しない。
- `needs_review` にある同一 ID は除去する。
- `reports/latest.json` の `items` は、該当 ID を `APPLIED` に更新する。
- 変更後は `python -m json.tool` で整合性を確認する。

## 使いどころ

- 「昨日改善したものは済にして」
- 「改善済み登録して」
- 「報告書に反映して」
- 「latest.json も同期して」

## 参照

- [scripts/sync_improvement_reports.py](scripts/sync_improvement_reports.py)
- [references/report-sync.md](references/report-sync.md)
