# 案件削除監査 整合性レポート

- 判定: **PASS**
- 検査時刻: `2026-09-06T19:01:17.609030+00:00`
- DB: `/Users/kobayashiisaoryou/clawd/tune_lease_55/data/lease_data.db`
- SQLite integrity: `ok`
- 削除イベント: 0件
- エラー: 0件 / 警告: 0件
- 状態別: `{}`

## 検査項目

- イベントの要求・一致・削除件数と対象明細の一致
- `affected_screening_count` と同一 `deletion_event_id` の審査記録数の一致
- 関連審査記録が `parent_deleted` であること
- 削除済み案件IDが `past_cases` に残存・再出現していないこと
- 監査イベントのない `screening_records.deletion_event_id` がないこと
- 長時間 `started`、`partial`、不正な状態値がないこと

## 検出事項

- なし

_読み取り専用監査です。自動修復・削除・復元は行いません。_
