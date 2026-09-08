---
agent: data-quality-checker
task: SQLite 審査データの件数・異常値チェック
timestamp: 2026-09-08 04:16
status: partial
reads_from: []
---

## サマリー
33 テーブルを確認し、要確認事項を 1 件検出した。

## 詳細
- テーブル数: 33
-   asset_price_history: 17378件
-   past_cases: 2352件
-   screening_records: 2109件
-   ml_features: 1941件
-   chat_messages: 1368件
-   emotion_history: 119件
-   screening_outcomes: 118件
-   excluded_grade_cases: 99件
-   base_rate_master: 44件
-   cloudrun_return_promotions: 29件

## 課題・リスク
- 空のテーブル: case_deletion_event_items, case_deletion_events, emotion_feedback, judgment_lifecycle_events, payment_history, phrase_weights, shion_followup_impact_feedback, shion_followup_sessions, sync_log

## 後続エージェントへの申し送り
なし

## 過去のエージェント指摘（未検証）
2026-03-28 14:55 時点のエージェント作成レポートを `previous-agent-report.md` に退避した。決定論的な監査では再現できない指摘を含むため、現在のコードで検証すること。

<!-- generated-by: build_agent_self_reports.py -->
<!-- 決定論的監査・LLM不使用。手で編集しても次回実行で上書きされる。
     『過去のエージェント指摘』は解消したら手で消すこと。 -->
