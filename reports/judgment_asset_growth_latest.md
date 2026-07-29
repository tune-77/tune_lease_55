# Judgment Asset Growth Score

## Current

- Date: 2026-07-30
- Score: 56.3
- Mode: local_measurement_only
- Guardrail: no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write

## Components

- Coverage: `████████████████░░░░` 79.5
- Reuse proxy: `████████████░░░░░░░░` 60.0
- Judgment change proxy: `████████████░░░░░░░░` 61.5
- Human alignment proxy: `██████████████░░░░░░` 70.0
- Field validation: `░░░░░░░░░░░░░░░░░░░░` 0.0
- Negative signal: `██████████░░░░░░░░░░` 49.0

## Counts

- Materials: 15
- Inbox candidates: 15
- Active rules: 9
- Risk axes: 4
- Concepts: 8
- User evidence: 13

## Field Validation

- Used: 0
- Helped: 0
- Challenged: 0
- Rejected: 0
- Unused active rules: 9

## Trend

- 2026-07-17: `█████████████░░░░░░░` 66.9
- 2026-07-18: `████████████░░░░░░░░` 62.0
- 2026-07-19: `████████████░░░░░░░░` 62.1
- 2026-07-20: `█████████████░░░░░░░` 62.6
- 2026-07-21: `██████████████░░░░░░` 67.5
- 2026-07-22: `█████████████░░░░░░░` 65.1
- 2026-07-23: `████████████░░░░░░░░` 62.1
- 2026-07-24: `█████████████░░░░░░░` 63.6
- 2026-07-25: `█████████████░░░░░░░` 65.1
- 2026-07-26: `███████████░░░░░░░░░` 53.2
- 2026-07-27: `███████████░░░░░░░░░` 52.9
- 2026-07-28: `███████████░░░░░░░░░` 53.4
- 2026-07-29: `███████████░░░░░░░░░` 53.7
- 2026-07-30: `███████████░░░░░░░░░` 56.3

## Notes

- reuse_proxy, judgment_change_proxy, human_alignment_proxy は現時点の保存証跡からの代理指標。
- 実利用ログ・結果登録で helped / challenged を付け、使われない判断資産は成長スコアで伸びにくくする。
- source=manual_example は実案件前の例題検証としてカウントし、本物の実案件とは source で分離する。
- source=simulation または sim-* case は試運転として除外する。
- ハッカソン中は測定とローカル可視化のみ。RAG・プロンプト・スコアリング・GCS・Cloud Runへ自動接続しない。
