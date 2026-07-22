# Judgment Asset Growth Score

## Current

- Date: 2026-07-22
- Score: 65.1
- Mode: local_measurement_only
- Guardrail: no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write

## Components

- Coverage: `███████████████████░` 94.0
- Reuse proxy: `████████████░░░░░░░░` 60.0
- Judgment change proxy: `███████████████░░░░░` 75.0
- Human alignment proxy: `██████████████░░░░░░` 70.0
- Field validation: `░░░░░░░░░░░░░░░░░░░░` 0.0
- Negative signal: `█████░░░░░░░░░░░░░░░` 24.0

## Counts

- Materials: 32
- Inbox candidates: 20
- Active rules: 10
- Risk axes: 5
- Concepts: 10
- User evidence: 18

## Field Validation

- Used: 0
- Helped: 0
- Challenged: 0
- Rejected: 0
- Unused active rules: 10

## Trend

- 2026-07-15: `█████████████░░░░░░░` 64.4
- 2026-07-16: `████████████░░░░░░░░` 61.5
- 2026-07-17: `█████████████░░░░░░░` 66.9
- 2026-07-18: `████████████░░░░░░░░` 62.0
- 2026-07-19: `████████████░░░░░░░░` 62.1
- 2026-07-20: `█████████████░░░░░░░` 62.6
- 2026-07-21: `██████████████░░░░░░` 67.5
- 2026-07-22: `█████████████░░░░░░░` 65.1

## Notes

- reuse_proxy, judgment_change_proxy, human_alignment_proxy は現時点の保存証跡からの代理指標。
- 実利用ログ・結果登録で helped / challenged を付け、使われない判断資産は成長スコアで伸びにくくする。
- ハッカソン中は測定とローカル可視化のみ。RAG・プロンプト・スコアリング・GCS・Cloud Runへ自動接続しない。
