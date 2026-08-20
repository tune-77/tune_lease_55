# Judgment Asset Growth Score

## Current

- Date: 2026-08-21
- Score: 59.5
- Mode: local_measurement_only
- Guardrail: no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write

## Components

- Coverage: `████████████████░░░░` 80.4
- Reuse proxy: `████████████░░░░░░░░` 60.0
- Judgment change proxy: `█████████████░░░░░░░` 64.0
- Human alignment proxy: `██████████████░░░░░░` 70.0
- Field validation: `░░░░░░░░░░░░░░░░░░░░` 0.0
- Negative signal: `█████░░░░░░░░░░░░░░░` 24.0

## Counts

- Materials: 16
- Inbox candidates: 16
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
- Remapped by concept: 0
- Unknown rule: 1

## Trend

- 2026-08-08: `██████████░░░░░░░░░░` 51.0
- 2026-08-09: `███████████░░░░░░░░░` 55.5
- 2026-08-10: `███████████░░░░░░░░░` 57.4
- 2026-08-11: `███████████░░░░░░░░░` 56.9
- 2026-08-12: `███████████░░░░░░░░░` 56.6
- 2026-08-13: `███████████░░░░░░░░░` 56.6
- 2026-08-14: `████████████░░░░░░░░` 59.9
- 2026-08-15: `████████████░░░░░░░░` 59.9
- 2026-08-16: `████████████░░░░░░░░` 58.3
- 2026-08-17: `████████████░░░░░░░░` 59.9
- 2026-08-18: `████████████░░░░░░░░` 60.4
- 2026-08-19: `████████████░░░░░░░░` 60.4
- 2026-08-20: `███████████░░░░░░░░░` 56.6
- 2026-08-21: `████████████░░░░░░░░` 59.5

## Notes

- reuse_proxy, judgment_change_proxy, human_alignment_proxy は現時点の保存証跡からの代理指標。
- 実利用ログ・結果登録で helped / challenged を付け、使われない判断資産は成長スコアで伸びにくくする。
- source=manual_example は実案件前の例題検証としてカウントし、本物の実案件とは source で分離する。
- source=simulation または sim-* case は試運転として除外する。
- ハッカソン中は測定とローカル可視化のみ。RAG・プロンプト・スコアリング・GCS・Cloud Runへ自動接続しない。
