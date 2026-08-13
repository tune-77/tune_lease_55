# Judgment Asset Growth Score

## Current

- Date: 2026-08-11
- Score: 56.9
- Mode: local_measurement_only
- Guardrail: no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write

## Components

- Coverage: `█████████████████░░░` 84.0
- Reuse proxy: `████████████░░░░░░░░` 60.0
- Judgment change proxy: `█████████████░░░░░░░` 64.0
- Human alignment proxy: `██████████████░░░░░░` 70.0
- Field validation: `░░░░░░░░░░░░░░░░░░░░` 0.0
- Negative signal: `████████████░░░░░░░░` 59.0

## Counts

- Materials: 25
- Inbox candidates: 20
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

- 2026-07-29: `███████████░░░░░░░░░` 53.7
- 2026-07-30: `███████████░░░░░░░░░` 56.3
- 2026-07-31: `███████████░░░░░░░░░` 57.1
- 2026-08-01: `███████████░░░░░░░░░` 57.4
- 2026-08-02: `███████████░░░░░░░░░` 57.4
- 2026-08-03: `███████████░░░░░░░░░` 57.4
- 2026-08-04: `███████████░░░░░░░░░` 57.4
- 2026-08-05: `███████████░░░░░░░░░` 56.8
- 2026-08-06: `███████████░░░░░░░░░` 57.4
- 2026-08-07: `███████████░░░░░░░░░` 57.4
- 2026-08-08: `██████████░░░░░░░░░░` 51.0
- 2026-08-09: `███████████░░░░░░░░░` 55.5
- 2026-08-10: `███████████░░░░░░░░░` 57.4
- 2026-08-11: `███████████░░░░░░░░░` 56.9

## Notes

- reuse_proxy, judgment_change_proxy, human_alignment_proxy は現時点の保存証跡からの代理指標。
- 実利用ログ・結果登録で helped / challenged を付け、使われない判断資産は成長スコアで伸びにくくする。
- source=manual_example は実案件前の例題検証としてカウントし、本物の実案件とは source で分離する。
- source=simulation または sim-* case は試運転として除外する。
- ハッカソン中は測定とローカル可視化のみ。RAG・プロンプト・スコアリング・GCS・Cloud Runへ自動接続しない。
