# Judgment Asset Growth Score

## Current

- Date: 2026-09-08
- Score: 46.6
- Mode: local_measurement_only
- Guardrail: no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write

## Components

- Coverage: `█████████░░░░░░░░░░░` 43.0
- Reuse proxy: `████████████░░░░░░░░` 60.0
- Judgment change proxy: `███████████░░░░░░░░░` 54.0
- Human alignment proxy: `██████████████░░░░░░` 70.0
- Field validation: `░░░░░░░░░░░░░░░░░░░░` 0.0
- Negative signal: `████████░░░░░░░░░░░░` 39.0

## Counts

- Materials: 0
- Inbox candidates: 0
- Active rules: 8
- Risk axes: 4
- Concepts: 8
- User evidence: 12

## Field Validation

- Used: 0
- Helped: 0
- Challenged: 0
- Rejected: 0
- Unused active rules: 8
- Remapped by concept: 0
- Unknown rule: 1

## Trend

- 2026-08-26: `████████████░░░░░░░░` 59.1
- 2026-08-27: `████████████░░░░░░░░` 58.8
- 2026-08-28: `███████████░░░░░░░░░` 52.8
- 2026-08-29: `██████████░░░░░░░░░░` 50.5
- 2026-08-30: `███████████░░░░░░░░░` 53.4
- 2026-08-31: `███████████░░░░░░░░░` 53.4
- 2026-09-01: `██████████░░░░░░░░░░` 52.4
- 2026-09-02: `████████████░░░░░░░░` 58.1
- 2026-09-03: `████████████░░░░░░░░` 58.6
- 2026-09-04: `███████████░░░░░░░░░` 55.9
- 2026-09-05: `███████████░░░░░░░░░` 55.9
- 2026-09-06: `██████████░░░░░░░░░░` 51.8
- 2026-09-07: `██████████░░░░░░░░░░` 47.6
- 2026-09-08: `█████████░░░░░░░░░░░` 46.6

## Notes

- reuse_proxy, judgment_change_proxy, human_alignment_proxy は現時点の保存証跡からの代理指標。
- 実利用ログ・結果登録で helped / challenged を付け、使われない判断資産は成長スコアで伸びにくくする。
- source=manual_example は実案件前の例題検証としてカウントし、本物の実案件とは source で分離する。
- source=simulation または sim-* case は試運転として除外する。
- ハッカソン中は測定とローカル可視化のみ。RAG・プロンプト・スコアリング・GCS・Cloud Runへ自動接続しない。
