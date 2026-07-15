# Obsidian Curator Report

## Summary
- generated_at: `2026-07-16T04:07:10`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `74`
- mana_status: `hold`

## Material Counts
- judgment_rule: `51`
- risk_signal: `21`
- user_preference: `2`

## Inbox Candidates
- `judgment_rule` 非財務情報・定性評価の重要性増: 審査では、単なる財務数値の悪化を機械的に判断するのではなく、物件の事業貢献度、将来のキャッシュフロー創出能力、物件の汎用性や換金性といった非財務情報や定性的な評価の重要性が増します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
- `judgment_rule` 条件付き承認にするなら必要な確認 「ビーグル加工」の事例で物件の回収可能性が重要だったように、今回の工作機械についても、稼働計画、保守体制、中古市場での換金性を具体的に確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
- `risk_signal` 最終的に、「この案件は、補助金未採択時の代替返済計画の提示を条件に、承認可能」という、リスクと保全策が明確に結びついた一つの結論へと収束します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`
- `risk_signal` 補助金案件の場合: 採択の確実性、入金時期、未採択時のリース料返済余力、補助金返還リスクに対する代替策（つなぎ融資や自己資金での対応）を明確にします / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
- `judgment_rule` 期待使用期間とリース期間がズレる案件の審査ポイントは / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
- `judgment_rule` 建設業のリース審査で確認すべきリスクは / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
- `judgment_rule` 未採択の場合や、補助金の入金が遅延した場合でも、リース料を滞りなく支払えるだけの返済余力があるかを確認することが重要です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
- `judgment_rule` 返済原資の評価: 審査の観点からは、補助金が採択されることを前提に返済計画を立てることはリスクが高いと判断します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
- `judgment_rule` これは、法定耐用年数が税務上の概念であり、リース審査においては、物件の経済的耐用年数や中古市場価値、貴社の運用期間といった「実際の価値と利用期間」がより重視されるためかもしれません / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`
- `risk_signal` 受注増に伴う更新投資で、返済原資と物件用途の説明がしやすい / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
- `risk_signal` 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
- `risk_signal` 得た教訓: 飲食新規は一括で通すより、投資範囲を絞って撤退時損失を小さくする方が現実的 / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- 再販価値
- 採択前
- 新店舗

## Related Gaps
- `Projects/tune_lease_55/検索語インデックス.md` related_section_missing missing=

## Mana Review Items
- {"action": "Inbox整理は可。GCS/RAG/active store接続は停止。", "reason": "mana_not_allow", "status": "hold"}

## After Hackathon Only
- Obsidianディレクトリ再編
- GCS Vault include/exclude変更
- accepted判断資産のactive store連携
- 判断資産レビューUI

## Next Safe Step
- Inbox候補を人間が採用・修正・却下・後回しに分類する。
- このレポート自体はObsidian本文、RAG、Cloud Run、active storeへ接続しない。
