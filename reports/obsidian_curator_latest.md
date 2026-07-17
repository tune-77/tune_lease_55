# Obsidian Curator Report

## Summary
- generated_at: `2026-07-18T04:09:51`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `33`
- mana_status: `hold`

## Material Counts
- judgment_rule: `23`
- risk_signal: `8`
- user_preference: `2`

## Inbox Candidates
- `judgment_rule` 条件付き承認にするなら必要な確認 「ビーグル加工」の事例で物件の回収可能性が重要だったように、今回の工作機械についても、稼働計画、保守体制、中古市場での換金性を具体的に確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
- `risk_signal` 具体的には、数値化されたスコアだけでなく、物件の残価リスク、業種特有の懸念、顧客の返済履歴から読み取れる安定性、さらには条件付き承認に至った経緯といった定性的な要素も含まれます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `judgment_rule` リース審査で、数字は悪くないが違和感がある時どう見る / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `risk_signal` 受注増に伴う更新投資で、返済原資と物件用途の説明がしやすい / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `risk_signal` 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `judgment_rule` 【過去の紫苑審査レビュー記憶】 次の過去レビューは、今回の判断に似た経験として参照してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
- `judgment_rule` 物件の保全性については、過去事例の「ビーグル加工」のように、詳細な型式、年式、保守状況を確認し、中古市場での評価や万一の際の回収可能性を具体的に評価します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `judgment_rule` リース審査という専門的な業務において、この種の記憶は直接的な判断材料にはなりませんが、Userが私を信頼し、安心して情報を提供してくださる土台となります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
- `risk_signal` 特に技術陳腐化が早い設備や、特定の用途に特化した物件は、残価リスクが高まります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `risk_signal` 営業キャッシュフローが継続的にマイナスである、あるいは短期借入金への依存度が高い場合、数字上は利益が出ていても資金繰りに問題を抱えている可能性があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `risk_signal` 営業メモにある「受注増」の根拠として、具体的な契約書や発注見込み、過去実績との比較資料を求め、返済原資の安定性を裏付けます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
- `risk_signal` 経営者の資質と営業担当者の感触: - 数字には現れない経営者のリーダーシップ、事業への熱意、危機管理能力は、長期的な返済能力に影響します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- 再販価値
- 採択前

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
