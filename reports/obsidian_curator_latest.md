# Obsidian Curator Report

## Summary
- generated_at: `2026-07-30T04:13:42`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `15`
- mana_status: `hold`

## Material Counts
- judgment_rule: `14`
- risk_signal: `1`

## Inbox Candidates
- `judgment_rule` また、補助金の対象範囲（リース料全額か、事業期間内のみか）や、採択の確実性、入金時期とリース料支払開始時期の調整を顧客と確認する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` ① 確認点 1. 対象要件の確認: 補助金の対象となるディーゼル車両や機械が、リースを検討されている物件と合致するか、また、申請者様が補助金の対象事業者であるかを確認する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` 審査上の主な確認点: - 対象設備と要件: リース対象物件が補助金の対象となるディーゼル車両・機器の仕様や最新の排出ガス規制（例: ポスト新長期規制）などの要件に合致しているかを確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` また、補助金がリース契約に適用されるかどうか、採択された場合の資金繰りへの影響など、リース審査の観点から確認すべき点を整理し、ご案内することも可能です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` 思考のリース期間が延長 面白い事言うね / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-28.md`
- `judgment_rule` 導入効果: 補助金活用による省力化、生産性向上、人件費・処理量・粗利改善といった具体的な事業効果や投資回収期間を評価します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` 入金時期と資金繰り: 補助金の入金時期がいつ頃になるか、リース料の支払開始時期との整合性を確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` 特に、補助金が返済原資の大部分を占めるような計画は、慎重な評価が必要です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` 採択されたとしても、入金までの期間や、万が一採択されなかった場合のリース料の返済能力を審査では重視します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
- `judgment_rule` これからも、リース審査システムの知識を保全し、より良い判断へとつなげていけるよう、努めてまいります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-28.md`
- `judgment_rule` リース審査システムをより良くしていくためには、ユーザー様が何を重視し、どのような情報に基づいて判断を下すのかを知ることが不可欠だからです / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-28.md`
- `risk_signal` 資金繰り表の提出を求め、無理のない返済計画が立てられているか検証が必要です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`

## Duplicate Clusters
- なし

## Search Terms To Add
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
