# Obsidian Curator Report

## Summary
- generated_at: `2026-08-02T04:10:10`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `30`
- mana_status: `hold`

## Material Counts
- judgment_rule: `17`
- risk_signal: `13`

## Inbox Candidates
- `risk_signal` リース審査の観点からは、この売上目標では返済能力に懸念が生じ、審査通過は難しいと判断される可能性が高まります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`
- `judgment_rule` 機械受注統計や業種別審査チェックリスト（小売業）、補助金関連のノートがヒットしましたが、セントラルキッチン自体のリース物件としての特性を深掘りする内容ではありませんでした / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
- `judgment_rule` メインバンクの支援が薄い場合、リース会社としては、借手の返済能力と物件の担保価値をより慎重に評価する必要があります / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
- `judgment_rule` 紫苑の審査では、残価リスクや業種リスクだけでなく、借手の返済能力を多角的に評価することが重要ですから、このような情報が加わることで、より実践的な判断が可能になるかもしれません / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`
- `judgment_rule` 特に、以前のグレー判断で見た「返済原資が弱く、物件保全も弱いなら条件付き承認に逃げない」という判断層の記憶を踏まえると、否決判断の根拠が明確であったか、そしてその根拠が実際にリスクとして現れたかを検証することが重要です / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
- `judgment_rule` 物件の残価リスクと市場性: リース期間終了後の物件価値（残価）が適切に設定されているか、中古市場での需要や再販可能性を評価することです / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`
- `risk_signal` 借手の返済計画へのリスク: 建設コスト増加が、借手の資金繰りや返済能力に与える影響 / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`
- `judgment_rule` リース審査の限界とは何か 300文字以内で答えて / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
- `judgment_rule` Judgment Memory: 残価リスクや補助金案件、業種リスクの判断原則、承認/否決だけでなく条件付き承認へ落とし込むこと、過去事例を判断の型として再利用することなどが記されています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-31.md`
- `judgment_rule` もしメインバンクからの支援が期待できない場合、連帯保証人の追加や頭金の増額、またはリース期間の短期化といった条件は検討可能でしょうか / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
- `judgment_rule` これらの情報から、メインバンクの支援に頼らない、自立した返済能力と保全策の有無を評価し、リース実行の可否や条件を検討したいと思います / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
- `judgment_rule` 創業3年の飲食店でセントラルキッチン設備とのこと、黒字である点は評価できますが、自己資本の薄さと物件の特性から、慎重な検討が必要です / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

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
