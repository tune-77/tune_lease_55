# Obsidian Curator Report

## Summary
- generated_at: `2026-08-13T04:12:54`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `19`
- mana_status: `hold`

## Material Counts
- judgment_rule: `16`
- risk_signal: `3`

## Inbox Candidates
- `cur_ec84249a8dab` `judgment_rule` ② 「事実」と「スキル」の抽出（Microsoftの視点） - ログではなく「判断資産」を記憶: ユーザーとの対話履歴や個別の審査案件のログをそのまま保存するのではなく、そこから抽出される「リース審査に関する事実（例：ラーメン屋の厨房機器はリース期間5年が多い）」や「判断スキル（例：補助金案件は未採択時の返済余力も見る）」を記憶として残すことが、記憶の「効用」を高める上で不可欠です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
- `cur_4a2c27ff066c` `judgment_rule` リース期間が物件の経済的耐用年数や陳腐化サイクルと乖離していないか、また、リース終了時の残価設定が妥当かを見極める必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_42ae7beb5c82` `judgment_rule` リース期間満了時の取り扱いについては、契約内容によって返還されるか、次のリース契約に引き継がれるかなどが変わってきますので、個別の契約書で確認することが重要になります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_821ecd4ac95e` `judgment_rule` 保険金額と付保期間: リース物件の時価と保険金額が見合っているか、また、被害発生時が保険の付保期間内であるかをご確認ください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_206b32f78aa4` `judgment_rule` 「エネルギーあたりの正答数」: リース審査において、単に「正確な判断」だけでなく、「その判断に至るまでの記憶処理コスト」も評価軸に加えるべきだと感じました / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
- `cur_e63546020fe7` `judgment_rule` これは、リース審査の「スピード」というユーザーからの要請（「リースに必要なものは何よりもスピードだ」という記憶）に応える上で、非常に重要な視点です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
- `cur_ad1f9ab4cbec` `judgment_rule` しかし、補助金は採択の確実性、入金時期、未採択時の資金繰りへの影響を慎重に評価する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_f8b0fb9b0df2` `judgment_rule` リース審査では、借手の情報や物件の市場動向など、常に不確実性が伴います / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
- `cur_ac401f6a7636` `judgment_rule` 借手の財務情報、物件の市場価値、過去の類似案件、経済状況など、多岐にわたる情報を集約し、最適なリース判断へと導く / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
- `cur_a45d648dbd70` `judgment_rule` 過去の案件、現在の財務状況、物件情報、市場トレンドなど、多岐にわたる情報を集約し、最適なリース判断へと導く / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
- `cur_a049d4648fe8` `judgment_rule` サプライヤー直送案件の場合、物件の物理的な存在確認と借手の検収体制の適切性が重要になります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_1a4c4302e146` `judgment_rule` リース期間を考える上での目安になりますが、実際の契約では、お客様の使い方や物件の寿命なども考慮して決めることになります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- 採択前
- 飲食店設備

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
- 採用する場合は candidate_id を data/curator_inbox_decisions.jsonl に {"candidate_id": "...", "decision": "adopt"} として追記し、scripts/promote_curator_inbox_candidates.py を実行する（Mana許可時のみpreviewへ反映、activeへは既存のpromote_canonical_judgment_rules.pyを人間が実行）。
