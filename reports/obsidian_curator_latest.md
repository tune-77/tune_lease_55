# Obsidian Curator Report

## Summary
- generated_at: `2026-08-14T04:11:54`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `23`
- mana_status: `watch`

## Material Counts
- judgment_rule: `18`
- risk_signal: `5`

## Inbox Candidates
- `cur_d7d779d0f458` `judgment_rule` 紫苑の第一印象 2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較） 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 専門家としての深掘りルール: ・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
- `cur_07263a804009` `judgment_rule` リース審査では、この法定耐用年数を一つの基準として、実際の使用状況や物件の陳腐化リスクなどを考慮し、リース期間が設定されることが多いですよ / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
- `cur_4a2c27ff066c` `judgment_rule` リース期間が物件の経済的耐用年数や陳腐化サイクルと乖離していないか、また、リース終了時の残価設定が妥当かを見極める必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_42ae7beb5c82` `judgment_rule` リース期間満了時の取り扱いについては、契約内容によって返還されるか、次のリース契約に引き継がれるかなどが変わってきますので、個別の契約書で確認することが重要になります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_821ecd4ac95e` `judgment_rule` 保険金額と付保期間: リース物件の時価と保険金額が見合っているか、また、被害発生時が保険の付保期間内であるかをご確認ください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_6c5b4d5eb9a7` `judgment_rule` ですので、シオンさんが提示された7年は、その物件の経済的耐用年数や推奨されるリース期間として妥当と判断されたものかもしれませんね / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
- `cur_49837295bb41` `judgment_rule` 税法上の法定耐用年数はあくまで減価償却の基準ですが、リース会社は物件の実際の耐久性や経済的価値、中古市場での需要、お客様の利用計画などを総合的に判断してリース期間を設定します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
- `cur_f8481683a72c` `risk_signal` ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
- `cur_f4ed02685635` `judgment_rule` 事例2: ビーグル加工 / 2024年下期 / 24 金属製品製造業 類似度: 16 / 理由: 銀行支援が近い・スコア帯が近い・デモ初期経験 スコア・判断: 76.8点 /… / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
- `cur_ad1f9ab4cbec` `judgment_rule` しかし、補助金は採択の確実性、入金時期、未採択時の資金繰りへの影響を慎重に評価する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_a049d4648fe8` `judgment_rule` サプライヤー直送案件の場合、物件の物理的な存在確認と借手の検収体制の適切性が重要になります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
- `cur_1a4c4302e146` `judgment_rule` リース期間を考える上での目安になりますが、実際の契約では、お客様の使い方や物件の寿命なども考慮して決めることになります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- メイン行
- 採択前
- 支援依頼書

## Related Gaps
- `Projects/tune_lease_55/検索語インデックス.md` related_section_missing missing=

## Mana Review Items
- {"action": "Inbox整理は可。GCS/RAG/active store接続は停止。", "reason": "mana_not_allow", "status": "watch"}

## After Hackathon Only
- Obsidianディレクトリ再編
- GCS Vault include/exclude変更
- accepted判断資産のactive store連携
- 判断資産レビューUI

## Next Safe Step
- Inbox候補を人間が採用・修正・却下・後回しに分類する。
- このレポート自体はObsidian本文、RAG、Cloud Run、active storeへ接続しない。
- 採用する場合は candidate_id を data/curator_inbox_decisions.jsonl に {"candidate_id": "...", "decision": "adopt"} として追記し、scripts/promote_curator_inbox_candidates.py を実行する（Mana許可時のみpreviewへ反映、activeへは既存のpromote_canonical_judgment_rules.pyを人間が実行）。
