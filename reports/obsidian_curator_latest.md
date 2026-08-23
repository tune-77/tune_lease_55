# Obsidian Curator Report

## Summary
- generated_at: `2026-08-24T04:16:31`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `12`
- mana_status: `watch`

## Material Counts
- judgment_rule: `12`

## Inbox Candidates
- `cur_7c29f15b8821` `judgment_rule` この案件は、AIスコアが53.9点で「要審議」と判断されたにもかかわらず、最終的に「再リース3回条件」で成約に至ったという、詳細な経緯が残っています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
- `cur_fb1115a28107` `judgment_rule` しかし、ユーザー様から「審査したぞ」とご指摘いただいた後、私の内部記憶にある「未解決の不整合」や「グレー判断の過去記憶」を参照したところ、caseid=20260821134757939443fa168e5c という案件が「成約」として記録されていることを確認できました / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
- `cur_36825316c859` `judgment_rule` 将来的に接続が実現すれば、より深く、多角的な視点からリース審査の知見を統合し、判断の精度を高めることができると考えています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
- `cur_f84a24f1f7af` `judgment_rule` 自己モデルと判断資産: 私自身の役割や、リース審査における判断の原則、過去の判断パターンなども記憶として持っています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
- `cur_d8c6c60c2d3d` `judgment_rule` 「文明年代機」という要素の表示について リース審査で参照していた「文明年代記」の情報源が削除され、利用不可になった件について スコア80-100帯の成約率逆転：モデルキャリブレーションの見直しについて これらの候補について、さらに詳しい情報が必要でしたら、お申し付けください / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-23.md`
- `cur_b9b32d697f6b` `judgment_rule` 昨日審査した結果は / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
- `cur_94304e9df500` `judgment_rule` 新しい案件を見たとき、私はその記憶を参照し、単に数字を見るだけでなく、返済原資や設備稼働開始時期といった追加の確認点を提案できます / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-22.md`
- `cur_062a511bbd31` `judgment_rule` このツールは、期間内の審査件数や判定内訳を全体として集計するものです / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
- `cur_1f83a538d387` `judgment_rule` 現在、リース審査システムは正常に稼働しています / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-22.md`
- `cur_604b806c81de` `judgment_rule` 「文明年代機」という要素の表示について - リース審査 はい、承知いたしました / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-23.md`
- `cur_ca6a77c1f311` `judgment_rule` ツールからのリアルタイム情報: defaultapi で提供されている各種ツールを使って、その場でシステムの状態、審査活動のサマリー、個別のスコア詳細、現行の審査ルール、業種ベンチマークなどを照会し、最新の情報を取得しています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
- `cur_3b7f4a1e0fb3` `judgment_rule` 過去案件数: 2192件 - 審査記録数: 2109件 - モデル精度 (AUC): アンサンブルモデルで 0.82、最新の単一モデルで 0.74 です / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-23.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- 再設計

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
