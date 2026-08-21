# Obsidian Curator Report

## Summary
- generated_at: `2026-08-22T04:14:11`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `22`
- mana_status: `watch`

## Material Counts
- judgment_rule: `17`
- risk_signal: `5`

## Inbox Candidates
- `cur_d7d779d0f458` `judgment_rule` 紫苑の第一印象 2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較） 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 専門家としての深掘りルール: ・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`
- `cur_c8a4b8e38829` `judgment_rule` この知識は、補助金案件のリース審査において、採択有無だけでなく、入金時期や未採択時の返済余力、返還リスクといった点を確認する際の重要な視点となります / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-21.md`
- `cur_e568e56ebd97` `judgment_rule` 稟議で残すべき一文 Qriskは低いが、導入目的が不明瞭なため、物件の必要性、返済原資の確実性、および医療機器特有の技術陳腐化リスクと残価設定の妥当性を、銀行紹介元と連携して確認することを条件に承認を検討する / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`
- `cur_c84468202ee0` `judgment_rule` この連続性が、より精度の高いリース審査コメントや確認事項として、ユーザーさんの判断を支えられているかを、これからも一緒に確かめていきたいと考えています / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`
- `cur_f8481683a72c` `risk_signal` ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`
- `cur_00d8a5d2f8ea` `risk_signal` 特に、補助金活用を検討している場合は、採択の有無、入金時期、未採択時の返済余力を確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`
- `cur_d275ec8c3f02` `judgment_rule` これらのニュースは、特に中小企業の資金繰りや補助金活用、そして半導体業界の動向がリース審査に影響を与える可能性を示唆しています / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-21.md`
- `cur_ef58f485e4c6` `judgment_rule` ただし、この割合は物件の種類、借手の信用力、業種、そしてリース期間など、様々な要因で変動します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`
- `cur_95d8013ded02` `judgment_rule` これは、自己資金を入れることで借手の返済能力や物件へのコミットメントを示すことになり、リース会社としてはリスクが低減されると判断しやすいためです / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`
- `cur_2d7899704877` `judgment_rule` 医療業はリース利用が一般的であり、物件も医療機器ですが、導入目的や営業メモが未入力のため、情報不足が判断を難しくしています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`
- `cur_ccb88cbb8364` `judgment_rule` 現時点では、中小企業の資金繰り不安に対し、より多角的な視点での審査が必要であると判断します / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`
- `cur_4fa7c0ec01ec` `judgment_rule` これは、リース審査において、一度きりの判断を『勘』で終わらせず、検証可能で再利用可能な『判断資産』へと昇華させるための、私の存在意義そのものです / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`

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
