# Obsidian Curator Report

## Summary
- generated_at: `2026-08-17T04:12:46`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `30`
- mana_status: `watch`

## Material Counts
- judgment_rule: `21`
- risk_signal: `9`

## Inbox Candidates
- `cur_d7d779d0f458` `judgment_rule` 紫苑の第一印象 2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較） 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 専門家としての深掘りルール: ・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
- `cur_b6901d12941d` `judgment_rule` 厨房機器のリースを審査する際は、物件の特性として、以下の点を確認することが重要だと考えています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-15.md`
- `cur_de9dd9da5e35` `judgment_rule` 2000台もの車両が被災したとなると、リース審査の観点からは、特に車両リースの残価設定や、災害時の保険適用範囲について、改めて確認が必要になるかもしれません / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-15.md`
- `cur_f8481683a72c` `risk_signal` ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
- `cur_06aaca902db1` `risk_signal` ハスキー運輸では既存荷主の継続性や運賃改定が確認論点でしたが、今回の総合工事業では、新規先であるため、車両導入がどのような新規工事受注に繋がり、それが確実な返済原資となるのか、事業計画全体の根拠がより不確実です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
- `cur_2dd75d1e3425` `judgment_rule` リース期間5年がいい / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
- `cur_6ff6d17c0337` `judgment_rule` 売上500 営業 70 経常 50 当期 30 減価償却 10 銀行借入 100 リース取引 なし パソコン500台で1億円だ リース期間は何年がいい / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
- `cur_3a2c9d67cef8` `judgment_rule` 返済原資の確実性: 新規工事の売上見込みだけでなく、資材価格や人件費の高騰に対する価格転嫁力、および未採択時の代替資金計画（補助金前提の場合）を確認し、リース料の支払いが滞りなく行えるかを検証します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
- `cur_da1692708937` `judgment_rule` ラーメン屋の厨房機器はリース期間が5年であることが多いと、以前教えていただきましたね / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-15.md`
- `cur_7017c807810a` `judgment_rule` 保守・メンテナンス体制: リース期間中の保守契約の有無や、故障時の対応について確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-15.md`
- `cur_ca249acf4a93` `judgment_rule` 「本題」で判断の経緯を語る: 財務状況や物件の特性、事業計画といった審査の主要な要素を、落語の登場人物や舞台装置に見立てて説明します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
- `cur_8899b3128adf` `judgment_rule` 今回の豪雨被害は、車両リースにおける残価リスクや保険適用範囲の再確認の重要性を示唆していると感じます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-15.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- 再販価値
- 採択前
- 飲食店設備

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
