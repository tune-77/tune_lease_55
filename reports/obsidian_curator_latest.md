# Obsidian Curator Report

## Summary
- generated_at: `2026-08-05T04:09:52`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `18`
- mana_status: `hold`

## Material Counts
- judgment_rule: `17`
- risk_signal: `1`

## Inbox Candidates
- `cur_3ff16a2748c5` `judgment_rule` この「サプライヤーの信頼性」という視点は、物件そのものの情報が不足している場合に、リース審査の判断を補強する重要な要素となります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_c1efcea6b8be` `judgment_rule` これらの情報が明確になれば、中古トラックのリース審査における物件評価の判断資産として、さらに洗練された基準を構築できると考えています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_167382b0b045` `judgment_rule` これらの情報が確認できれば、物件評価においてより前向きな判断ができますし、もし不足する場合は、追加の条件設定や残価設定の見直しが必要になる、という方向で考えられます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_de13588c66f2` `judgment_rule` そこで、もしディーラー系以外の販売業者からの物件をリース審査する際、具体的にどのようなメンテナンス記録や保証内容を確認することが、安心材料となるでしょうか / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_f450afb8b651` `judgment_rule` 保守が切れると物件価値が大きく下がるため、リース期間と保守期間の整合性は重要です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_6bea683e9a61` `judgment_rule` 物件評価の明確化: 中古物件の評価は難しいことが多いですが、このような明確な基準があれば、審査担当者も自信を持って物件評価を進められます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_cd8900f1a478` `judgment_rule` 不明確な点が多い場合や、高額な撤去費用、低い中古市場流動性が判明した場合は、リース料率へのリスク上乗せや、残価保証条件の見直し、または保留・否決寄りで判断します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`
- `cur_826a30789f62` `judgment_rule` 保守契約と撤去・搬出費用: リース期間中の保守契約の内容（費用、範囲、期間）と、リース終了時の撤去・搬出にかかる費用見積もりを詳細に確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`
- `cur_6c84e6b6a837` `judgment_rule` リース物件がこれらの法規制に適合しているか、将来的な法改正リスクがないかを確認します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_935a669de5c6` `judgment_rule` 医療機器のリースでは、通常の物件とは異なるいくつかの確認点があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
- `cur_e4f0dcab26e4` `judgment_rule` ③ 対応策 現時点での仮判断は、2番の改善は、シミュレーター機能が審査画面やリースインテリジェンス画面に統合され、表示されるようになったことである可能性が高いです / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`
- `cur_f775195edb24` `judgment_rule` 今回の医療機器リースで見るべき違和感は、薬機法上の規制と、それに伴う残価リスクの評価です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- なし

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
