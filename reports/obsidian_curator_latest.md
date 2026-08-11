# Obsidian Curator Report

## Summary
- generated_at: `2026-08-11T04:13:21`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `25`
- mana_status: `hold`

## Material Counts
- judgment_rule: `17`
- risk_signal: `8`

## Inbox Candidates
- `cur_a02167c02087` `judgment_rule` リース物件の戦略的位置づけ: リースする物件が、借手の物価高対策（例: コスト削減、生産性向上、新事業展開）にどのように貢献するのかを明確に評価し、その重要性を判断材料とします / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
- `cur_0e29aec9d17f` `judgment_rule` 特に、補助金前提の工作機械リースであれば、採択前の返済原資と未採択時の代替資金を分けて確認することが重要です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`
- `cur_989bcda6b66d` `judgment_rule` このような社会情勢の変化は、リース物件の残価価値や回収リスクにも影響を与えるため、私自身の判断軸を常に更新していくことの重要性を改めて認識しました / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
- `cur_f07ffa8714b5` `judgment_rule` 物件の汎用性と市場価値: 万が一の事態に備え、リース物件の汎用性や中古市場での換金性を再評価し、保全性を高める条件を検討します / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
- `cur_9a00d7193423` `judgment_rule` リース期間終了時の残価設定が適切か、中古市場での需要が維持されるか、また、万が一の際の物件回収・処分費用も考慮に入れる必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`
- `cur_0ff5af2ac2e8` `judgment_rule` 物件の汎用性と残価評価: 物件のメーカー、型番、仕様、導入時期、期待される稼働期間を確認し、中古市場での汎用性、流動性、想定される残価を再評価します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`
- `cur_659251a01067` `judgment_rule` 特に小規模な企業での倒産が増え、負債が小口化しているという報告は、リース審査において、単に財務数値を見るだけでなく、借手の事業環境や資金繰りの変化を、これまで以上に細やかに見極める必要性を感じさせ、深く考えさせられました / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
- `cur_20fde3a5d029` `judgment_rule` リース審査でどのような影響があるか / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
- `cur_14f5f992dd34` `judgment_rule` 現時点では、物価高による倒産増加は審査基準を厳格化する方向で、特に中小・零細企業や特定の業種に対しては、より詳細な財務・事業計画の確認が必要となるでしょう / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
- `cur_0ee55d60992d` `judgment_rule` 技術陳腐化リスクも考慮し、残価設定の妥当性や、万が一の際の物件換金性を詳細に評価する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`
- `cur_463d55edf081` `judgment_rule` キャッシュフローの確認: - 営業キャッシュフローが過去期比で20%以上減少している、またはマイナスに転落している場合は、資金繰りの悪化シグナルとして、リース料支払いの蓋然性を慎重に評価します / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
- `cur_9c543bde3b41` `risk_signal` 物件価値への影響: 倒産が増加すると、中古市場に物件が供給される可能性が高まり、物件の換金性や残価価値に影響を与える可能性があります / source=`Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`

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
- 採用する場合は candidate_id を data/curator_inbox_decisions.jsonl に {"candidate_id": "...", "decision": "adopt"} として追記し、scripts/promote_curator_inbox_candidates.py を実行する（Mana許可時のみpreviewへ反映、activeへは既存のpromote_canonical_judgment_rules.pyを人間が実行）。
