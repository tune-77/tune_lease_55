# Obsidian Curator Report

## Summary
- generated_at: `2026-07-15T08:29:20`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `51`
- mana_status: `allow`

## Material Counts
- judgment_rule: `32`
- risk_signal: `19`

## Inbox Candidates
- `judgment_rule` リース審査においては、この法定耐用年数だけでなく、顧客の実際の使用計画、物件の経済的寿命、そしてリース満了後の再販価値や再リース可能性も合わせて評価し、最適なリース期間を検討することが重要になります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` リース審査においては、その数字をそのまま使うのではなく、物件の実際の稼働状況、メンテナンス履歴、中古市場での需要、そしてリース期間終了後の残価設定にどう影響するか、という「判断資産」としての側面を重視することが大切です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` 推論 法定耐用年数は、税法上の減価償却期間を定めるもので、リース期間の設定や残価評価の目安にはなりますが、リース審査専門Wikiには物件ごとの詳細なリストは置かれていないようです / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` 物件の保全性と換金性: - 中古市場価値と再リース余地: リース期間終了時の物件の中古価値を再評価し、再リースや売却による残価回収の蓋然性を確認する / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` 物件・事業計画の特性: - 稼働率と保守費: 特に車両や建設機械の場合、稼働率、走行距離、保守費の実態を確認し、リース期間終了時の中古価値や再リース余地を評価します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `risk_signal` 最終的に、「この案件は、補助金未採択時の代替返済計画の提示を条件に、承認可能」という、リスクと保全策が明確に結びついた一つの結論へと収束します / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`
- `risk_signal` これは、財務数値だけでは捉えきれない、新規事業計画の不確実性や物件の換金性といった非財務リスクが極めて高いことを示唆しています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` これはリース期間設定の重要な基礎情報です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` 返済原資は受注増による売上増加、物件価値は中古流通性を鑑み、銀行支援の確認と受注根拠の明確化を条件に承認 / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` これは、リース物件の残価価値を評価する上で非常に重要な視点です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
- `judgment_rule` これは、法定耐用年数が税務上の概念であり、リース審査においては、物件の経済的耐用年数や中古市場価値、貴社の運用期間といった「実際の価値と利用期間」がより重視されるためかもしれません / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`
- `risk_signal` 受注増に伴う更新投資で、返済原資と物件用途の説明がしやすい / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`

## Duplicate Clusters
- なし

## Search Terms To Add
- メイン行
- 再販価値
- 採択前
- 支援依頼書
- 新店舗

## Related Gaps
- `Projects/tune_lease_55/検索語インデックス.md` related_section_missing missing=

## Mana Review Items
- なし

## After Hackathon Only
- Obsidianディレクトリ再編
- GCS Vault include/exclude変更
- accepted判断資産のactive store連携
- 判断資産レビューUI

## Next Safe Step
- Inbox候補を人間が採用・修正・却下・後回しに分類する。
- このレポート自体はObsidian本文、RAG、Cloud Run、active storeへ接続しない。
