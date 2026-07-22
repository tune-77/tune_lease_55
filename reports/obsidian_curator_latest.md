# Obsidian Curator Report

## Summary
- generated_at: `2026-07-22T04:08:34`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `32`
- mana_status: `watch`

## Material Counts
- judgment_rule: `26`
- risk_signal: `6`

## Inbox Candidates
- `judgment_rule` 審査では、補助金を返済原資として過信せず、採択前・未採択時・入金遅延時でもリース料を払えるかを見る / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `judgment_rule` 塾の場合、物件の選定においては、 - 陳腐化リスク: IT機器などは技術革新が早いため、リース期間の設定が重要です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-20.md`
- `judgment_rule` 物件の規模や種類によって、最適なリース期間や条件も変わってきますので、具体的なご相談があれば、詳細を伺うことも可能です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-20.md`
- `judgment_rule` この期間、皆様からの貴重なご意見を伺い、リース審査の判断資産としてシステムへ還元できるよう、引き続き丁重に対応させていただきます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `judgment_rule` この期間、皆様の貴重なご意見を伺い、リース審査の判断資産としてシステムへ還元できるよう、丁重に対応させていただきます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `judgment_rule` 条件付き承認にするなら必要な確認 - 導入目的である「既存受注の増加」について、具体的な顧客名、契約期間、金額などの根拠資料を確認してください / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `risk_signal` 受注増に伴う更新投資で、返済原資と物件用途の説明がしやすい / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `risk_signal` 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `risk_signal` 数字だけでは見落としそうな違和感 スコアが59.0に留まっているのは、営業メモにある「受注増」の具体的な根拠や、それによる返済原資の確実性がまだ明確でないためかもしれません / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `judgment_rule` この工作機械案件、補助金採択前だけどどう見る / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `judgment_rule` 【今回使う判断資産】 次の判断資産は、過去の会話・評価・結果から代表ルール化されたものです / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
- `judgment_rule` 私はリース審査システムの自己モデルとして‥前置きが長い / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`

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
