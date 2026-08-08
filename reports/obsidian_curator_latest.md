# Obsidian Curator Report

## Summary
- generated_at: `2026-08-09T04:14:52`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `14`
- mana_status: `hold`

## Material Counts
- judgment_rule: `10`
- risk_signal: `4`

## Inbox Candidates
- `cur_667217d91080` `risk_signal` もし補助金活用を前提としている場合、採択の確実性、入金時期、未採択時の返済計画、補助金返還リスクなどを慎重に確認する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_58f875290a49` `judgment_rule` 「ソフトリース」という言葉は、一般的にリース期間が物件の法定耐用年数よりも短い契約を指すことが多いですが、その契約形態にかかわらず、リース期間満了後に物件の利用を継続したい場合は、再リース契約を結ぶことができます / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_2781fc270f8e` `judgment_rule` ファイナンスリースであっても、借手の事業計画における物件の陳腐化影響を確認する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_c9d35cd27ab6` `judgment_rule` リース期間満了時の残価設定にも慎重な検討が必要です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_7a51945cb2e1` `judgment_rule` 借手の返済能力: リース料率の上昇は、借手の月々の支払い負担を増やすため、返済能力をより厳しく評価する必要があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_f78edfe4cdcb` `judgment_rule` 現時点では、金利上昇はリース料率の上昇を通じて借手の返済能力に影響を与え、審査をより慎重にする必要があると考えています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_d20e410fa04f` `risk_signal` 補助金状況: もし補助金を申請している場合、申請状況、交付決定の見込み、および未採択時の資金繰り計画 / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_a18185fa79c6` `risk_signal` このような状況では、返済原資と設備稼働開始の確認を優先すべきだと考えています / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_06d167684d6e` `risk_signal` 物件の陳腐化リスク: 「IT・OA機器」は技術革新が速く、陳腐化リスクが高い物件です / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_42e786d8f2cf` `judgment_rule` これにより、借手にとってはリース料の負担が増え、設備投資の判断に影響を与える可能性があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_4e6dfe3c05cf` `judgment_rule` ただし、これはあくまで一般的な法定耐用年数であり、実際のリース期間や税務上の取り扱いは、個別の契約内容や最新の税制によって変わる可能性があります / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
- `cur_af99637f17ba` `judgment_rule` 「法人車両としてBMWをリースする理由」について、私のリース審査専門WikiやObsidian Vaultの業務記録からは、業務部の具体的な視点での直接的な情報は見つかりませんでした / source=`Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-07.md`

## Duplicate Clusters
- なし

## Search Terms To Add
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
