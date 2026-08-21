# Memory Engineering Report

- Generated: 2026-08-22T04:01:02
- Mode: `read_only_memory_engineering_observation`
- Guardrail: `no_memory_delete_no_promotion_no_prompt_or_scoring_change`

## Summary

- Write path records: 247
- Active canonical rules: 11
- Write amplification / active rule: 22.45
- Open human review records: 31
- Memory records: 765
- Recent memory usage: 529 events / 170 refs
- Maintenance status records: 144
- Contradiction candidates: 0
- Write policy metadata completion: 0.0
- Candidate / active pressure: 0.528
- Quarantine records: 147
- Sleeping active rules: 9

## Stanford Lens: Write Cost

- `judgment_materials_preview`: 22 records, 14 open review, ~22 tokens
- `autoresearch_candidates`: 83 records, 17 open review, ~83 tokens
- `reflection_action_candidates`: 10 records, 0 open review, ~1432 tokens
- `prediction_error_candidates`: 1 records, 0 open review, ~1 tokens
- `obsidian_memory_insight_candidates`: 123 records, 0 open review, ~123 tokens
- `canonical_preview`: 8 records
- Write policy required fields: `['importance', 'confidence', 'trust_level', 'provenance']`
- Write policy missing fields: `{'confidence': 217, 'importance': 239, 'provenance': 239, 'trust_level': 239}`

## Microsoft Lens: Utility Density

- Latest accepted preview: 1
- Promoted to active rules: 1
- Promotion rate: 1.0

## Anthropic Lens: Control

- Lifecycle inventory: `{'active': 608, 'candidate_or_review': 321, 'quarantine': 147, 'rejected_or_dismissed': 158, 'maintenance_or_forgetting_review': 153}`
- Utility KPIs: `{'checklist_review_rate': 0.0, 'field_feedback_coverage': 0.0, 'candidate_to_active_pressure': 0.528, 'quarantine_rate_in_experience_flywheel': 0.327}`
- Status counts: `{'active': 597, 'private': 24, 'stale': 144}`
- Type counts: `{'dialogue_memory': 41, 'factual_memory': 257, 'judgment_memory': 146, 'reflection_memory': 43, 'technical_memory': 262, 'value_memory': 16}`

### Forgetting Review Sample

- `mem_4de5f04f517736a1` technical_memory last_used=none source=MEMORY.md: [2026-07-07] AIチャット履歴が画面遷移で消える問題に対応。Cloud Run demo/readonly ではDB保存が失敗するため、`frontend/src/lib/chatLocalHistory.ts` を追加し、通常 `/chat` と右下 `Floati
- `mem_53df17aaa3f14f36` technical_memory last_used=none source=MEMORY.md: [2026-07-07] Cloud Run input sync was downloading events and writing summary notes, but `improvement_note` events were not materialized into
- `mem_5734bba9e52b2bb8` technical_memory last_used=none source=MEMORY.md: [2026-07-07] 結果登録画面の未登録一覧が Cloud Run版の審査入力を拾えていなかった。原因は `/api/cases/pending` が `past_cases.final_status='未登録'` だけを見ており、Cloud Run demo/readon
- `mem_a969441d5de90bf9` technical_memory last_used=none source=MEMORY.md: [2026-07-07] Audited all existing `data/cloudrun_inputs/*.jsonl` event types: `shion_memory_usage` 155, `improvement_note` 27, `rag_feedback
- `mem_ab1867025d0e2b55` factual_memory last_used=none source=MEMORY.md: [2026-07-07] The personal-memory extractor had incorrectly stored incomplete dog-name text as `Dog name: は`. Removed the bad confirmed fact 

## Forgetting Policy

- `raw_logs`: Keep as evidence; distill into fact/skill/judgment candidates before recall.
- `candidate_memory`: Hold for human review; do not inject into prompts until accepted/revised.
- `active_but_sleeping`: Do not delete immediately; ask for real-case feedback or move to hold if it stays unused.
- `quarantine`: Keep out of memory and prompts; review only as a failure/poisoning/noise signal.
- `contradiction`: Surface with dates and applicability; never auto-merge contradictory memories.
- Current pressure: `{'active_non_value_without_top_usage': 439, 'sleeping_active_rules': 9, 'review_active_rules': 0, 'experience_quarantine': 147}`

### Sleeping Rule Sample

- `cf61a9701fc8cc42` asset_life_and_residual: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- `b259411afb954d6d` business_plan_specificity: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- `a61f3a316a651126` conditional_approval_checks: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- `64e054542be673e4` demo_renewal_asset: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `a34492fe19a18e3a` demo_renewal_asset: 更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。

## Daily Review Focus

### 候補を少し採否する

- Why: 候補圧を下げ、active memory へ進めるものと捨てるものを分ける。
- `judgment_materials_preview::f07dcd715dadb336` judgment_rule: 物件の保全性と残価: 医療機器は技術陳腐化が早いため、中古市場での流通性、保守契約の有無、撤去・搬出費用を含めた残価設定の妥当性を確認します
- `judgment_materials_preview::ea6f87a2b966080f` risk_signal: ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください
- `judgment_materials_preview::e96b470222df47ed` risk_signal: 返済原資の確実性: 導入目的と連動した収益改善計画、または既存事業からの安定したキャッシュフローでリース料を賄えるか
- `judgment_materials_preview::d2bc70f15cdab5e4` judgment_rule: 稟議で残すべき一文 Qriskは低いが、導入目的が不明瞭なため、物件の必要性、返済原資の確実性、および医療機器特有の技術陳腐化リスクと残価設定の妥当性を、銀行紹介元と連携して確認することを条件に承認を検討する
- `judgment_materials_preview::c5da5e67676b6cf2` risk_signal: 過去事例の「ダックス物流」と比較すると、あちらもスコア帯が近く、銀行支援が弱い中で資料不足で見送りとなりました

### quarantine が多い抽出元を弱める

- Why: 隔離候補は学習材料ではなく、抽出条件のノイズを示す。
- Count: 147
- Sample count: 30
- Sample by source: `{'shion_experience': 30}`
- Review hint: Do not promote these. Use samples to tighten extraction gates or leave as evidence.
- `xfly_c9d7d9143d1f4104` shion_experience insufficient feedback or context: 私の応答の前置きが冗長だった点してあった
- `xfly_3e0006f3c4936173` shion_experience insufficient feedback or context: 修正した
- `xfly_bb027728cd05fc58` shion_experience insufficient feedback or context: 何が気になる点ある？
- `xfly_52add9e0db7d34de` shion_experience insufficient feedback or context: 【審査分析画面からの紫苑レビュー依頼】 この案件を、審査担当者の横にいる紫苑としてレビューしてください。 出力は短く、次の4項目でお願いします。 1. 紫苑の第一印象 2. 数字だけでは見落としそうな違和感 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 前提…
- `xfly_45a338a8f6f07aa7` shion_experience insufficient feedback or context: 君の目的は？

### sleeping active rule を次案件で試すか保留する

- Why: active でも実利用フィードバックがなければ、判断資産として効いているか不明。
- `cf61a9701fc8cc42` asset_life_and_residual: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- `b259411afb954d6d` business_plan_specificity: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- `a61f3a316a651126` conditional_approval_checks: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- `64e054542be673e4` demo_renewal_asset: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `a34492fe19a18e3a` demo_renewal_asset: 更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。


## Nvidia Lens: Retrieval Pressure

- Notes: 2536 / Edges: 5463
- Estimated raw tokens: 2691696
- Estimated index tokens: 238763
- Estimated token reduction: 0.9113

## Recommended Next Checks

- `memory_entry_metadata_gate` (write_policy): 候補記憶に importance/confidence/trust_level/provenance が不足している。昇格前に共通メタを補う。
- `candidate_budget_review` (write_path): active判断基準1件あたりの候補量が多い。抽出条件・重複統合・候補上限を見直す。
- `human_review_batch` (control): 人間レビュー待ちが溜まっている。自動昇格せず、上位候補だけ短時間で採否する。
- `candidate_to_active_pressure_review` (utility_density): active記憶に対して候補・評価待ちが重い。候補生成より採否・圧縮・却下の運用を優先する。
- `quarantine_sample_review` (forgetting): 経験フライホイールに隔離候補がある。学習材料ではなく、抽出条件のノイズとして扱う。
- `sleeping_active_rule_feedback` (utility_density): active判断資産に実利用フィードバック未記録のものがある。削除前に次案件で効いたか確認する。
- `stale_or_revised_cleanup` (forgetting): stale/revised/deprecated の記憶がある。削除ではなく、隔離・統合・改訂を確認する。
