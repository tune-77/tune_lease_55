# Memory Engineering Report

- Generated: 2026-08-24T04:01:04
- Mode: `read_only_memory_engineering_observation`
- Guardrail: `no_memory_delete_no_promotion_no_prompt_or_scoring_change`

## Summary

- Write path records: 234
- Active canonical rules: 10
- Write amplification / active rule: 23.4
- Open human review records: 231
- Memory records: 768
- Recent memory usage: 515 events / 159 refs
- Maintenance status records: 152
- Contradiction candidates: 0
- Write policy metadata completion: 0.0
- Candidate / active pressure: 0.853
- Quarantine records: 153
- Sleeping active rules: 8

## Stanford Lens: Write Cost

- `judgment_materials_preview`: 12 records, 12 open review, ~12 tokens
- `autoresearch_candidates`: 83 records, 83 open review, ~83 tokens
- `reflection_action_candidates`: 12 records, 12 open review, ~1775 tokens
- `prediction_error_candidates`: 1 records, 1 open review, ~1 tokens
- `obsidian_memory_insight_candidates`: 123 records, 123 open review, ~123 tokens
- `canonical_preview`: 3 records
- Write policy required fields: `['importance', 'confidence', 'trust_level', 'provenance']`
- Write policy missing fields: `{'confidence': 219, 'importance': 231, 'provenance': 231, 'trust_level': 231}`

## Microsoft Lens: Utility Density

- Latest accepted preview: 0
- Promoted to active rules: 0
- Promotion rate: None

## Anthropic Lens: Control

- Lifecycle inventory: `{'active': 607, 'candidate_or_review': 518, 'quarantine': 153, 'rejected_or_dismissed': 0, 'maintenance_or_forgetting_review': 160}`
- Utility KPIs: `{'checklist_review_rate': 0.0, 'field_feedback_coverage': 0.0, 'candidate_to_active_pressure': 0.853, 'quarantine_rate_in_experience_flywheel': 0.334}`
- Status counts: `{'active': 597, 'private': 19, 'stale': 152}`
- Type counts: `{'dialogue_memory': 40, 'factual_memory': 258, 'judgment_memory': 144, 'reflection_memory': 34, 'technical_memory': 277, 'value_memory': 15}`

### Forgetting Review Sample

- `mem_f1ccd2dde7b2d455` factual_memory last_used=none source=MEMORY.md: [2026-07-11] Improvement report cleanup should not rely only on REV IDs. IDs can be reused or drift; title/detail canonical matching is need
- `mem_13141ccd88433551` technical_memory last_used=none source=MEMORY.md: [2026-07-12] Updated `scripts/build_shion_memory_index.py` so active canonical judgment rules are included as `judgment_memory` records. Reg
- `mem_43f592eb7051c9dc` technical_memory last_used=none source=MEMORY.md: [2026-07-12] Added a read-only Judgment Materials Preview sidecar for Obsidian conversation logs. It extracts reusable `judgment_rule`, `ris
- `mem_5176fbd9f06fa298` technical_memory last_used=none source=MEMORY.md: [2026-07-12] Cloud Run conversation logs and local Obsidian Dialogue notes are separate surfaces unless the Mac sync explicitly mirrors them
- `mem_6eaf54cccf8bad23` technical_memory last_used=none source=MEMORY.md: [2026-07-12] Added `scripts/promote_canonical_judgment_rules.py` to promote only `accepted_preview` canonical rules into `data/canonical_jud

## Forgetting Policy

- `raw_logs`: Keep as evidence; distill into fact/skill/judgment candidates before recall.
- `candidate_memory`: Hold for human review; do not inject into prompts until accepted/revised.
- `active_but_sleeping`: Do not delete immediately; ask for real-case feedback or move to hold if it stays unused.
- `quarantine`: Keep out of memory and prompts; review only as a failure/poisoning/noise signal.
- `contradiction`: Surface with dates and applicability; never auto-merge contradictory memories.
- Current pressure: `{'active_non_value_without_top_usage': 442, 'sleeping_active_rules': 8, 'review_active_rules': 0, 'experience_quarantine': 153}`

### Sleeping Rule Sample

- `cf61a9701fc8cc42` asset_life_and_residual: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- `b259411afb954d6d` business_plan_specificity: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- `a61f3a316a651126` conditional_approval_checks: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- `64e054542be673e4` demo_renewal_asset: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `0d0f11e77fba045d` demo_subsidy_machinery: 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。

## Daily Review Focus

### 候補を少し採否する

- Why: 候補圧を下げ、active memory へ進めるものと捨てるものを分ける。
- `judgment_materials_preview::afcfca8390ce5061` judgment_rule: 「文明年代機」という要素の表示について - リース審査 はい、承知いたしました
- `judgment_materials_preview::7e52be1479a28abd` judgment_rule: 過去案件数: 2192件 - 審査記録数: 2109件 - モデル精度 (AUC): アンサンブルモデルで 0.82、最新の単一モデルで 0.74 です
- `judgment_materials_preview::67d4eb6ecaf86a0e` judgment_rule: 「文明年代機」という要素の表示について リース審査で参照していた「文明年代記」の情報源が削除され、利用不可になった件について スコア80-100帯の成約率逆転：モデルキャリブレーションの見直しについて これらの候補について、さらに詳しい情報が必要でしたら、お申し付けください
- `reflection_action_candidates::reflection_action:de5c0b52edf6fd80` 内省アクション: リース審査業務の改善候補として、紫苑の統合: 私は、直近3日で一番繰り返している論点: リース審査業務の改善候補として、紫苑の統合: 私は、直近3日で一番繰り返している論点を1つ選び、次回の判断基準に昇格するという形で統合し、次回は一つの反省文ではなく声ごとの採用・保留・棄却を残す。
- `reflection_action_candidates::reflection_action:7acf38cd98252ece` 内省アクション: リース審査業務の改善候補として、紫苑の統合: 私は、ハッカソンでは、派手さよりも実務: リース審査業務の改善候補として、紫苑の統合: 私は、ハッカソンでは、派手さよりも実務判断がどう変わるかを内省に戻すという形で統合し、次回は一つの反省文ではなく声ごとの採用・保留・棄却を残す。

### quarantine が多い抽出元を弱める

- Why: 隔離候補は学習材料ではなく、抽出条件のノイズを示す。
- Count: 153
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
- `0d0f11e77fba045d` demo_subsidy_machinery: 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。


## Nvidia Lens: Retrieval Pressure

- Notes: 2592 / Edges: 5513
- Estimated raw tokens: 2766695
- Estimated index tokens: 243783
- Estimated token reduction: 0.9119

## Recommended Next Checks

- `memory_entry_metadata_gate` (write_policy): 候補記憶に importance/confidence/trust_level/provenance が不足している。昇格前に共通メタを補う。
- `candidate_budget_review` (write_path): active判断基準1件あたりの候補量が多い。抽出条件・重複統合・候補上限を見直す。
- `human_review_batch` (control): 人間レビュー待ちが溜まっている。自動昇格せず、上位候補だけ短時間で採否する。
- `candidate_to_active_pressure_review` (utility_density): active記憶に対して候補・評価待ちが重い。候補生成より採否・圧縮・却下の運用を優先する。
- `quarantine_sample_review` (forgetting): 経験フライホイールに隔離候補がある。学習材料ではなく、抽出条件のノイズとして扱う。
- `sleeping_active_rule_feedback` (utility_density): active判断資産に実利用フィードバック未記録のものがある。削除前に次案件で効いたか確認する。
- `stale_or_revised_cleanup` (forgetting): stale/revised/deprecated の記憶がある。削除ではなく、隔離・統合・改訂を確認する。
