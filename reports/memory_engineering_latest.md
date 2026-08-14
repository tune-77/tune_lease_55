# Memory Engineering Report

- Generated: 2026-08-14T16:53:18
- Mode: `read_only_memory_engineering_observation`
- Guardrail: `no_memory_delete_no_promotion_no_prompt_or_scoring_change`

## Summary

- Write path records: 280
- Active canonical rules: 11
- Write amplification / active rule: 25.45
- Open human review records: 154
- Memory records: 773
- Recent memory usage: 692 events / 185 refs
- Maintenance status records: 0
- Contradiction candidates: 0
- Write policy metadata completion: 0.0
- Candidate / active pressure: 0.575
- Quarantine records: 140
- Sleeping active rules: 9

## Stanford Lens: Write Cost

- `judgment_materials_preview`: 23 records, 0 open review, ~23 tokens
- `autoresearch_candidates`: 122 records, 31 open review, ~122 tokens
- `reflection_action_candidates`: 4 records, 0 open review, ~493 tokens
- `prediction_error_candidates`: 1 records, 0 open review, ~1 tokens
- `obsidian_memory_insight_candidates`: 123 records, 123 open review, ~123 tokens
- `canonical_preview`: 7 records
- Write policy required fields: `['importance', 'confidence', 'trust_level', 'provenance']`
- Write policy missing fields: `{'confidence': 250, 'importance': 273, 'provenance': 273, 'trust_level': 273}`

## Microsoft Lens: Utility Density

- Latest accepted preview: 2
- Promoted to active rules: 2
- Promotion rate: 1.0

## Anthropic Lens: Control

- Lifecycle inventory: `{'active': 762, 'candidate_or_review': 438, 'quarantine': 140, 'rejected_or_dismissed': 37, 'maintenance_or_forgetting_review': 9}`
- Utility KPIs: `{'checklist_review_rate': 0.0, 'field_feedback_coverage': 0.0, 'candidate_to_active_pressure': 0.575, 'quarantine_rate_in_experience_flywheel': 0.321}`
- Status counts: `{'active': 751, 'private': 22}`
- Type counts: `{'dialogue_memory': 57, 'factual_memory': 252, 'judgment_memory': 150, 'reflection_memory': 43, 'technical_memory': 254, 'value_memory': 17}`

### Forgetting Review Sample

- `mem_0091b5c5e75261b5` technical_memory last_used=none source=MEMORY.md: **Active Script**: `lease_logic_sumaho3.py` (Replaced `lease_logic.py` as the main driver).
- `mem_08782ec7b9ee36c9` factual_memory last_used=none source=knowledge_base/okf_lease_concepts/README.md: Related: [Statutory useful life](rules/statutory_useful_life.md)
- `mem_0c86c77f2c6a0258` factual_memory last_used=none source=knowledge_base/okf_lease_concepts/assets/machine_tool_resale_risk.md: 根拠: 中古市場がある物件でも、実際の換価額は「売れる価格」から撤去・搬出・再整備コストを引いた金額になる。
- `mem_0de35462b2e34771` technical_memory last_used=none source=MEMORY.md: [2026-06-28] Cloud Run bundleに日次知性などの生成JSONを含める時は、`.dockerignore` / `.gcloudignore` の `reports` 除外に注意する。`reports/` ではなく `.cloudrun_bundle/ob
- `mem_13c326dd9999e28b` factual_memory last_used=none source=MEMORY.md: QCL は本番導線から外した。`train_quantum.py` の QCL CLI/ helper を削除し、`quantum_qcl.py` と生成レポートも除去した。

## Forgetting Policy

- `raw_logs`: Keep as evidence; distill into fact/skill/judgment candidates before recall.
- `candidate_memory`: Hold for human review; do not inject into prompts until accepted/revised.
- `active_but_sleeping`: Do not delete immediately; ask for real-case feedback or move to hold if it stays unused.
- `quarantine`: Keep out of memory and prompts; review only as a failure/poisoning/noise signal.
- `contradiction`: Surface with dates and applicability; never auto-merge contradictory memories.
- Current pressure: `{'active_non_value_without_top_usage': 593, 'sleeping_active_rules': 9, 'review_active_rules': 0, 'experience_quarantine': 140}`

### Sleeping Rule Sample

- `cf61a9701fc8cc42` asset_life_and_residual: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- `b259411afb954d6d` business_plan_specificity: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- `a61f3a316a651126` conditional_approval_checks: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- `64e054542be673e4` demo_renewal_asset: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `a34492fe19a18e3a` demo_renewal_asset: 更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。

## Daily Review Focus

### 候補を少し採否する

- Why: 候補圧を下げ、active memory へ進めるものと捨てるものを分ける。
- `autoresearch_candidates::05115599bc4c4125` 設備稼働率・保守・更新投資の確認方法: 予知保全の費用対効果（ROI）の算出根拠や実績は、未確認の推論に基づく場合があるため、慎重な確認が必要です。
- `autoresearch_candidates::0243bb8ff0f8dcec` 設備稼働率・保守・更新投資の確認方法: 保守記録（点検・修理・部品交換履歴）は整備されていますか。また、保守担当者の専門性や外部委託の状況はどうですか。
- `autoresearch_candidates::ef8d448fa50f2468` 契約・所有権・検収・詐欺リスク: 法人顧客の高額物件では、動産譲渡登記の有無を確認し、二重譲渡リスクを評価します。必要に応じて、リース会社が譲受人として動産譲渡登記を行うことを承認条件とします。
- `autoresearch_candidates::ecbc932c4c9a090a` 契約・所有権・検収・詐欺リスク: リース契約の種類（ファイナンス/オペレーティング、所有権移転の有無）を契約書で確認します。会計処理上の分類だけでなく、契約書上の所有権の帰属、危険負担、中途解約条項などを詳細に確認します。
- `autoresearch_candidates::eb6dc1184738fc12` 契約・所有権・検収・詐欺リスク: リースバック案件で、物件評価額と売却価格が著しく乖離している、または顧客が融資と誤認している場合、詐欺の可能性を疑います。

### quarantine が多い抽出元を弱める

- Why: 隔離候補は学習材料ではなく、抽出条件のノイズを示す。
- Count: 140
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

- Notes: 2319 / Edges: 3218
- Estimated raw tokens: 2435318
- Estimated index tokens: 211772
- Estimated token reduction: 0.913

## Recommended Next Checks

- `memory_entry_metadata_gate` (write_policy): 候補記憶に importance/confidence/trust_level/provenance が不足している。昇格前に共通メタを補う。
- `candidate_budget_review` (write_path): active判断基準1件あたりの候補量が多い。抽出条件・重複統合・候補上限を見直す。
- `human_review_batch` (control): 人間レビュー待ちが溜まっている。自動昇格せず、上位候補だけ短時間で採否する。
- `candidate_to_active_pressure_review` (utility_density): active記憶に対して候補・評価待ちが重い。候補生成より採否・圧縮・却下の運用を優先する。
- `quarantine_sample_review` (forgetting): 経験フライホイールに隔離候補がある。学習材料ではなく、抽出条件のノイズとして扱う。
- `sleeping_active_rule_feedback` (utility_density): active判断資産に実利用フィードバック未記録のものがある。削除前に次案件で効いたか確認する。
