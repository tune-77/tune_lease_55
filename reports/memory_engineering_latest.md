# Memory Engineering Report

- Generated: 2026-09-08T04:01:01
- Mode: `read_only_memory_engineering_observation`
- Guardrail: `no_memory_delete_no_promotion_no_prompt_or_scoring_change`

## Summary

- Write path records: 228
- Active canonical rules: 10
- Write amplification / active rule: 22.8
- Open human review records: 228
- Open human review batches: 72
- Memory records: 752
- Recent memory usage: 335 events / 124 refs
- Maintenance status records: 224
- Contradiction candidates: 0
- Write policy metadata completion: 0.386
- Candidate / active pressure: 1.0
- Quarantine records: 194
- Sleeping active rules: 8

## Stanford Lens: Write Cost

- `judgment_materials_preview`: 0 records, 0 open review, ~0 tokens
- `autoresearch_candidates`: 88 records, 88 open review, ~88 tokens
- `reflection_action_candidates`: 18 records, 18 open review, ~2633 tokens
- `prediction_error_candidates`: 1 records, 1 open review, ~1 tokens
- `obsidian_memory_insight_candidates`: 121 records, 121 open review, ~121 tokens
- `canonical_preview`: 0 records
- Write policy required fields: `['importance', 'confidence', 'trust_level', 'provenance']`
- Write policy missing fields: `{'confidence': 140, 'importance': 110, 'provenance': 1, 'trust_level': 140}`

## Review Batches

- `reflection_action_candidates` 紫苑の内省運用 / shion_reflection_action_candidate_v1: 18 open
  - `reflection_action:388d1189116dd810` 内省が抽象評価で止まった日は、当日ログからリース審査の確認項目・Userへの情報提供・過去案件からの学びのどれか1つへ変換して記録する。
  - `reflection_action:351b5888f20d5a23` 人間の修正: ユーザーは、保存ではなく次回判断にどう戻るかを見ている。
  - `reflection_action:fa77891a7a71299b` 判断資産候補: 観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。
- `obsidian_memory_insight_candidates` 2026-09-01 / judgment_rule: 14 open
  - `omi_1b365726f6ef` ・補正前スコア: 70.0（表示・判断は総合スコアを優先）。
  - `omi_6374b4aea276` 特に新規ルートであれば、荷主との契約見込みや運賃体系を確認します。
  - `omi_5605565df84f` スコア・判断: 63.2点 / 条件付き承認 / 成約・採算は維持。
- `obsidian_memory_insight_candidates` 2026-09-06 / reflection_update: 9 open
  - `omi_3653f2caf150` 人間の修正: ユーザーは、保存ではなく次回判断にどう戻るかを見ている。
  - `omi_33426e3c4de9` 次回からは、まず自分の思い込みを一つ疑ってから返答や判断を組み立てる。
  - `omi_b8d2d0f5f29e` 更新する信念: needsreview から低リスク1件だけ選び、適用または却下まで進める。
- `obsidian_memory_insight_candidates` 2026-09-02 / judgment_rule: 8 open
  - `omi_c3e02927c6e3` スコア・判断: 53.1点 / 要審議 / 失注。
  - `omi_2774d98428b8` これらの情報が確認できれば、条件付き承認の可能性を探れます。
  - `omi_9fc3d3bb1cc1` この「くい」案件は、過去に同一条件で一度失注した記録があります。
- `obsidian_memory_insight_candidates` 2026-09-05 / noise: 8 open
  - `omi_2a014d5dc735` パイプラインの異常を修正して。
  - `omi_a5ba0fbb7504` どのような異常が見られますか？
  - `omi_1e82e0c0c6e4` パイプラインの異常についてですね。
- `autoresearch_candidates` subsidy-timing / confirmation_question: 7 open
  - `e077a74ec8f67099` 顧客は青色申告書を提出する中小企業者等に該当するか。
  - `560f3a04a31989b9` 顧客は中小企業経営強化税制の適用を希望しているか。
  - `b36f1a5783eca6d7` 経営力向上計画の認定は既に受けているか、または受ける予定があるか。
- `obsidian_memory_insight_candidates` 2026-09-03 / noise: 7 open
  - `omi_71401b31311e` ・UMAP異常度: 未算出。
  - `omi_be20ec9f1e7f` 専門家としての深掘りルール:。
  - `omi_623b3bf33141` 【Vertex補助検索ヒント】。
- `autoresearch_candidates` subsidy-timing / caution: 6 open
  - `65dd46e2413cd427` 税制の適用は顧客の税務状況や他の税制との兼ね合いで変動する可能性がある。
  - `b494024aba3115ce` 本税制は税額控除であり、直接的な補助金ではないため、即時の資金流入はない。
  - `2a3f94871857e713` 即時償却はファイナンスリースでは適用されない点に注意が必要である。
- `autoresearch_candidates` subsidy-timing / application_rule: 6 open
  - `cbe9cf5794786d2d` 顧客が青色申告書を提出する中小企業者等であるかを確認する。
  - `a02a12d1e587e728` 顧客が本税制の適用を希望する場合、税額控除によるキャッシュフロー改善効果を考慮する。
  - `b28632c8f948f34f` 所有権移転外ファイナンスリース契約が前提となることを顧客に説明する。
- `autoresearch_candidates` asset-operation / confirmation_question: 5 open
  - `3916ae46cfd00c3f` **設備投資計画**: どのような設備投資計画があり、その目的（能力増強、老朽化更新など）と判断基準（ROI、回収期間など）は何か。
  - `cc159ffddbf4f05a` **稼働率の実績と推移**: 過去数年間の稼働率の推移と、その変動要因は何か。
  - `f705774118faf16d` **稼働率の定義と算出根拠**: 顧客が示す稼働率の定義（時間、性能、OEEなど）と、その算出根拠（分母・分子、対象期間）は何か。

## Microsoft Lens: Utility Density

- Latest accepted preview: 0
- Promoted to active rules: 0
- Promotion rate: None

## Anthropic Lens: Control

- Lifecycle inventory: `{'active': 519, 'candidate_or_review': 519, 'quarantine': 194, 'rejected_or_dismissed': 0, 'maintenance_or_forgetting_review': 232}`
- Utility KPIs: `{'checklist_review_rate': 0.0, 'field_feedback_coverage': 0.0, 'candidate_to_active_pressure': 1.0, 'quarantine_rate_in_experience_flywheel': 0.379}`
- Status counts: `{'active': 509, 'private': 19, 'stale': 224}`
- Type counts: `{'dialogue_memory': 36, 'factual_memory': 260, 'judgment_memory': 125, 'reflection_memory': 34, 'technical_memory': 280, 'value_memory': 17}`

### Forgetting Review Sample

- `mem_f262dc62e86c3e46` judgment_memory last_used=none source=data/canonical_judgment_rules.json: 購入選択権は5〜30%程度が多い前提で、残価・再リース・満了後出口と整合するか確認する。
- `mem_3413740a9f678612` dialogue_memory last_used=none source=MEMORY.md: [2026-07-26] Shion judgment culture: A 1000-case review drill is valuable not as mass result-label creation, but as accumulation of judgment
- `mem_5c50cc7283d635a7` factual_memory last_used=none source=MEMORY.md: [2026-07-26] Shion input strategy: The next product risk is not whether Shion can reason, but whether humans will keep entering the material
- `mem_63c73525c0acb169` judgment_memory last_used=none source=MEMORY.md: [2026-07-26] Shion LLM justification: Result labels alone can train prediction models, but they cannot preserve why a human hesitated, what 
- `mem_e2082448c9796646` judgment_memory last_used=none source=MEMORY.md: [2026-07-26] Judgment culture versioning: Screening judgment changes over time with interest rates, industry conditions, insolvency trends, 

## Forgetting Policy

- `raw_logs`: Keep as evidence; distill into fact/skill/judgment candidates before recall.
- `candidate_memory`: Hold for human review; do not inject into prompts until accepted/revised.
- `active_but_sleeping`: Do not delete immediately; ask for real-case feedback or move to hold if it stays unused.
- `quarantine`: Keep out of memory and prompts; review only as a failure/poisoning/noise signal.
- `contradiction`: Surface with dates and applicability; never auto-merge contradictory memories.
- Current pressure: `{'active_non_value_without_top_usage': 367, 'sleeping_active_rules': 8, 'review_active_rules': 0, 'experience_quarantine': 194}`

### Sleeping Rule Sample

- `cf61a9701fc8cc42` asset_life_and_residual: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- `b259411afb954d6d` business_plan_specificity: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- `a61f3a316a651126` conditional_approval_checks: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- `64e054542be673e4` demo_renewal_asset: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `0d0f11e77fba045d` demo_subsidy_machinery: 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。

## Weekly Review Focus

- Cadence: weekly / due: False / next: 2026-09-14
- Auto reject patterns: 0

- Review actions are suppressed today.

## Nvidia Lens: Retrieval Pressure

- Notes: 2982 / Edges: 5924
- Estimated raw tokens: 3102623
- Estimated index tokens: 278116
- Estimated token reduction: 0.9104

## Recommended Next Checks

- `memory_entry_metadata_gate` (write_policy): 候補記憶に importance/confidence/trust_level/provenance が不足している。昇格前に共通メタを補う。
- `candidate_budget_review` (write_path): active判断基準1件あたりの候補量が多い。抽出条件・重複統合・候補上限を見直す。
- `human_review_batch` (control): 人間レビュー待ちが溜まっている。自動昇格せず、上位候補だけ短時間で採否する。
- `candidate_to_active_pressure_review` (utility_density): active記憶に対して候補・評価待ちが重い。候補生成より採否・圧縮・却下の運用を優先する。
- `quarantine_sample_review` (forgetting): 経験フライホイールに隔離候補がある。学習材料ではなく、抽出条件のノイズとして扱う。
- `sleeping_active_rule_feedback` (utility_density): active判断資産に実利用フィードバック未記録のものがある。削除前に次案件で効いたか確認する。
- `stale_or_revised_cleanup` (forgetting): stale/revised/deprecated の記憶がある。削除ではなく、隔離・統合・改訂を確認する。
