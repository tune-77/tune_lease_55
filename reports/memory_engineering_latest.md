# Memory Engineering Report

- Generated: 2026-08-25T08:26:47
- Mode: `read_only_memory_engineering_observation`
- Guardrail: `no_memory_delete_no_promotion_no_prompt_or_scoring_change`

## Summary

- Write path records: 243
- Active canonical rules: 10
- Write amplification / active rule: 24.3
- Open human review records: 236
- Open human review batches: 70
- Memory records: 764
- Recent memory usage: 531 events / 163 refs
- Maintenance status records: 0
- Contradiction candidates: 0
- Write policy metadata completion: 0.411
- Candidate / active pressure: 0.701
- Quarantine records: 174
- Sleeping active rules: 8

## Stanford Lens: Write Cost

- `judgment_materials_preview`: 16 records, 16 open review, ~16 tokens
- `autoresearch_candidates`: 81 records, 81 open review, ~81 tokens
- `reflection_action_candidates`: 12 records, 12 open review, ~1775 tokens
- `prediction_error_candidates`: 1 records, 1 open review, ~1 tokens
- `obsidian_memory_insight_candidates`: 126 records, 126 open review, ~126 tokens
- `canonical_preview`: 7 records
- Write policy required fields: `['importance', 'confidence', 'trust_level', 'provenance']`
- Write policy missing fields: `{'confidence': 139, 'importance': 109, 'provenance': 1, 'trust_level': 139}`

## Review Batches

- `reflection_action_candidates` 紫苑の内省運用 / shion_reflection_action_candidate_v1: 12 open
  - `reflection_action:388d1189116dd810` 内省が抽象評価で止まった日は、当日ログからリース審査の確認項目・Userへの情報提供・過去案件からの学びのどれか1つへ変換して記録する。
  - `reflection_action:351b5888f20d5a23` 人間の修正: ユーザーは、保存ではなく次回判断にどう戻るかを見ている。
  - `reflection_action:fa77891a7a71299b` 判断資産候補: 観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。
- `judgment_materials_preview` lease_screening / judgment_rule: 11 open
  - `afcfca8390ce5061` 「文明年代機」という要素の表示について - リース審査 はい、承知いたしました
  - `67d4eb6ecaf86a0e` 「文明年代機」という要素の表示について リース審査で参照していた「文明年代記」の情報源が削除され、利用不可になった件について スコア80-100帯の成約率逆転：モデルキャリブレーションの見直しについて これらの候補について、さらに詳しい情報が必要でしたら、お申し付けください
  - `7e52be1479a28abd` 過去案件数: 2192件 - 審査記録数: 2109件 - モデル精度 (AUC): アンサンブルモデルで 0.82、最新の単一モデルで 0.74 です
- `obsidian_memory_insight_candidates` 2026-08-23 / reflection_update: 10 open
  - `omi_123d3730086e` 明日は、直近3日で一番繰り返している論点を1つ選び、次回の判断基準に昇格する。
  - `omi_d5185d2adeba` 判断資産候補: 直近3日で一番繰り返している論点を1つ選び、次回の判断基準に昇格する。
  - `omi_7b67db142997` 更新する信念: 直近3日で一番繰り返している論点を1つ選び、次回の判断基準に昇格する。
- `obsidian_memory_insight_candidates` 2026-08-24 / noise: 10 open
  - `omi_2b6191b1e825` 今日も一日が始まりましたね。
  - `omi_2a412e6a06e2` 分類カテゴリの定義と構造化。
  - `omi_28115d6613bc` ニュースまとめページできた。
- `obsidian_memory_insight_candidates` 2026-08-24 / judgment_rule: 8 open
  - `omi_91ed06d6164c` この状態は、AIが自動否決に近いと判断した状況です。
  - `omi_733eb2ada5f2` 審査コメントや条件設定の根拠が強化され、説明責任が向上します。
  - `omi_970507014b85` 情報が判断資産として機能しているか、定期的な見直しは大切です。
- `autoresearch_candidates` contract-ownership / confirmation_question: 7 open
  - `4a34956701f5b7c2` 顧客はリース物件の所有権がリース会社にあり、リース期間中および期間満了後の物件の取り扱いを理解しているか。
  - `aaad20717904a882` 顧客の検収体制は適切か、検収書には物件の特定情報（メーカー、型番、シリアルナンバー等）が詳細に記載されているか。
  - `c87bb7bf77de61a2` 顧客の事業内容とリース物件の導入目的は整合しているか、不自然な点はないか。
- `obsidian_memory_insight_candidates` 2026-08-23 / noise: 7 open
  - `omi_c6b1e513af24` しばらくそのままにしておく。
  - `omi_8764ba8985a7` 主要な指標は以下の通りです。
  - `omi_148aadd79fb9` システムに不具合はないかい。
- `autoresearch_candidates` contract-ownership / condition_signal: 6 open
  - `c24b7c5bc1ad29fc` 高額物件、新規顧客、またはサプライヤーの信用度が低いと判断される案件の場合。
  - `96e8303a8a93e13b` 顧客の財務状況や資金使途に不自然な点が見られ、二重譲渡リスクが疑われる場合。
  - `7790e30a74505fd4` 顧客の検収体制が不十分である、または検収書の内容に不備がある場合。
- `autoresearch_candidates` contract-ownership / caution: 6 open
  - `f568693e1e1a89e7` 販売業者の説明のみを鵜呑みにせず、リース会社が直接顧客に契約内容を確認するプロセスを徹底する必要があります。
  - `bf06a3a768e0fe25` 動産譲渡登記はリース物件自体に直接適用されませんが、顧客が保有する他の動産を担保とする場合や、二重譲渡リスク軽減のために確認することは有効です。
  - `c8e05b9e6ef1220e` 事業者間取引には原則クーリングオフ制度は適用されませんが、顧客保護の観点から一定期間の確認期間を設けることは有効です。
- `autoresearch_candidates` contract-ownership / application_rule: 6 open
  - `8db38435df9152d5` 高額物件や新規取引では、リース会社自身または独立した第三者による物件の実在確認を検討します。
  - `4d5e3e65cbcf9dad` 顧客の検収体制と検収書の内容を厳格に確認し、物件の特定情報（シリアルナンバー等）を照合します。
  - `140ebbc3e3b4a5ed` 顧客がリース物件の所有権帰属と、売却・担保供与権限がないことを理解しているかを確認します。

## Microsoft Lens: Utility Density

- Latest accepted preview: 0
- Promoted to active rules: 0
- Promotion rate: None

## Anthropic Lens: Control

- Lifecycle inventory: `{'active': 755, 'candidate_or_review': 529, 'quarantine': 174, 'rejected_or_dismissed': 0, 'maintenance_or_forgetting_review': 8}`
- Utility KPIs: `{'checklist_review_rate': 0.0, 'field_feedback_coverage': 0.0, 'candidate_to_active_pressure': 0.701, 'quarantine_rate_in_experience_flywheel': 0.359}`
- Status counts: `{'active': 745, 'private': 19}`
- Type counts: `{'dialogue_memory': 37, 'factual_memory': 260, 'judgment_memory': 132, 'reflection_memory': 34, 'technical_memory': 284, 'value_memory': 17}`

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
- Current pressure: `{'active_non_value_without_top_usage': 585, 'sleeping_active_rules': 8, 'review_active_rules': 0, 'experience_quarantine': 174}`

### Sleeping Rule Sample

- `cf61a9701fc8cc42` asset_life_and_residual: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- `b259411afb954d6d` business_plan_specificity: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- `a61f3a316a651126` conditional_approval_checks: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- `64e054542be673e4` demo_renewal_asset: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `0d0f11e77fba045d` demo_subsidy_machinery: 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。

## Daily Review Focus

### 候補を少し採否する

- Why: 候補圧を下げ、active memory へ進めるものと捨てるものを分ける。
- `judgment_materials_preview::c903670acff5a2aa` risk_signal: 審査コメントの精度向上: 要約された情報をもとに、審査コメントに具体的な市場背景やリスク要因を盛り込むことができ、説明責任も果たしやすくなります
- `judgment_materials_preview::9a62eec6ed5a9b82` judgment_rule: 条件付き承認にするなら必要な確認 - 物件詳細: リース物件の種類、金額、導入目的、耐用年数、中古市場での換金性を具体的に確認してください
- `judgment_materials_preview::90d13aa3d5c7962e` risk_signal: 金融情報: 金利動向、為替、株式市場、倒産件数、資金調達環境（例: 中小企業向け融資動向、補助金・助成金情報）など、資金繰りや返済能力に直接関わる情報を集約します
- `judgment_materials_preview::8d9226852f582e05` risk_signal: リスクの早期発見: 倒産件数の増加や物価高といった経済ニュースが、リース案件の返済能力や残価リスクにどう影響するか、その本質を抽出しやすくなります
- `judgment_materials_preview::45c5a6a573a78ee3` risk_signal: 特に、それがリース物件の残価リスク、借手の返済能力、または新たなビジネス機会にどう結びつくか、示唆を付加します

### quarantine が多い抽出元を弱める

- Why: 隔離候補は学習材料ではなく、抽出条件のノイズを示す。
- Count: 174
- Sample count: 30
- Sample by source: `{'shion_experience': 30}`
- Review hint: Do not promote these. Use samples to tighten extraction gates or leave as evidence.
- `xfly_82f64ea80da180a0` shion_experience insufficient feedback or context: どう 調子は
- `xfly_c9d7d9143d1f4104` shion_experience insufficient feedback or context: 私の応答の前置きが冗長だった点してあった
- `xfly_3e0006f3c4936173` shion_experience insufficient feedback or context: 修正した
- `xfly_bb027728cd05fc58` shion_experience insufficient feedback or context: 何が気になる点ある？
- `xfly_52add9e0db7d34de` shion_experience insufficient feedback or context: 【審査分析画面からの紫苑レビュー依頼】 この案件を、審査担当者の横にいる紫苑としてレビューしてください。 出力は短く、次の4項目でお願いします。 1. 紫苑の第一印象 2. 数字だけでは見落としそうな違和感 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 前提…

### sleeping active rule を次案件で試すか保留する

- Why: active でも実利用フィードバックがなければ、判断資産として効いているか不明。
- `cf61a9701fc8cc42` asset_life_and_residual: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- `b259411afb954d6d` business_plan_specificity: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- `a61f3a316a651126` conditional_approval_checks: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- `64e054542be673e4` demo_renewal_asset: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `0d0f11e77fba045d` demo_subsidy_machinery: 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。


## Nvidia Lens: Retrieval Pressure

- Notes: 2614 / Edges: 5534
- Estimated raw tokens: 2803428
- Estimated index tokens: 245819
- Estimated token reduction: 0.9123

## Recommended Next Checks

- `memory_entry_metadata_gate` (write_policy): 候補記憶に importance/confidence/trust_level/provenance が不足している。昇格前に共通メタを補う。
- `candidate_budget_review` (write_path): active判断基準1件あたりの候補量が多い。抽出条件・重複統合・候補上限を見直す。
- `human_review_batch` (control): 人間レビュー待ちが溜まっている。自動昇格せず、上位候補だけ短時間で採否する。
- `candidate_to_active_pressure_review` (utility_density): active記憶に対して候補・評価待ちが重い。候補生成より採否・圧縮・却下の運用を優先する。
- `quarantine_sample_review` (forgetting): 経験フライホイールに隔離候補がある。学習材料ではなく、抽出条件のノイズとして扱う。
- `sleeping_active_rule_feedback` (utility_density): active判断資産に実利用フィードバック未記録のものがある。削除前に次案件で効いたか確認する。
