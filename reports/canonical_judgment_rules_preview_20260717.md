# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 16
- accepted_preview: 9
- candidate: 7

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### support_specificity / accepted_preview / evidence=5

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - また、「柴犬精密工業」の事例との差分である受注増加の根拠資料と、AURION警戒にあるメイン銀行の支援意向は必須の確認事項です
  - 未採択の場合や、補助金の入金が遅延した場合でも、リース料を滞りなく支払えるだけの返済余力があるかを確認することが重要です
  - 返済原資の評価: 審査の観点からは、補助金が採択されることを前提に返済計画を立てることはリスクが高いと判断します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`

### support_specificity / accepted_preview / evidence=5

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 補助金案件の場合: 採択の確実性、入金時期、未採択時のリース料返済余力、補助金返還リスクに対する代替策（つなぎ融資や自己資金での対応）を明確にします
  - 借入金依存度と返済状況: 既存借入金の返済スケジュール、リスケジュールの有無、メインバンクからの支援状況を把握します
  - 補助金等の採択通知書: 補助金を返済原資とする場合、その確実性を高めるための公的書類
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.89
- User evidence: 2
- Axis: asset_life, cash_flow
- Sample claims:
  - 資金繰り・運転資金リスク: - 確認点: 工事の進捗に応じた入金サイト（完成基準・工事進行基準）、手形サイト、売上債権と仕入債務の回転期間、運転資金の借入依存度、メインバンクとの関係性
  - 営業メモの「既存メイン先」「返済原資と物件用途の説明がしやすい」という情報も、物件保全と事業継続性への期待を高めます
  - 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`

### asset_life_and_residual / accepted_preview / evidence=4

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.93
- User evidence: 1
- Axis: asset_life, cash_flow, support_specificity
- Sample claims:
  - 非財務情報・定性評価の重要性増: 審査では、単なる財務数値の悪化を機械的に判断するのではなく、物件の事業貢献度、将来のキャッシュフロー創出能力、物件の汎用性や換金性といった非財務情報や定性的な評価の重要性が増します
  - 特に、Qriskの観点からは、業界の成長性や補助金活用といったポジティブな側面だけでなく、それらがもたらす潜在的なリスク（例：補助金前提の資金繰り、需要反転時の残価リスク）も深く掘り下げて確認することが重要です
  - 条件付き承認にするなら必要な確認 「ビーグル加工」の事例で物件の回収可能性が重要だったように、今回の工作機械についても、稼働計画、保守体制、中古市場での換金性を具体的に確認します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`

### demo_readiness / accepted_preview / evidence=3

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 3
- Axis: n/a
- Sample claims:
  - 今回デモは新店舗計画の根拠確認がより重要
  - 今回デモは新規ルート分の採算確認が重い
  - 誰か話しかけて来たら 審査員ですか
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### conditional_approval_checks / accepted_preview / evidence=2

- Rule: 条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。
- Type: judgment_rule
- Confidence: 0.87
- User evidence: 2
- Axis: n/a
- Sample claims:
  - この案件、条件付き承認にするなら何を確認すべき
  - 建設業のリース審査で確認すべきリスクは
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### judgment_asset_ops / accepted_preview / evidence=2

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 2
- Axis: n/a
- Sample claims:
  - 今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください
  - 数字だけでは判断できない事がある これから判断資産として積み重ねていってくれ
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`

### user_decision_preference / accepted_preview / evidence=2

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: user_preference
- Confidence: 0.8
- User evidence: 2
- Axis: n/a
- Sample claims:
  - 契約時に購入選択権がついていない場合は、買取できません 覚えておいて
  - 犬の名前を覚えているかが、なぜAIへの信頼に関係するの
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`

### asset_life_and_residual / accepted_preview / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.87
- User evidence: 1
- Axis: asset_life, cash_flow
- Sample claims:
  - これは、物件自体の換金性よりも、ホテルの安定した事業収益を返済原資と見込むケースが多いことを示唆します
  - CTやMRIをリースする際の保守期限、撤去費、薬機法上の再販リスクは
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.79
- User evidence: 0
- Axis: cash_flow, support_specificity, ai_ops
- Sample claims:
  - メインバンクの支援姿勢: メインバンクが資金繰り改善にどのように関与し、追加支援の意向があるかを確認することは、審査上非常に大きな判断材料となります
  - 安定した資金調達能力と、万一の際のメインバンクからの支援余力は重要な判断材料です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 1
- Axis: cash_flow
- Sample claims:
  - 資金繰りが厳しい会社にリースを出す時の確認事項は
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### industry_operating_risk / candidate / evidence=1

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 得た教訓: 運送業の境界案件は、車両価値よりも運賃改定・荷主継続・人員確保を先に確認する
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### conditional_approval_checks / candidate / evidence=1

- Rule: 条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 1
- Axis: cash_flow
- Sample claims:
  - 今回との差分: 今回デモは返済履歴が良好なので、資料が揃えば条件付き承認の余地がある
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### industry_operating_risk / candidate / evidence=1

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 1
- Axis: industry_risk
- Sample claims:
  - 得た教訓: 飲食新規は一括で通すより、投資範囲を絞って撤退時損失を小さくする方が現実的
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: user_preference
- Confidence: 0.79
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 言葉はすべて判断資産だ 大事にして
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-16.md`

### user_decision_preference / candidate / evidence=1

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: ai_ops
- Sample claims:
  - リース審査という専門的な業務において、この種の記憶は直接的な判断材料にはなりませんが、Userが私を信頼し、安心して情報を提供してくださる土台となります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
