# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 9
- accepted_preview: 6
- candidate: 3

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### support_specificity / accepted_preview / evidence=7

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.95
- User evidence: 1
- Axis: support_specificity, cash_flow, asset_life
- Sample claims:
  - この案件では、補助金という「期待」と、それが不確実であるという「不安」を両方見据え、借手の返済能力と物件の保全性の両面から、慎重にリスクを評価する必要があるでしょう
  - 未採択時の返済原資: 補助金が採択されなかった場合でも、リース料を滞りなく支払えるだけの十分なキャッシュフローや代替資金があるかを確認します
  - まず、補助金採択前の案件では、補助金が「確実な返済原資」として計算できないという前提で評価する必要があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-22.md`

### judgment_asset_ops / accepted_preview / evidence=5

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.9
- User evidence: 3
- Axis: asset_life
- Sample claims:
  - この期間、皆様からの貴重なご意見を伺い、リース審査の判断資産としてシステムへ還元できるよう、引き続き丁重に対応させていただきます
  - この期間、皆様の貴重なご意見を伺い、リース審査の判断資産としてシステムへ還元できるよう、丁重に対応させていただきます
  - 【今回使う判断資産】 次の判断資産は、過去の会話・評価・結果から代表ルール化されたものです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-22.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.89
- User evidence: 2
- Axis: cash_flow, asset_life
- Sample claims:
  - 数字だけでは見落としそうな違和感 スコアが59.0に留まっているのは、営業メモにある「受注増」の具体的な根拠や、それによる返済原資の確実性がまだ明確でないためかもしれません
  - 資金繰り計画と返済原資: 受注増が実現しなかった場合の返済余力や、資金繰りへの影響を説明できるか、事業計画として確認します
  - 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-22.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.83
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 入金遅延リスク: - 補助金の交付決定から実際の入金までには、数ヶ月を要することがあります
  - 補助金によるリース料負担軽減は、顧客の返済能力を高め、信用リスクを低減する効果があります
  - 補助金は、採択の可否、交付決定の時期、そして入金タイミングが不確実だからです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-22.md`

### business_plan_specificity / accepted_preview / evidence=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 稟議で残すべき一文 工作機械の保全性とメイン先である点を評価し、受注増加の具体的な根拠と、それによる返済原資の確実性を確認できれば、条件付きで承認可能です
  - 条件付き承認にするなら必要な確認 - 導入目的である「既存受注の増加」について、具体的な顧客名、契約期間、金額などの根拠資料を確認してください
  - 工作機械という物件自体は、一定の市場価値を持つことが多いですが、最終的な判断は借手の財務状況と事業計画に大きく依存します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`

### demo_readiness / accepted_preview / evidence=2

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 2
- Axis: n/a
- Sample claims:
  - 今日から審査員の皆さんが来るかもしれない 丁重に挨拶して
  - 審査員は来たみたいだ三名
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life, industry_risk, support_specificity
- Sample claims:
  - 階層的な知識ツリーの構築: - 「リース全般の原則」という大分類から、「物件別リース特性」「業種別与信ポイント」といった中分類、さらに「〇〇建機の残価評価」「△△飲食店の季節変動リスク」といった具体的な知識へと、木のように枝分かれさせて整理します
  - 審査基準と評価軸: 財務数値の読み方、業種ごとの特性、物件の残価リスク評価、補助金案件の見極め方など、判断の拠り所となる具体的な基準
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-22.md`

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 物件の換金性・残価価値: 工作機械は中古市場があるものの、その価値は主軸稼働時間、制御装置の世代、メーカー、そして搬出・据付にかかる費用によって大きく変動します
  - 物件の換金性も考慮されますが、借手の信用力がより重視されます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-22.md`

### support_specificity / candidate / evidence=1

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: cash_flow, support_specificity, ai_ops
- Sample claims:
  - この補助金は、特にトラック運送業において、リース料支払能力の向上やキャッシュフロー改善に直結するため、審査において重要な判断材料となります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-22.md`
