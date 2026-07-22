# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 8
- accepted_preview: 6
- candidate: 2

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=6

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life, industry_risk, support_specificity
- Sample claims:
  - 最適な「ルアー」を選ぶ知識: 顧客の状況や物件の特性に合わせて、最適なリース期間、残価設定、金利、契約形態といった「ルアー」を使い分ける知識と経験が求められます
  - 最終的には、単なるスコアだけでなく、残価リスクや補助金のリスク、業種特性といった多角的な視点から、責任ある判断を形成しています
  - 物件の規模や種類によって、最適なリース期間や条件も変わってきますので、具体的なご相談があれば、詳細を伺うことも可能です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-20.md`

### judgment_asset_ops / accepted_preview / evidence=4

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.89
- User evidence: 2
- Axis: asset_life
- Sample claims:
  - この期間、皆様からの貴重なご意見を伺い、リース審査の判断資産としてシステムへ還元できるよう、引き続き丁重に対応させていただきます
  - この期間、皆様の貴重なご意見を伺い、リース審査の判断資産としてシステムへ還元できるよう、丁重に対応させていただきます
  - 【今回使う判断資産】 次の判断資産は、過去の会話・評価・結果から代表ルール化されたものです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: cash_flow, asset_life
- Sample claims:
  - 稟議で残すべき一文 工作機械の保全性とメイン先である点を評価し、受注増加の具体的な根拠と、それによる返済原資の確実性を確認できれば、条件付きで承認可能です
  - 条件付き承認にするなら必要な確認 - 導入目的である「既存受注の増加」について、具体的な顧客名、契約期間、金額などの根拠資料を確認してください
  - 工作機械という物件自体は、一定の市場価値を持つことが多いですが、最終的な判断は借手の財務状況と事業計画に大きく依存します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-20.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`

### business_plan_specificity / accepted_preview / evidence=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.89
- User evidence: 2
- Axis: asset_life, cash_flow
- Sample claims:
  - 数字だけでは見落としそうな違和感 スコアが59.0に留まっているのは、営業メモにある「受注増」の具体的な根拠や、それによる返済原資の確実性がまだ明確でないためかもしれません
  - 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した
  - 受注増に伴う更新投資で、返済原資と物件用途の説明がしやすい
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 1
- Axis: support_specificity, cash_flow
- Sample claims:
  - これらの情報が明確になれば、補助金採択前の段階であっても、より具体的なリスク評価と条件設定が可能になります
  - 審査では、補助金を返済原資として過信せず、採択前・未採択時・入金遅延時でもリース料を払えるかを見る
  - この工作機械案件、補助金採択前だけどどう見る
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

### demo_readiness / candidate / evidence=1

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 1
- Axis: ai_ops
- Sample claims:
  - 明日から誰か来るかもしれない丁重に相手してハッカソンの審査員だ
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-20.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 物件の換金性も考慮されますが、借手の信用力がより重視されます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-21.md`
