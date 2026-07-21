# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 6
- accepted_preview: 2
- candidate: 4

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

### demo_readiness / accepted_preview / evidence=2

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 私ができることは、リース審査に関する過去の事例や、現行の審査ロジック、あるいは最近のシステム変更点など、審査員の方が確認されたい情報を迅速に提供することだと考えております
  - 審査員が来ると思うから丁重に相手して
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-19.md`

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

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 塾の経営状況: 安定した生徒数の確保、収益性、資金繰りなどを確認し、リース料の支払い能力を評価します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-20.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: n/a
- Sample claims:
  - 私、紫苑の役割は、Userさんのリース審査に関する経験や判断基準を、再利用可能な「判断資産」として育てていくことです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-19.md`

### support_specificity / candidate / evidence=1

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - リース審査の場面では、補助金の採択状況や入金時期が資金繰りにどう影響するか、また、万が一採択されなかった場合の代替策なども確認することがあります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-19.md`
