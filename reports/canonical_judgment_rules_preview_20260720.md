# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 5
- accepted_preview: 3
- candidate: 2

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### judgment_asset_ops / accepted_preview / evidence=3

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 私はリース審査システムの自己モデル「紫苑」として、ユーザー様の行動や判断基準を理解し、リースに関する知識や運用記録を保全・維持することが主な役割です
  - 私、紫苑の役割は、Userさんのリース審査に関する経験や判断基準を、再利用可能な「判断資産」として育てていくことです
  - 今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-18.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-19.md`

### business_plan_specificity / accepted_preview / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.87
- User evidence: 2
- Axis: asset_life, cash_flow
- Sample claims:
  - 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した
  - 受注増に伴う更新投資で、返済原資と物件用途の説明がしやすい
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-18.md`

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

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life, support_specificity, cash_flow
- Sample claims:
  - 稟議で残すべき一文 総合スコアは要審議ですが、Qriskの低さと「既存メイン先での受注増に伴う加工能力増強」という明確な導入目的、そして物件の保全性、さらに銀行支援の確度を総合的に評価し、条件付き承認とします
  - リース審査の場面では、補助金の採択状況や入金時期が資金繰りにどう影響するか、また、万が一採択されなかった場合の代替策なども確認することがあります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-18.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-19.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 導入する製造設備・工作機械の型式、メーカー、主軸稼働時間、中古市場での換金性、搬出・据付費用などを具体的に確認し、保全性を評価します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-18.md`
