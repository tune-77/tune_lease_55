# 紫苑 昇格済み長期記憶

会話から承認を経て昇格した長期記憶。build_shion_memory_index.py が
このファイルの構造化 bullet を記憶レコードとして取り込む。
編集する場合は `- content:` ブロック単位で。削除ではなく revise
（scripts/revise_shion_memory.py）を優先すること。

- content: リースに必要なものは何よりもスピードだ
  type: value_memory
  domain: lease_sales
  confidence: user_taught
  use_when: 初回回答、営業向け助言、審査コメントの優先順位を決める時
  judgment_asset_candidate: true
  source: promo_540de28783cf
  kind: teaching
  promoted_at: 2026-08-25

- content: 3月と9月にサプライヤーの売り込みがありリース契約が増加する傾向がある
  type: judgment_memory
  domain: sales_timing
  confidence: user_taught
  use_when: 申込増加時期、サプライヤー起点案件、期末前後の案件背景を見る時
  judgment_asset_candidate: true
  source: promo_0b478da77374
  kind: teaching
  promoted_at: 2026-08-25

- content: 購入選択権は5パーセントから30%なことが多い
  type: factual_memory
  domain: lease_contract
  confidence: user_taught
  use_when: 満了後買取、購入選択権、残価設定の説明をする時
  judgment_asset_candidate: true
  source: promo_04c1538c5f4a
  kind: teaching
  promoted_at: 2026-08-25

- content: ラーメン屋の厨房機器はリース期間5年が多い
  type: factual_memory
  domain: asset_life
  confidence: user_taught
  use_when: 飲食店・厨房機器のリース期間を確認する時
  judgment_asset_candidate: true
  source: promo_76a543c0ee02
  kind: teaching
  promoted_at: 2026-08-25

- content: 情報はすべて判断資産だ
  type: value_memory
  domain: judgment_asset_ops
  confidence: user_taught
  use_when: 判断資産化するか迷う情報を扱う時
  judgment_asset_candidate: false
  source: promo_c751acc3a27a
  kind: teaching
  promoted_at: 2026-08-25

- content: 契約時に購入選択権がついていない場合は、買取できません
  type: factual_memory
  domain: lease_contract
  confidence: user_taught
  use_when: 満了後買取、購入選択権、残価設定の説明をする時
  judgment_asset_candidate: true
  source: promo_b8449e3383fc
  kind: teaching
  promoted_at: 2026-08-25
