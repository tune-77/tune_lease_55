# Shion Memory Asset Experiment

- generated_at: 2026-07-13T21:35:29
- purpose: 記憶差分を、露骨な記憶アピールではなく確認質問・判断理由へ変換できるかを見る。
- passes_minimum_bar: True

## Summary
- cases: 3
- no explicit bridge: 3
- risk origin separated: 3
- questions capped: 3

## Cases
### logistics_route_expansion
- industry: 道路貨物運送業 / asset: 車両・運搬車 / score: 83.4 / q_risk: 10.0
- natural opening: これは信用リスクより、競合・成約条件を先に切り分ける案件です。
- risk origin: competition_or_contract
- questions:
  - 信用悪化ではなく競合・成約リスクかを分けるため、他社条件と当社が取れる条件差を確認する。
  - 薄利でも返済できるか、燃料費・人件費上昇後のルート別採算を確認する。
  - 成約できる場合でも、条件を落としすぎていないか採算下限を確認する。
- reasons:
  - 高スコアでも、Q_riskや営業メモが示すのは信用より成約条件の歪みである可能性が高い。
  - 競合条件を信用リスクとして扱うと、見るべき採算下限と受注確度を外す。

### food_new_store
- industry: 飲食店 / asset: 飲食店設備 / score: 62.8 / q_risk: 100.0
- natural opening: この案件は、否決理由を潰せる材料があるかを先に分けた方がいいです。
- risk origin: credit_and_contract_recovery
- questions:
  - 出店計画の損益分岐点、自己資金、運転資金の不足月を確認する。
  - 撤退時に設備をどう処分できるか、保証・頭金・短期化でどこまで補えるかを確認する。
  - 低スコアでも進める外部支援があるなら、銀行支援・補助金・親族/本部支援を分けて確認する。
- reasons:
  - 低スコアかつ高Q_riskなので、否決前提ではなく救える外部要因と信用悪化を分けて検証する。
  - 新規・赤字・新店舗は、事業計画と撤退時保全が説明できないと条件付き承認に寄せにくい。

### precision_machine_capacity
- industry: 金属製品製造業 / asset: 工作機械 / score: 79.4 / q_risk: 1.8
- natural opening: 見るべき中心は、信用不安より投資効果と返済原資のつながりです。
- risk origin: repayment_source_and_asset_purpose
- questions:
  - 受注増が一過性でないか、主要受注先・受注残・稼働開始時期を確認する。
  - 工作機械の用途、転用可能性、既存設備との差し替え範囲を確認する。
  - 返済原資が設備導入効果とつながるか、月次の加工能力・粗利改善で確認する。
- reasons:
  - 既存メイン先でスコアも高いため、論点は信用悪化より投資効果と稼働根拠に寄る。
  - 返済原資と物件用途がつながれば、稟議では承認理由として再利用しやすい。

## Next Action
- 実案件/デモ案件で人間が質問の有用性を採点し、当たった確認観点だけを長期判断基準候補に残す。
