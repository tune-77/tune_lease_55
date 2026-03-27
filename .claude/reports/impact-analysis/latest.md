---
agent: change-impact-analyzer
task: 変更影響分析（bayesian_engine.py Parent_Guarantor エッジ追加 + Streamlit API 一括置換）
timestamp: 2026-03-28 10:30
status: success
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー

影響の大きさ: **高**

`bayesian_engine.py` へ `Parent_Guarantor → Financial_Creditworthiness` エッジを追加したことで、審査スコアリングの中核ロジックが変化した。`Parent_Guarantor` は既存の `Hedge_Condition` ノードにも親ノードとして存在するため、同一フラグが FC・HC の二重経路で最終判断に寄与し、承認確率を約 +0.53 押し上げる。`High_Network_Risk` と `Parent_Guarantor` の相互作用については修正済み（pg底上げ後にnr割引を適用する順序に変更）。UI 修正（18 ファイルの Streamlit API 置換）はロジック変更を含まない。

---

## 影響を受ける業務ドメイン

- [x] 審査スコアリング
- [x] Slack ボット・審査フロー（フォールバック推論パスも修正済み、スコア値は変化する）
- [x] ダッシュボード・分析（スコア履歴 DB への書き込み値が変わる可能性あり）

---

## ユーザー影響

### 審査担当者
- `Parent_Guarantor=True` の案件で承認確率が上昇する
- `High_Network_Risk=True` が同時に立っている案件では nr 割引が維持される（修正済み）

### DB・スコア履歴
- 新規案件の審査スコアが変わる。過去案件との比較で乖離が生じる
- 既存案件の再計算を実行すると、pg=1 案件のスコアが過去記録と一致しなくなる

---

## 課題・リスク

### [高] Parent_Guarantor の FC・HC 二重寄与
pg=1 フラグが FC と HC の両方の親ノードになっており、単体で承認確率を約 +0.53 押し上げる。設計意図の明文化が必要。

### [修正済み] High_Network_Risk 抑制効果の実質無効化
`bayesian_engine.py:311-324` の適用順序を修正済み（pg底上げ→nr割引の順）。

### [中] Streamlit バージョン互換
`width='stretch'` / `width='content'` が動作する最低バージョンが requirements.txt で固定されているか未確認。

---

## 後続エージェントへの申し送り

- **security-checker**: Slack 経由での pg フラグ不正セットによる審査結果操作リスクを確認
- **rule-validator**: Parent_Guarantor の FC・HC 二重寄与がビジネスルール上意図的かを確認
