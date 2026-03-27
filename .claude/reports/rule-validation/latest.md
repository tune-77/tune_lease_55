---
agent: rule-validator
task: bayesian_engine.py ビジネスルール整合性検証
timestamp: 2026-03-28 11:00
status: success
reads_from: [.claude/reports/file-searcher/latest.md, .claude/reports/code-review/latest.md]
---

## サマリー

構造的バグ・整合性エラーはゼロ。Parent_Guarantor の FC・HC 二重寄与は設計意図として数値的に整合。
nr と pg の相互作用は修正後に正しく動作していることを確認。
低深刻度の文書化不足 2 件、クリップ発動による情報損失 1 件。

---

## 詳細

### Parent_Guarantor FC・HC 二重寄与 — 設計意図として整合

- pg=1 の承認確率押し上げ（全ゼロ基準）: **+0.0803**（非線形爆発なし）
- i=1条件下での pg 追加効果: **+0.5560**（最悲観ベースラインからの上昇）
- pg=1, i=1 の最終値（0.6505）は承認閾値（0.70）未満の「要審議」域に留まる
- FC 経由(+0.0489) + HC 経由(+0.0345) = 線形和 0.0834 ≈ 実測 0.0803（加算的、非線形爆発なし）

### nr と pg の相互作用（修正後確認）

修正後: nr=1,pg=1,i=1 → FC=0.484, approval_prob=0.6167（期待通り）
pg=1 条件下でも nr=1 による FC 低下は -0.1020（pg=0時 -0.0900 より大きい）。

### ASSET_WEIGHT / グレード閾値 / CPT 列数 — 全項目問題なし

### _prob_final_decision クリップ発動

fc=1,hc=1 の最優良ケース（5パターン）で min(0.99,...) クリップが発動。
av/st/ot の個別寄与が消失する（設計通りだが情報損失あり）。

---

## 課題・リスク

| 深刻度 | 内容 |
|--------|------|
| 低 | BN閾値 THRESHOLD_APPROVAL=0.70 vs constants.py APPROVAL_LINE=71 の1点ズレ |
| 低 | fc=1,hc=1最優良ケースで av/st/ot 加算が無効化（クリップ） |

---

## 後続エージェントへの申し送り

- **test-runner**: _prob_final_decision(1,1,1,1,1) クリップパターンのテスト追加推奨
- **test-runner**: _prob_financial_creditworthiness(1,0,0,0,1,1)=0.484 の回帰テスト追加推奨
