---
agent: scoring-auditor
task: bayesian_engine.py CPT拡張後スコアリング異常検出
timestamp: 2026-03-28 10:45
status: success
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー

`Parent_Guarantor → Financial_Creditworthiness` エッジ追加（a4e4820）により、判定逆転が3シナリオで確認された。うち2件は否決から承認への逆転であり審査判定に直接影響する。設計上の盲点として「High_Network_Risk の抑制効果が Parent_Guarantor の max() 底上げで完全無効化される」構造的問題（深刻度：高）が1件検出され、**修正済み**（nr割引をpg底上げの後に適用する順序に変更）。

---

## 詳細

### 判定逆転案件（3件、修正前）

| シナリオ | pg=0（保証なし） | pg=1（保証あり） | 変化 |
|---------|----------------|----------------|------|
| 子会社＋親保証＋本業必需＋本件限り | 0.1885 | 0.7269 | 否決 → 承認 |
| 高NetworkRisk＋親保証＋債務超過＋本業＋流動性＋期間短縮 | 0.2145 | 0.7456 | 否決 → 承認 |
| 債務超過＋親保証のみ（他全ゼロ） | 0.0946 | 0.6505 | 否決 → 要審議 |

### ASSET_WEIGHT / グレード閾値

- `scoring/` ディレクトリ内に asset_scorer.py, total_scorer.py が存在しないため直接監査不可
- ベイジアンエンジン単体では ASSET_WEIGHT 合計・グレード閾値の問題なし（file-searcher 確認済み）

---

## 課題・リスク

### [高] → 修正済み: High_Network_Risk の抑制効果が Parent_Guarantor の max() で完全無効化

- 該当箇所: `bayesian_engine.py:311-324`（修正後は順序入れ替え済み）
- 修正内容: pg 底上げを先に適用 → nr 割引（×0.88）を後に適用
- 修正後: `nr=1,pg=1,i=1` → 0.55 × 0.88 = **0.484**（修正前: 0.55 固定）

### [中] Parent_Guarantor が FC・HC 両ノードへ二重寄与

- `pg=1` 観測時に FC・HC 両方から承認確率を押し上げ（+0.53 程度）
- 設計意図の明文化、またはドメイン専門家による閾値見直しを推奨

### [低] 債務超過＋親保証のみで要審議昇格（0.6505）

- 他の支持要素がゼロの案件でも親保証単独で 0.0946 → 0.6505
- ビジネスルール上の意図確認を推奨

---

## 後続エージェントへの申し送り

- **rule-validator**: Parent_Guarantor の FC・HC 二重寄与がビジネスルール上意図的かを確認
- **test-runner**: 以下のエッジケーステストを追加推奨
  1. `High_Network_Risk=1, Parent_Guarantor=1, Insolvent=1` の承認確率が `nr=0` ケースより有意に低いこと
  2. `pg` 引数の位置が itertools.product 展開順序（第6要素）と一致していることの回帰テスト
