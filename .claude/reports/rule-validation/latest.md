---
agent: rule-validator
task: category_config.py / coeff_definitions.py / constants.py / batch_scoring.py / asset_scorer.py のビジネスルール整合性検証
timestamp: 2026-03-27 14:30
status: success
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー

ルール総数: スコアリング項目 20、グレード閾値 5段階、ASSET_WEIGHT 4カテゴリ、係数モデル 11セット。
検出された問題: **6件**（致命的1件、高リスク2件、中リスク2件、低リスク1件）。
最大の問題は「グレードB(条件付き承認)とAPPROVAL_LINE(71)の矛盾」と「batch_scoring.py が COEFFS に存在しないキーを参照することによる意図しないウェイト取得」の2点。

---

## 1. ウェイト合計一覧

### ASSET_WEIGHT（物件/借手配分）

| カテゴリ | asset_w | obligor_w | 合計 | 問題 |
|---------|---------|-----------|------|------|
| 車両 | 0.35 | 0.65 | 1.00 | なし |
| 産業機械 | 0.25 | 0.75 | 1.00 | なし |
| 医療機器 | 0.20 | 0.80 | 1.00 | なし |
| IT機器 | 0.10 | 0.90 | 1.00 | なし |

全カテゴリで asset_w + obligor_w = 1.00 が成立している。

### CATEGORY_SCORE_ITEMS（スコアリング項目ウェイト合計）

| カテゴリ | 内訳 | 合計 | 問題 |
|---------|------|------|------|
| IT機器 | 30+25+20+15+10 | 100 | なし |
| 産業機械 | 25+20+20+15+15+5 | 100 | なし |
| 車両 | 35+25+20+10+10 | 100 | なし |
| 医療機器 | 30+25+25+15+5 | 100 | なし |

すべてのカテゴリでウェイト合計 100 が成立している。

### data_cases.py デフォルト重み

| 組み合わせ | 合計 | 問題 |
|-----------|------|------|
| DEFAULT_WEIGHT_BORROWER(0.85) + DEFAULT_WEIGHT_ASSET(0.15) | 1.00 | なし |
| DEFAULT_WEIGHT_QUANT(0.6) + DEFAULT_WEIGHT_QUAL(0.4) | 1.00 | なし |

### batch_scoring.py 借手スコア計算ウェイト（問題あり）

| キー | デフォルト値 | 実際の取得値(既存先) | 実際の取得値(新規先) |
|------|------------|---------------------|---------------------|
| w_grd (grade) | 0.30 | 0.30 (COEFFS不在) | 0.30 (COEFFS不在) |
| w_rr (rieki_rate) | 0.20 | 0.20 (COEFFS不在) | 0.20 (COEFFS不在) |
| w_eq (equity_ratio) | 0.25 | 0.25 (COEFFS不在) | 0.25 (COEFFS不在) |
| w_roe (roe) | 0.10 | 0.10 (COEFFS不在) | 0.10 (COEFFS不在) |
| w_cnt (contracts) | 0.05 | **0.28131** (COEFFS上書き) | **0.00** (COEFFS上書き) |
| weight_sum | 1.00 | **1.13131** | **0.85000** |

---

## 2. グレード閾値の矛盾リスト

### [CRITICAL] グレードBとAPPROVAL_LINEの意味論的矛盾
**category_config.py:220 と constants.py:245**

```
グレード S: 90-100  → 全件 APPROVAL_LINE(71) 超え → 承認圏内
グレード A: 80-89   → 全件 APPROVAL_LINE(71) 超え → 承認圏内
グレード B: 65-79   → 71以上は承認圏内、65-70は「要審議」（矛盾ゾーン: 7点幅）
グレード C: 50-64   → 全件 APPROVAL_LINE(71) 未満 → 要審議
グレード D:  0-49   → 全件 APPROVAL_LINE(71) 未満 → 否決圏
```

グレードBのテキスト「条件付き承認」と判定「要審議」が同一スコアレンジで同時に表示されると
審査担当者が誤解してそのまま稟議書を提出するリスクがある。
asset_scorer.get_recommendation() のグレードB向けメッセージ「条件付き承認として稟議書に根拠を記載してください」も承認ライン未達案件に表示される。

### [MED] ALERT_BORDERLINE_MIN(68)とグレードBの部分重複
**constants.py:248**

グレードB（65-70点）のうち 65-67点は要確認ゾーン外だが 68-70点は要確認ゾーン内となり、
同じグレードBの物件が3点差でアラート有無が変わる。

### [LOW] REVIEW_LINE(40)がグレード定義と独立
**constants.py:246**

REVIEW_LINE=40 は D帯(0-49) 内にあり、グレードDを暗黙に「D1:40-49」と「D2:0-39」に2分割している。
batch_scoring.py の判定では 40-70点を「ボーダー」として SCORE_GRADES の C/D と乖離している。

---

## 3. 係数定義の矛盾

### [HIGH] batch_scoring.py が COEFFS の contracts キーを誤参照
**components/batch_scoring.py:87 と coeff_definitions.py**

batch_scoring.py の `w_cnt = coeffs.get("contracts", 0.05)` は借手スコア計算のウェイト（0〜1）を
取得しようとしているが、COEFFS["全体_既存先"]["contracts"] = 0.28131（ロジスティック回帰係数）が
存在するため、デフォルト値(0.05)が使われず 0.28131 が無言で適用される。

COEFFS["全体_新規先"]["contracts"] = 0 のため新規先は w_cnt=0 となり、
既存先の weight_sum=1.13131 と新規先の weight_sum=0.85000 で正規化基準が異なる。
正規化処理により最終値は0-100に収まるが、contracts の寄与比率が設計意図と全く異なっている。

### [MED] APPROVAL_LINE の二重定義
**constants.py:245 と scoring_core.py:24**

同名変数 APPROVAL_LINE = 71 が2箇所で独立定義されている。
batch_scoring.py は constants から import するが、scoring_core.py はローカル定義で使用。
現在は値が同一だが将来の変更時に同期漏れが発生するリスクがある。

---

## 4. D帯分割（D1:35-50 / D2:0-34）の実装推奨案

現状の問題点:
- D帯（0-49点）は50点幅で他グレード（各10-15点幅）の3倍以上の幅がある
- REVIEW_LINE(40)で暗黙に2分割されているがSCORE_GRADESに反映されていない

推奨実装案（category_config.py への変更）:

```python
SCORE_GRADES = [
    {"min": 90, "label": "S",  "text": "積極承認",    "color": "#22c55e"},
    {"min": 80, "label": "A",  "text": "通常承認",    "color": "#3b82f6"},
    {"min": 71, "label": "B+", "text": "条件付き承認", "color": "#f59e0b"},  # APPROVAL_LINEと一致
    {"min": 65, "label": "B-", "text": "要稟議検討",  "color": "#fb923c"},  # 旧B帯65-70を独立
    {"min": 50, "label": "C",  "text": "要慎重検討",  "color": "#f97316"},
    {"min": 35, "label": "D1", "text": "要否決検討",  "color": "#dc2626"},
    {"min":  0, "label": "D2", "text": "原則否決",    "color": "#991b1b"},
]
```

最小変更での対応案（変更影響を最小化する場合）:
- グレードB の text を「条件付き承認」から「承認検討（APPROVAL_LINE確認要）」に変更し意図を明示
- constants.py に `GRADE_B_MIN = 65` と `APPROVAL_LINE = 71` を並記して関係をコメントで明示

---

## 課題・リスク

1. グレードB(条件付き承認)と APPROVAL_LINE(71)の矛盾により、担当者がグレード表示を見て承認できると誤解するリスクがある（稟議ミスリード）。
2. batch_scoring.py の contracts ウェイト問題: 既存先でウェイト合計が 1.13（設計値 1.00）になり、新規先は 0.85 となる。同一財務内容でも取引区分でスコアが変わる。
3. APPROVAL_LINE の二重定義: constants.py と scoring_core.py に同名定数が存在。将来変更時の同期漏れリスク。

---

## 後続エージェントへの申し送り

- **code-reviewer**: 修正が必要な箇所
  - components/batch_scoring.py:87 — w_cnt の取得先を COEFFS から分離する（専用の BATCH_WEIGHTS 定数を設けるか、ハードコードで分離する）
  - scoring_core.py:24 — APPROVAL_LINE のローカル定義を削除して from constants import APPROVAL_LINE に統一する
  - category_config.py:220 — グレードB の min を 71 に変更するか、text を修正して APPROVAL_LINE との関係をコメントで明示する

- **test-runner**: 境界値テストを追加すべき関数
  - asset_scorer._get_grade(): score=65, 70, 71, 80, 90 の境界値テスト
  - batch_scoring._score_one(): 取引区分「既存先」と「新規先」で同一財務データの借手スコア差異を確認するテスト（contracts ウェイト問題の検出）
  - scoring_core.run_quick_scoring(): asset_score 省略時と入力時のスコア差異テスト（デフォルト50の影響確認）

- **scoring-auditor**: batch_scoring.py の簡易スコアルート（物件IDなし）と asset_scorer.calc_asset_score() ルート（物件IDあり）の同一入力に対するスコア差異の定量計測を推奨
