---
agent: file-searcher
task: 物件資産価値に関連する実装の全調査
timestamp: 2026-03-21 17:00
status: success
reads_from: []
---

## サマリー
「物件資産価値」に関連するファイルを合計 **14 ファイル**（コアファイル 8、関連ファイル 6）特定した。
物件スコアリングは `asset_scorer.py` + `category_config.py` が中核で、カテゴリ別（IT機器/産業機械/車両/医療機器）の加重平均スコアを計算し、`total_scorer.py` で借手スコアと合成して最終スコアを算出する。
別系統として `components/asset_finance.py` は BEP（損益分岐点）・LGD（残価）による物件担保価値評価エンジンを持ち、アセット・ファイナンス型審査に使われる。

---

## コアファイル（直接関連）

### 1. `asset_scorer.py`
物件スコア計算エンジンの本体。
- `calc_asset_score(category, scores, contract)` がメイン関数
- カテゴリ（IT機器/産業機械/車両/医療機器）の各スコアリング項目を加重平均して 0-100 点の物件スコアを返す
- 契約条件（リース期間 vs 技術寿命比率・買取オプション有無・大手メーカー）によって動的にウェイトを調整する `_adjust_weights()` を持つ
- 返却値: `total_score`, `grade`, `item_scores`, `warnings`, `weight_adjusted`
- `get_recommendation(grade)` でグレード別の推奨最長リース年数・残価率を返す

### 2. `category_config.py`
物件カテゴリ別スコアリング設定の定義ファイル（ver 1.1）。

**ASSET_WEIGHT** — カテゴリ別の物件/借手スコア配分比率:
- 車両: 物件 35% / 借手 65%（成熟した中古車市場による換金性の高さ）
- 産業機械: 物件 25% / 借手 75%（汎用機械は担保価値あるが業況依存）
- 医療機器: 物件 20% / 借手 80%（薬機法規制リスクで担保価値が限定）
- IT機器: 物件 10% / 借手 90%（技術陳腐化が速く残存価値はほぼゼロ）

**CATEGORY_SCORE_ITEMS** — カテゴリ別スコアリング項目（weight 合計 100）:
- IT機器: 技術陳腐化リスク低さ(30)、サポート期間(25)、汎用性(20)、市場流動性(15)、リース期間適合性(10)
- 産業機械: 汎用性(30)、メーカーブランド(25)、稼働環境(20)、物理的耐久性(15)、再販市場(10)
- 車両: 汎用性(35)、走行距離リスク低さ(25)、中古市場価格(20)、EV技術変化リスク低さ(10)、改造・カスタム状況(10)
- 医療機器: 規制リスク低さ(30)、技術サイクル安定性(25)、メーカーサポート(25)、移設コストの低さ(15)、施設非依存度(5)

各項目には `tag`（`obsolescence_risk` / `residual_value` / `liquidity_support`）が付いており、動的重み調整の対象タグとして利用される。

**SCORE_GRADES** — グレード閾値: S(90+)、A(80+)、B(65+)、C(50+)、D(0+)

**ASSET_ID_TO_CATEGORY** — `lease_assets.json` の id → スコアリングカテゴリのマッピング:
`vehicle`→車両、`medical`→医療機器、`it_equipment`→IT機器、`manufacturing`→産業機械

### 3. `total_scorer.py`
総合スコア計算モジュール（ver 1.1）。
- `calc_total_score(category, asset_item_scores, obligor_score, contract)` が統合関数
- `asset_score * asset_w + obligor_score * obligor_w` で総合スコアを算出
- カテゴリ未定義の場合はデフォルト配分（物件15%：借手85%）を使用
- 返却値に `asset_score`, `asset_weight`, `obligor_weight`, `recommendation`（推奨リース条件）, `rationale`（設定根拠）を含む

### 4. `components/asset_finance.py`
アセット・ファイナンス型物件審査エンジン（別系統）。
- `AssetFinanceEngine` クラスに物件別の実効減価率（`ASSET_PARAMS`）を定義:
  - 建機: r=0.15、工作機械: r=0.20、PC/IT: r=0.40（最高）、医療機器: r=0.10（最低）、車両: r=0.25
- `calculate_bep()` — 損益分岐点（BEP）算出: V(t)=(1+maint_bonus)×(1-r)^(t/12) が残債 L(t) を超える月を特定
- `calculate_score()` — 財務(最大40点) + 物件LGD/BEP緩和(最大50点) + 逆転定性因子(最大105点)
- 車両は過走行補正（年2万km以上で+10%）・メンテナンスリース補正（-5%）あり

### 5. `scoring_core.py`
クイックスコアリング関数。
- `run_quick_scoring()` 内で `asset_score = _safe_float(inputs.get("asset_score"), default=50.0)` として物件スコアを取得（省略時 50 点デフォルト）
- `final_score = w_borrower * score_borrower + w_asset * asset_score` で最終スコアを計算

### 6. `components/score_calculation.py`
Streamlit 審査フォームの判定ロジック。
- `form_result.get("asset_score", 0)` で物件スコアを受け取る（行 61）
- カテゴリ対応物件がある場合は `ASSET_WEIGHT` 配分を適用（行 695-713）:
  - `final_score = round(asset_score * _aw + contract_prob * _ow, 1)`
- カテゴリ未定義の場合は従来の回帰最適化重みにフォールバック:
  - `final_score = w_borrower * score_percent + w_asset * asset_score`

### 7. `components/form_apply.py`
審査入力フォームのUI。
- `lease_assets.json` を読み込み、物件選択セレクトボックスを表示
- 物件選択で `asset_score = int(_a_item.get("score", 50))` をセット（行 225）
- `selected_asset_id == "vehicle"` の場合は車種タイプ選択が追加表示
- `acquisition_cost`（取得価格・千円）フィールド（行 334）
- `lease_term`（リース期間・月）フィールド（行 327）
- form_result に `asset_score`, `asset_name`, `selected_asset_id`, `lease_term`, `acquisition_cost` を含める

### 8. `slack_screening.py`
Slack 版リース審査ウィザード。
- `ASSET_LIST` に物件マスターを定義（行 83-93）:
  - vehicle(車両・運搬具): score=90、machinery(機械設備): score=80、it_equipment: score=75、medical: score=85、construction: score=70、food: score=65、office: score=70、solar: score=60、other: score=50
- ユーザーが番号で物件を選択すると `d["asset_score"] = value["score"]`（行 819）がセットされ審査に反映

---

## 関連ファイル（間接的）

- `components/analysis_results.py` — 物件スコア配分内訳の表示（「物件スコア × X% ＋ 成約可能性 × Y% ＝ 総合スコア」ブロック、行 326-370）; BNネットワーク表示内で「物件価値【カテゴリ】」metric を表示（行 597-607）
- `components/chat_wizard.py` — チャットウィザードのSTEP: asset で `sel_asset.get("score", 0)` を `asset_score` として設定（行 482-498）
- `components/batch_scoring.py` — 物件スコアの簡易算出ロジック: `(term_ok + cost_ok) / 2.0 * 100`（行 104-111）。asset_scorer.py のカテゴリ別スコアリングとは別系統
- `components/shinsa_gunshi.py` — 物件名キーワード → リセール評価マッピング（行 1385+）; 中古買取業者査定書・耐用年数内リース期間設定で担保評価向上の提案フレーズ
- `components/sidebar.py` — サイドバーで各物件名・スコアを表示（行 406-411）
- `data_cases.py` — `lease_asset_score` キーとしてケースログに保存

---

## 物件資産価値の評価フロー（全体像）

```
[lease_assets.json] ← 物件マスター（id, name, score, category）
        |
[form_apply.py / chat_wizard.py / slack_screening.py]
  物件選択UI → asset_score（0-100）+ asset_category を取得
        |
[category_config.py: CATEGORY_SCORE_ITEMS]
  カテゴリ別スコアリング項目定義（重み合計 100）
        |
[asset_scorer.py: calc_asset_score()]
  各項目スコア × 調整済みウェイトの加重平均
  契約条件による動的ウェイト調整（_adjust_weights）:
    - リース期間 > 技術寿命×80% → obsolescence_risk タグを 1.3倍
    - 買取オプションあり → residual_value タグを 0.7倍
    - 大手メーカー → liquidity_support タグを 1.2倍
  → 物件スコア（0-100）+ グレード（S/A/B/C/D）
        |
[total_scorer.py: calc_total_score()]
  ASSET_WEIGHT を参照
  total = asset_score × asset_w + obligor_score × obligor_w
  → 総合スコア（0-100）+ 推奨リース条件
        |
[score_calculation.py]
  ASSET_WEIGHT 対応物件は上記ルートを使用
  未対応は get_score_weights() の回帰最適化重みにフォールバック
        |
[analysis_results.py]
  「物件スコア × X% ＋ 成約可能性 × Y% ＝ 総合スコア」を表示
```

---

## 使用されているデータ項目

| 項目名 | 型 | 説明 |
|--------|-----|------|
| `asset_score` | int (0-100) | 物件の評価スコア（lease_assets.json から取得） |
| `asset_name` | str | 物件名（UI表示・ログ保存用） |
| `selected_asset_id` | str | 物件ID（vehicle / medical / it_equipment / manufacturing 等） |
| `asset_category` | str | スコアリングカテゴリ（IT機器/産業機械/車両/医療機器） |
| `acquisition_cost` | int (千円) | 取得価格 |
| `lease_term` | int (月) | リース期間 |
| `lease_months` | int (月) | リース期間（asset_scorer.py の contract dict キー） |
| `tech_life_months` | int (月) | 想定技術寿命（asset_scorer.py の contract dict キー） |
| `has_buyout_option` | bool | 買取オプション有無 |
| `is_major_maker` | bool | 大手メーカー品か |
| `lease_asset_score` | float | スコアリング DB ログ用キー（data_cases.py） |

---

## 課題・リスク

- `lease_assets.json` が未配置だとフォームが `asset_score=50`（中立値）にフォールバックし、物件評価が事実上無効化される
- `slack_screening.py: ASSET_LIST` と `category_config.py: ASSET_ID_TO_CATEGORY` のマッピング不整合: Slack側は machinery/construction/food/office/solar を持つが、ASSET_ID_TO_CATEGORY はこれらを定義していない。Slack審査ではカテゴリ別詳細スコアリングが適用されない
- `batch_scoring.py` の物件スコア算出は簡易ロジックで `asset_scorer.py` とは別系統（一貫性なし）
- `coeff_definitions.py` に物件スコア関連の係数定義はない（物件への組み込み係数は `data_cases.py: get_score_weights()` から取得）

---

## 後続エージェントへの申し送り

- **change-impact-analyzer**: `asset_scorer.py` + `category_config.py` + `total_scorer.py` の3ファイルが密結合。変更時の影響範囲分析を推奨
- **code-reviewer**: `slack_screening.py: ASSET_LIST` と `category_config.py: ASSET_ID_TO_CATEGORY` のマッピング不整合を確認推奨
- **security-checker**: `lease_assets.json` の外部読み込みパスのパストラバーサルリスク確認推奨
