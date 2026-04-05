---
agent: scoring-auditor
task: スコアリングロジック全面監査（DB異常値・ロジック整合性・MLモデル・重複データ）
timestamp: 2026-04-04 11:30
status: success
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー

全160件の審査レコードを調査した結果、**5種類の深刻な問題**が確認された。最も重大なのは「lease_credit_log 係数が異常に大きく（1.14）、リース信用枠の入力だけで財務内容に関わらず承認圏内へ引き上げる」構造的バグと、「全財務値が10百万円という同一のテストデータが51件も蓄積されており、承認圏内（97.3点）で記録されている」点である。scoring_core.py と predict_one.py が別パスで動作し、意味論（承認確率 vs デフォルト確率）が混在している問題も深刻。

---

## 詳細

### 1. 異常値の検出

#### 1-1. スコアが95点以上に張り付く案件群（深刻度: 高）

160件中80件（50%）がスコア95点以上の「承認圏内」。そのうち51件は以下の同一財務値を持つテストデータ由来と推定される。

| 財務項目 | 値 |
|--------|-----|
| revenue_m | 10百万円 |
| op_profit_m | 10百万円 |
| total_assets_m | 10百万円 |
| net_assets_m | 10百万円 |
| lease_amount_m | 10百万円 |
| lease_term | 60ヶ月 |
| score | 97.3点 |
| judgment | 承認圏内 |
| contract_prob | 100.0% |

これらは2026-03-01 から2026-03-27の期間にかけて繰り返し登録されており、開発テスト中のデータが本番DBに混在している状態である。

**なぜ97.3点になるか**: scoring_core.py の lease_credit_log 係数が 1.140132 と異常に大きく、np.log1p(10000) * 1.14 = 10.5 という logit 空間での寄与が生じる。これにより sigmoid 後の score_prob がほぼ 1.0 になり、final_score = 0.85 * 100.0 + 0.15 * 50 = 92.5点 以上が保証される。

#### 1-2. 純資産マイナス案件（深刻度: 中）

7件が net_assets_m < 0（債務超過）であるにも関わらず、全件 judgment = "要審議"（否決なし）で記録されている。

| id | net_assets_m | score | judgment |
|----|-------------|-------|---------|
| 4 | -10百万円 | 47.3 | 要審議 |
| 5 | -20百万円 | 47.3 | 要審議 |
| 6 | -20百万円 | 47.3 | 要審議 |
| 7 | -20百万円 | 42.5 | 要審議 |
| 8 | -20百万円 | 47.3 | 要審議 |
| 9 | -20百万円 | 47.3 | 要審議 |
| 44 | -10百万円 | 46.7 | 要審議 |

scoring_core.py の z 値計算では user_equity_ratio は比較表示用にのみ使われ、スコア計算には直接影響しない構造になっている。つまり**債務超過のペナルティが存在しない**。

#### 1-3. contract_prob と score の意味論的矛盾（深刻度: 高）

10件で contract_prob = 100.0% かつ score < 50点、うち2件で contract_prob >= 85% かつ judgment = '否決' という矛盾が確認された。

代表例:
- id=7: score=42.5, judgment=要審議, contract_prob=100.0%, net_assets_m=-20百万円
- id=159: score=34.2, judgment=否決, contract_prob=85.3%
- id=160: score=34.2, judgment=否決, contract_prob=85.3%

contract_prob は lease_logic_sumaho8.py 内で別ロジックにより計算され、score_calculation.py の final_score とは独立して動く。2つの指標の意味論が統一されていない。

#### 1-4. 重複データ（深刻度: 中）

同一財務データが大量に重複登録されている。上位4パターンの重複数:

| revenue/op_profit/total_assets/net_assets/lease（百万円） | 件数 |
|----------------------------------------------|------|
| 10/10/10/10/10 | 51件 |
| 2180/260/2680/830/0 | 26件 |
| 60/5/650/40/5 | 24件 |
| 260/5/540/20/0 | 6件 |

160件中107件（67%）が4パターンのいずれかに集中しており、**実質的に独立したサンプルは53件程度**。

---

### 2. スコアリングロジックの整合性

#### 2-1. lease_credit_log 係数の異常な大きさ（深刻度: 高）

ファイル: coeff_definitions.py（全体_既存先）, scoring_core.py:122-125

lease_credit_log = 1.140132

np.log1p(lease_credit) の係数として、lease_credit = 10000千円（1千万円）の場合:
- 寄与 = 1.140132 * 9.21 = **10.5 logit ポイント**
- これ単独でほぼ sigmoid(z) → 1.0 に飽和させる

結果として、リース信用枠が入力されているだけで（財務内容に関わらず）**スコアが90点超に張り付く構造的問題**がある。これはモデルが「リース信用枠あり = 既存リース顧客 = 承認実績あり」を過剰に学習した過学習の典型。

#### 2-2. w_borrower + w_asset = 1.0（正常）

data_cases.py: DEFAULT_WEIGHT_BORROWER = 0.85, DEFAULT_WEIGHT_ASSET = 0.15。合計 1.0。問題なし。

#### 2-3. 承認ラインの不統一（深刻度: 中）

以下の箇所で承認ラインが異なる値が使われている:

| ファイル | 行 | 値 |
|--------|-----|---|
| scoring_core.py | 24 | APPROVAL_LINE = 71 |
| constants.py | 245 | APPROVAL_LINE = 71 |
| screening_report.py | 46 | score >= 70 |
| charts.py | 417, 460, 537 | score >= 70 |

screening_report.py と charts.py が 70点を使用しており、公式の APPROVAL_LINE = 71 と1点ズレている。境界線上（70点ちょうど）の案件で表示上の判定が乖離する。

#### 2-4. 否決の生成経路の混乱（深刻度: 中）

scoring_core.py の run_quick_scoring() は "承認圏内" または "要審議" しか返さない。しかし constants.py:246 に REVIEW_LINE = 40（これ未満は即否決圏）が定義されており、score_calculation.py:994-995 の以下のコードで "否決" が設定される:

```python
# components/score_calculation.py:994-995
elif _hantei_score < _eff_review:
    st.session_state['last_result']["hantei"] = "否決"
```

この "否決" が customer_db.py 経由で screening_records に保存される。実際にDBに5件の "否決" が存在し、スコアは 6.9〜35.0点でいずれも REVIEW_LINE(40) 未満。ロジック自体は一貫しているが、scoring_core.py のコメントと実態が乖離しており可読性が低い。

#### 2-5. predict_one.py の decision ロジック（深刻度: 高）

ファイル: scoring/predict_one.py:192-193

```python
hybrid_prob = 0.3 * legacy_prob + 0.7 * ai_prob
decision = "承認" if hybrid_prob < 0.5 else "否決"
```

legacy_prob（industry_hybrid_model.py）は industry_coefficients.pkl の係数で計算され、ROA・equity_ratio に正の係数を持つ。つまり**健全企業ほど hybrid_prob が高く → "否決" になるという意味論が逆転した判定ロジック**になっている。

検証: 建設業、全値健全（ROA=100%, equity_ratio=100%）の場合:
- linear score = -2.8 + 12.0 + 10.0 + 12.0 + ... = 31.2
- legacy_prob = sigmoid(31.2) = 1.000
- hybrid_prob ≈ 1.0 → decision = "否決"（正しくは「承認」のはずが逆になる）

現状は scoring_result = None のデッドコード状態（score_calculation.py:837-841）のため実害なし。ただし将来有効化される際に誤判定を招く可能性がある。

#### 2-6. pd_percent >= 50 のとき -50点ペナルティ（深刻度: 中）

score_calculation.py:858-867: pd_percent >= 50 の場合 final_score -= 50。

これにより「デフォルト確率50%以上」の案件がスコアから50点引かれ、DBに低スコアで保存される。しかし contract_prob は別途加算調整されており、スコアとcontract_probが大きく乖離するケースが生まれる（id=159: score=34.2, contract_prob=85.3%）。

---

### 3. MLモデルの状態

#### 3-1. 学習サンプル数（深刻度: 高）

RobustScaler の n_features_in_ = 29。DB の実質独立サンプル数は160件から重複107件を除くと**実質53件程度**。

unified_ai_model.pkl は lightgbm 形式（環境に lightgbm 未インストールのため詳細確認不可）。34〜53件程度の学習データで n_estimators=100, max_depth=6 の LightGBM を学習している場合、完全な過学習が発生している蓋然性が高い。

#### 3-2. hybrid 重みの問題（深刻度: 高）

predict_one.py:192: hybrid_prob = 0.3 * legacy_prob + 0.7 * ai_prob

過少データで学習した ai_prob に **70% の重みが置かれている**。過学習モデルの ai_prob を多用するのは信頼性の観点から危険。

#### 3-3. 業種マッピングの近似（深刻度: 低）

predict_one.py:74: 運輸業（H）を建設業として近似。

```python
if code == "H":
    return "建設業"  # 運輸は建設業で近似
```

業種特性が大きく異なる業種を近似しており、運輸業案件の scoring が建設業係数で評価される。DBには運輸業7件が存在し、全て建設業係数で評価されている。

---

### 4. top5_reasons の妥当性

screening_records テーブルの memo カラムは全160件 NULL（top5_reasons が永続化されていない）。

scoring_core.py の compute_score_contributions() による SHAP 近似では、lease_credit_log の寄与（logit空間で10.5ポイント）が圧倒的に大きく、lease_credit が入力されている案件では top5 の 1位が常に「リース信用枠」になる計算。これは lease_credit_log 係数の問題と連動している。

---

### 5. DB全体の統計サマリー

| 指標 | 値 |
|-----|-----|
| 総レコード数 | 160件 |
| スコア最小 | 6.9点 |
| スコア最大 | 98.0点 |
| スコア平均 | 76.6点 |
| 95点以上 | 80件（50%） |
| 60点未満 | 40件（25%） |
| 承認圏内 | 93件（58%） |
| 要審議 | 62件（39%） |
| 否決 | 5件（3%） |
| equity_ratio カラム | 全160件 NULL（DBスキーマ INTEGER だが customer_db.py の _round_ratio() が常に None を返している可能性） |
| 同一財務値の重複 | 4パターンで107件（67%） |
| 純資産マイナス案件 | 7件（全て要審議、否決なし） |
| contract_prob=100.0 かつ score<50 | 10件 |
| 建設業（D）偏重 | 119件（74%） |

---

## 課題・リスク

1. **[高] lease_credit_log 係数の飽和問題**: scoring_core.py 内の全体_既存先係数。リース信用枠さえ入力すれば財務内容に関わらず90点超に張り付く。財務悪化企業が正当に否決されない可能性がある。係数の再調整か、上限スケーリングが必要。

2. **[高] テストデータの本番DB混在**: 全値10百万円の51件が蓄積されており、MLモデルの再学習に使われた場合にモデルが歪む。削除またはフラグ管理が必要。

3. **[高] predict_one.py の判定論理逆転**: industry_hybrid_model の legacy_prob は「承認確率」だが hybrid_prob < 0.5 → 承認 と「デフォルト確率」として扱われており逆の判定になる。現在はデッドコードだが将来有効化で重大な誤判定を招く。

4. **[高] 学習データ不足による過学習**: 実質独立サンプル約53件、建設業偏重74%でLightGBM（n_estimators=100）を学習。ai_prob の信頼性は極めて低く、70%の重みで使用するのは危険。

5. **[中] 承認ライン71点 vs 70点の不統一**: screening_report.py と charts.py が70点を使用、境界線案件で表示上の判定が乖離する。

6. **[中] 債務超過のペナルティ欠如**: scoring_core.py の z 値計算に純資産マイナスのハード制約がなく、債務超過でも「要審議」止まり。is_negative_equity フラグは feature_engineering_custom.py に定義されているが scoring_core.py で未使用。

7. **[中] contract_prob とスコアの意味論非統一**: pd_percent >= 50 ペナルティにより score が大幅低下する一方、contract_prob は別ロジックで算出されるため「スコア低いが成約確率85%」という矛盾した情報が審査担当者に提示される。

---

## 後続エージェントへの申し送り

- **code-reviewer**: scoring_core.py の lease_credit_log 係数（1.140132）の再調整が最優先。scoring/models/industry_specific/industry_coefficients.pkl の係数もサービス業の rent_to_revenue = -0.22 など符号・大きさの妥当性を確認すること。scoring/predict_one.py:193 の hybrid_prob < 0.5 → 承認 という判定の意味論が逆転していないか要確認。customer_db.py の equity_ratio カラムが全件 NULL になる原因（_round_ratio() 関数）の調査も必要。

- **test-runner**: 追加すべきエッジケーステスト:
  1. lease_credit = 0 かつ全財務健全 → スコアが正常範囲（50〜80点）に収まるか
  2. 債務超過（net_assets < 0）案件 → 要審議または否決になるか
  3. scoring/predict_one.py で健全財務 → decision が "承認" になるか（現在は "否決" になる可能性あり）
  4. スコア70点ちょうど → screening_report.py と scoring_core.py で判定が一致するか

- **data-quality-checker**: screening_records の全値10百万円テストデータ51件、および同一財務値の重複107件の扱いを確認・クリーニングを推奨。equity_ratio カラムが全件 NULL の原因調査も必要。

- **rule-validator**: constants.py:246 の REVIEW_LINE = 40 と scoring_core.py の APPROVAL_LINE = 71 の間の「40-70点 = 要審議」という3段階判定の設計意図を文書化し、screening_report.py（70点ライン）との整合性を確認すること。
