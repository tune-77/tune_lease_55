---
agent: code-reviewer
task: bayesian_engine.py CPT拡張（Parent_Guarantor→Financial_Creditworthinessエッジ追加）のレビュー
timestamp: 2026-03-28 10:30
status: success
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー

`bayesian_engine.py` の CPT 拡張実装（`Parent_Guarantor` エッジ追加）は、関数引数の順序・`itertools.product` の展開順序・`evidence` リストの順序がすべて一致しており、構造的なバグは検出されなかった。ただし、`High_Network_Risk` と `Parent_Guarantor` の適用順序がネットワークリスクを完全に無効化するケースがあり、ビジネス意図の要確認事項として記録する。`bayesian_engine.py` 専用のテストが存在しない点が継続的なリスクである。

---

## 詳細

### [bayesian_engine.py:428-436] Financial_Creditworthiness CPT — 引数・evidence順序の整合性

`parents_fc` リストと `_prob_financial_creditworthiness` の引数順序を突き合わせた結果、完全に一致している。

| 位置 | parents_fc リスト | 関数引数 |
|------|-------------------|--------|
| 0 | Insolvent_Status | i |
| 1 | Main_Bank_Support | m |
| 2 | Related_Bank_Status | r |
| 3 | Related_Assets | ra |
| 4 | High_Network_Risk | nr |
| 5 | Parent_Guarantor | pg |

`itertools.product(*[range(2)] * 6)` の展開順序は pgmpy `TabularCPD` の列順序仕様（最後の evidence が最速で変化）と一致する。コミット差分でも `* 5` → `* 6`、`evidence_card=[2] * 5` → `[2] * 6` の変更が正確に行われている。

### [bayesian_engine.py:597] フォールバック推論への pg 引数渡し

`_run_inference_fallback` 内の `p_fc = _prob_financial_creditworthiness(i, m, r, ra, nr, pg)` は今回のコミット（a4e4820）で正しく修正されている。変更前は `pg` が渡されずデフォルト値 `0` のまま動作していた。pgmpy 非使用環境でも `Parent_Guarantor` が正しく反映される。

### [bayesian_engine.py:311-326] nr と pg の適用順序（ビジネスロジック確認事項）

`High_Network_Risk`（nr）の割引（×0.88）を適用した後に `Parent_Guarantor`（pg）の `max` 底上げを適用する実装になっている。これにより以下のケースで nr の効果が完全に無効化される。

- 非債務超過 + ネットワークリスク高 + 親会社保証: base 0.75 → ×0.88 → 0.66 → max(0.66, 0.85) = 0.85
- ネットワークリスクなし + 親会社保証: base 0.75 → max(0.75, 0.85) = 0.85
- 両者で同一の 0.85 に収束し、nr=1 の情報が pg=1 の状況下で失われる

バグではなく「親会社保証は産業ネットワークリスクを上書きする」という設計意図である可能性が高いが、ドキュメントに明記されていないため確認が望ましい。

### [bayesian_engine.py:288-326] 債務超過+保証のフロア値

`pg=1 かつ i=1（債務超過）` のフロアは 0.55 に設定されている。これはドキュメントコメントと一致しており、case_005（運送業・子会社）の実務事例とも整合している。非債務超過（i=0）のフロアは 0.85 で、こちらも妥当な設定。

### [components/form_apply.py] use_container_width=False → width='content' 変換

コミットメッセージ（3cc1bb5）に「use_container_width=False → width='content'」と明示されており、意図的な変換である。他ファイルはすべて `width='stretch'` のため見かけ上の差異があるが、「新しく入力する」ボタンがコンテンツ幅で表示されるという UI 設計の差であり問題なし。

### bayesian_engine.py 専用テストの不在

`tests/` 配下に `bayesian_engine.py` を対象としたテストファイルが存在しない。今回の CPT 拡張（64列への変更）は自動テストなしに動作確認されている状態であり、今後の回帰リスクが残る。

---

## 課題・リスク

- **[中] High_Network_Risk が Parent_Guarantor によって完全に無効化される** — `/bayesian_engine.py:311-326`。nr=1, pg=1, i=0 のとき、ネットワークリスクの割引が pg の max でキャンセルされる。設計意図が正しければコメントへの明記を推奨。意図しない場合はビジネスロジックの修正が必要。

- **[中] bayesian_engine.py の専用テストが存在しない** — `_prob_financial_creditworthiness`、`build_bn_model`、`_run_inference_fallback` のいずれもテストなし。CPT の組み合わせ数が 64 に増加した今回の変更で、境界値テストが特に重要になっている。

- **[低] pg=1 の状況下で nr の識別力が消失する** — 審査精度上の懸念として記録。m=0, r=0, ra=0, i=0 の条件下で nr=0/1 にかかわらず pg=1 であれば 0.85 に収束する。

---

## 後続エージェントへの申し送り

- **security-checker**: `bayesian_engine.py` に機密情報の露出・インジェクション経路はない。DBアクセスは `estimate_empirical_priors` のみで例外処理済み。確認の優先度は低。
- **test-runner**: 以下の関数に対するテスト追加が推奨される
  - `_prob_financial_creditworthiness(i, m, r, ra, nr, pg)` の境界値テスト（特に pg=1 との組み合わせ）
  - `build_bn_model()` の `model.check_model()` 通過確認テスト
  - `_run_inference_fallback` で `Parent_Guarantor=True` が `approval_prob` に反映されることの確認テスト
  - `run_inference` で pgmpy あり/なしの両経路で結果が近似することの確認テスト
- **rule-validator**: `High_Network_Risk` と `Parent_Guarantor` の相互作用（nr割引を pg max で上書き）がビジネスルールとして正しいかを `coeff_definitions.py` / `category_config.py` と照合すること
