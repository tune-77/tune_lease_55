---
agent: file-searcher
task: 変更ファイル調査（直近3コミット）
timestamp: 2026-03-28 00:00
status: success
reads_from: []
---

## サマリー

直近3コミット（a4e4820, 3cc1bb5, 4afc5aa）で計20ファイルが変更された。
変更の性質は2種類に明確に分類される。

1. **Streamlit API 廃止警告対応**（3cc1bb5）: `use_container_width=True/False` を新 API `width='stretch'` / `width='content'` に一括置換。18ファイルに渡る純粋なUI修正。
2. **ベイジアンBN モデルの機能追加**（a4e4820）: `Parent_Guarantor → Financial_Creditworthiness` エッジを追加し、親会社連帯保証を信用力評価に反映。
3. **セッションデータ更新**（4afc5aa）: `data/last_case.json`, `data/recent_modes.json` の内容更新（コード変更なし）。

---

## 詳細

### カテゴリA: Streamlit API 置換（use_container_width → width パラメータ）

影響ファイル18件。変更内容はすべて機械的な置換であり、ロジック変更はない。

| ファイル | 役割 | 置換箇所数 |
|--------|------|---------|
| `backup_manager.py` | バックアップUI（サイドバー） | 1箇所 |
| `draft_manager.py` | 下書き保存・復元UI（サイドバー） | 3箇所 |
| `future_simulation.py` | 将来シミュレーションのPlotlyチャート表示 | 2箇所 |
| `lease_logic_sumaho12.py` | エントリポイント・審査ルール設定ページのDataFrame | 1箇所 |
| `components/agent_hub.py` | エージェントHub（異常検知・数学者・ビジュアルレポート・週次バッチ） | 4箇所 |
| `components/agent_team.py` | エージェントチーム会議UI（コード生成・プリセット・議論ボタン） | 6箇所 |
| `components/ai_consultation.py` | AI相談・議論タブのボタン群 | 4箇所 |
| `components/analysis_qual.py` | 定性分析のDataFrame表示 | 2箇所 |
| `components/analysis_quant.py` | 定量分析のDataFrame表示 | 4箇所 |
| `components/batch_scoring.py` | 一括スコアリングのDataFrame表示 | 2箇所 |
| `components/dashboard.py` | ダッシュボードの財務指標DataFrame | 1箇所 |
| `components/form_apply.py` | 審査フォーム（判定開始ボタン・新規入力・再判定） | 4箇所（`use_container_width=False` → `width='content'` も含む） |
| `components/home.py` | ホーム画面のFABボタン・カードボタン | 2箇所 |
| `components/report.py` | 審査レポートのチャート・ダウンロードボタン | 6箇所 |
| `components/settings.py` | 係数分析・係数履歴のDataFrame | 4箇所 |
| `components/shap_explanation.py` | SHAP説明のImage・DataFrame | 4箇所 |
| `components/shinsa_gunshi.py` | 審査軍師の分析・登録ボタン・DataFrameなど | 8箇所 |
| `components/sidebar.py` | サイドバー全体のボタン群 | 6箇所 |
| `components/subsidy_master.py` | 補助金マスタ管理のDataFrame | 1箇所 |

注意点: `form_apply.py` の「新しく入力する」ボタンのみ `use_container_width=False` → `width='content'` と異なるパターンで置換されている。その他はすべて `width='stretch'`。

### カテゴリB: ベイジアンネット（BN）モデルの構造変更

ファイル: `bayesian_engine.py`

- `BN_EDGES` に `("Parent_Guarantor", "Financial_Creditworthiness")` エッジを追加
- `_prob_financial_creditworthiness()` 関数に引数 `pg: int = 0`（親会社保証フラグ）を追加
  - 債務超過＋保証あり: `base = max(base, 0.55)` で信用力を底上げ
  - 非債務超過＋保証あり: `base = max(base, 0.85)` で追加信用補完
- `build_bn_model()` の `Financial_Creditworthiness` CPT を5親ノード→6親ノードに拡張（`combos_fc` が 2^5=32 → 2^6=64 パターン）
- フォールバック推論 `_run_inference_fallback()` も `pg` 引数を渡すよう修正

---

## 課題・リスク

- **BNモデルの CPT 再検証が必要**: 親ノードが1つ増えたことで `TabularCPD` の組み合わせ数が倍増した。`pt_fc` の計算が `_prob_financial_creditworthiness(*c)` のアンパック展開に依存しており、`pg` 引数の順序が `evidence` リストの順序と一致しているか要確認。
- **Streamlit バージョン依存**: `width='stretch'` / `width='content'` は Streamlit の新 API。旧バージョンを使う環境ではエラーになる可能性がある。requirements.txt / pyproject.toml のバージョン固定状況を確認すること。
- **`form_apply.py` の `width='content'`**: 他ファイルは `'stretch'` に統一されているが、このボタンのみ `'content'` に変換されており、UIの意図的な差異である可能性がある。意図通りかレビューで確認が望ましい。

---

## 後続エージェントへの申し送り

- **scoring-auditor**: `bayesian_engine.py` の CPT 変更（`Parent_Guarantor` エッジ追加）が審査スコアに与える影響を検証すること。特に `Financial_Creditworthiness` の事後確率分布が既存案件で大きく変化していないか確認を推奨。
- **rule-validator**: `bayesian_engine.py` の `BN_EDGES` および CPT 整合性チェックを実施すること。`_prob_financial_creditworthiness(*c)` のアンパック順序が `evidence` リストの順序と一致しているか要確認。
- **code-reviewer**: `bayesian_engine.py` の論理変更部分のレビューを推奨。UIファイル群（18件）は機械的置換のみのため優先度低。
- **build-runner**: Streamlit の新 API `width='stretch'` が requirements に定義されたバージョンで動作するか確認を推奨。
