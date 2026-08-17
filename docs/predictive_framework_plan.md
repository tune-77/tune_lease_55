# 予測的フレームワーク導入計画（REV-297）

理由: 「予測」がこのリポジトリに3系統バラバラに存在し、どれも「外れたかどうか」を測られていない。新しい予測器を足すのではなく、既存の3系統を「予測 → 観測 → 誤差 → 更新候補 → 人間レビュー」の一本の輪に繋ぐための計画を正本化する。
適用条件: `api/prediction_error_loop.py` / `timesfm_engine.py` / `future_simulation.py` / `montecarlo.py` / `api/shion_proactive_alert.py` に関わる変更を設計・実装する時。
削除条件: 下記 Phase 1〜4 がすべて完了し、`reports/predictive_framework_latest.md` が日次で安定して出るようになった時（その時点でこの文書は `docs/loop_engineering_map.md` の1行へ縮約する）。

## 決定事項（2026-08-17）

計画時に未確定だった3点は、以下で確定した。

| 論点 | 決定 | 理由 |
|---|---|---|
| 予測スナップショットの接続点 | `/api/score/full` の DB保存直後 | 調査の結果、`/api/cases/register` は**結果登録**（成約/失注）のエンドポイントで、案件が最初に生成されるのは `/api/score/full` の `save_case_log()` だった。ここが「case_id が発行され、かつ結果をまだ知らない」唯一の瞬間になる |
| レポートの粒度 | 案件単位のみ | チャット側 `response_impact_predictions.jsonl` が予測しているのは「言葉がどう響くか」で、案件側の「成約するか」とは的が違う。同じキャリブレーション表に混ぜると両方の意味が濁る |
| フェーズ順序 | 1 → 2 → 4 → 3 | Phase 4 は Phase 2 の出力を読むだけで軽く、Phase 3 は計算部の分離が必要で最も重い |

Phase 1〜4 はすべて実装済み（PR分割は当初案から変更し、1ブランチ内でフェーズごとにコミットを分けた）。以下の計画本文は設計意図の記録として残す。

---

## 1. 現状：予測は3.5系統に散っている

| 層 | 既存資産 | できていること | 欠けていること |
|---|---|---|---|
| 判断予測 | `api/prediction_error_loop.py`、`api/routers/judgment_assets.py` | 結果登録時に予測誤差を分類し、信念更新候補を `data/prediction_error_update_candidates.jsonl` へ出す。人間レビュー台帳あり | **事前予測が保存されていない**。`build_case_result_prediction_error_payload()` は結果登録の瞬間に `case_data` からスコアを逆算して「予測だったことにしている」 |
| 数値予測 | `timesfm_engine.py`、`future_simulation.py`、`montecarlo.py`、`montecarlo_pricing.py` | GBM / TimesFM による将来売上・営業利益のシミュレーション | 呼び出し元が `components/analysis_results.py`（Streamlit）だけ。Next.js 本流の審査画面に出ていない。**当たったかどうかを誰も測っていない** |
| 先回り提示 | `api/shion_proactive_alert.py` | ログのエラー急増を検知して割り込み発言 | 対象が運用ログのみ。判断領域（この案件で何が外れそうか）は予測していない |
| 回答予測（0.5系統） | `api/chat_language_feedback.py: record_response_impact_prediction()` | 回答が相手にどう響くかを `shadow_only` で事前記録。`data/response_impact_predictions.jsonl` | 案件審査側には同じ仕組みが無い |

### 一番の穴

`api/prediction_error_loop.py:175` の `build_case_result_prediction_error_payload()` は、結果登録時にこう予測を組み立てている。

```python
"confidence": None if score is None else min(0.95, max(0.35, abs(score - 60.0) / 60.0 + 0.35)),
```

これは「承認ラインから遠いほど自信がある」という代理指標を、**結果がわかった後に**計算しているだけで、審査時点の予測ではない。つまり現状の予測誤差ループは、後半（観測・誤差・候補化）だけが実装され、前半（予測の事前固定）が存在しない。

そして `api/chat_language_feedback.py` には、**チャット側だけ**に正しい形（事前に予測して `shadow_only` で寝かせる）が既にある。案件審査側にこの型を移植するのが最短ルートになる。

---

## 2. 目指す形

```
審査時点                    結果登録時                 翌朝
   │                          │                        │
   ├─ 予測スナップショット ──→ ├─ 観測（成約/失注/延滞）  │
   │  ・判断予測（risk/懸念）  │  ↓                      │
   │  ・数値予測（3〜5期）     ├─ 予測誤差の分類 ────────→ ├─ 観測レポート
   │  ・confidence の根拠      │  ↓                      │  ・カバー率
   │                          ├─ 信念更新候補           │  ・キャリブレーション
   │                          │                        │  ・繰り返し外れる前提
   │                          ↓                        ↓
   └──────────────── 人間レビュー ←──── 先回り提示（次の案件で警告）
```

自動で回るのは「記録・集計・候補化」まで。**採否は人間**。

---

## 3. 設計原則（`docs/loop_engineering_map.md` を継承）

- 正本ファイルを増やさない。既存の `data/prediction_error_events.jsonl` 系を使う
- 予測は必ず**事前に**固定して保存する。後付け再構成を正本にしない（フォールバックとしては残す）
- 自動反映しない。更新候補・レポートまでで止める
- `scoring_core.py` の重みと `APPROVAL_LINE` は触らない。閾値を参照する時は必ず `scoring_core` から import する
- 予測記録の失敗がスコアリング応答を壊さないこと（fire-and-forget、`try/except` で握る。既存の `api/main.py:1876` 周辺の `prediction_error_result` と同じ扱い）
- 新規ファイルは `api/chat_language_feedback.py` の記録形式（`schema_version` / `status: shadow_only` / `fingerprint` / `use_policy`）に揃える

---

## 4. フェーズ計画

### Phase 1: 事前予測の固定 ― 欠けている「前半」を作る

**目的**: 審査時点の予測を、結果を知る前に保存する。

**触るファイル**
- 新規 `api/prediction_snapshot.py`
- `api/prediction_error_loop.py`（スナップショット優先のフォールバック分岐のみ）
- `api/main.py`（`/api/cases/register` の記録箇所付近。既存の `prediction_error_result` と同じ形で呼ぶ）

**やること**
1. `record_prediction_snapshot()` を追加し、`data/prediction_snapshots.jsonl` へ追記する。保存するのは `case_id` / `captured_at` / `score` / `hantei` / `risk_level` / `main_concern` / `recommended_action` / `confidence` / `confidence_basis` / `used_judgment_assets` / `model_version` / `source` / `schema_version` / `status: shadow_only`
2. `build_case_result_prediction_error_payload()` を「スナップショットがあればそれを使う。無ければ現行の逆算にフォールバック」へ変更する。既存の逆算ロジックは削除しない（過去案件が壊れるため）
3. `confidence` の**式は変えない**。代わりに `confidence_basis` に「何を根拠にした値か」（score 距離、`completeness_ratio`、`used_default_asset_score`、類似事例件数）を記録するだけに留める

**やらないこと**: スコア計算そのものへの介入、confidence 式の変更、UI 変更。

**完了条件**: 結果登録した案件で、`prediction_error_events.jsonl` の `prediction` が審査時点の値と一致する。スナップショットの無い過去案件は従来通り動く。

**テスト**: `tests/test_prediction_error_loop.py` に追記（スナップショット有り／無しの両経路）。

**リスク**: 記録処理がスコアリング応答をブロックする → 例外を握って `{"status": "error"}` を返すだけにする。

---

### Phase 2: 誤差の集計と観測レポート

**目的**: 「予測がどれだけ当たっているか」を朝1枚で見られるようにする。

**触るファイル**
- 新規 `scripts/build_predictive_framework_report.py`（読み取り専用）
- `run_daily_improvement_pipeline.sh`（**追記のみ・`|| true` 付き**）

**出力**: `reports/predictive_framework_latest.{json,md}`

**指標**

| 指標 | 意味 |
|---|---|
| `prediction_coverage` | 結果登録済み案件のうち事前スナップショットがある割合 |
| `calibration` | 予測 risk_level 別の実際の成約率・失注率（高リスク予測が本当に失注しているか） |
| `surprise_rate` | 予測誤差が大きい案件の割合 |
| `repeat_belief_top` | 同じ前提が繰り返し外れている上位（`_existing_repeat_count` を再利用） |
| `candidate_review_rate` | 更新候補のうち人間が採否を付けた割合 |

**完了条件**: 日次 post で1枚出る。自動反映は無し。

**リスク**: パイプライン停止（`CLAUDE.md` の要注意領域）→ 追記のみ・`|| true` を厳守。

---

### Phase 3: 数値予測の本流化

**目的**: Streamlit に閉じている将来財務シミュレーションを Next.js の審査結果へ出し、**その予測も採点対象にする**。

**触るファイル**
- `future_simulation.py` / `montecarlo.py` の計算部を UI から分離（`render_*` と計算関数の切り離し。Streamlit 側の呼び出しは壊さない）
- 新規 API（`api/routers/analytics.py` に追加。既存 `/api/forecast` と並べる）
- `frontend/src/app/` の審査結果画面（ファンチャートは `frontend/src/app/timesfm/page.tsx` の `buildFanChartData()` を再利用）

**注意点**
- 単位: フロント **百万円** → `toThousandYenPayload()`（×1000）→ モジュール内 **千円**
- TimesFM は**オプションのまま**。`api/main.py:99` の `_load_timesfm_engine()` と同じ遅延ロード方針を守る（PyTorch/MPS を startup から外す）
- 予測結果を Phase 1 のスナップショットへ含め、Phase 2 のキャリブレーション対象にする

**完了条件**: 審査結果から3〜5期の見通しが見え、その予測が後で採点される。

**リスク**: Streamlit 側の回帰。→ 計算関数の分離のみで、既存の `render_future_simulation_ui()` の振る舞いは変えない。

---

### Phase 4: 先回り提示

**目的**: 聞かれる前に「この案件、同じ前提で直近3件外している」を出す。

**触るファイル**
- `api/shion_proactive_alert.py`（判断予測レーンを追加。既存のエラー急増検知とは別関数）
- `frontend/src/app/judgment-review/page.tsx`

**出すもの**: Phase 2 の `repeat_belief_top`、キャリブレーション崩れ、寝ている active ルール。

**完了条件**: 案件を開いた時に、該当があれば警告が1件出る。該当が無ければ何も出ない（沈黙をデフォルトにする）。

**リスク**: 過剰通知で無視される → 閾値は「同一前提が3回以上外れ」から始め、緩めない。

---

## 5. やらないこと（明示）

- スコアリング本体の重み・`APPROVAL_LINE` の自動変更
- 予測誤差からのモデル自動再学習
- 判断資産の自動昇格
- 新しい正本ファイルの乱立
- `eslint --fix`（`CLAUDE.md` 絶対禁止）
- Phase をまたいだ一括 PR

---

## 6. 進め方の提案

**Phase 1 だけを先に1PRで出す。** 朝レポートに `prediction_coverage` の実数が出てから Phase 2 以降の要否を判断する。カバー率が上がらないなら、そもそも接続点が間違っているので Phase 3・4 を作っても無駄になる。

実装は REV-297 の1ブランチにまとめ、フェーズごとにコミットを分けた。

| コミット | 内容 | 主なファイル |
|---|---|---|
| 1 | この計画文書 | `docs/predictive_framework_plan.md` |
| 2 | Phase 1: 予測スナップショット | `api/prediction_snapshot.py`、`api/prediction_error_loop.py`、`api/main.py` |
| 3 | Phase 2: 観測レポート + 日次接続 | `scripts/build_predictive_framework_report.py`、`scripts/run_daily_improvement_post.sh` |
| 4 | Phase 4: 先回り提示 | `api/shion_proactive_alert.py`、`frontend/src/app/judgment-review/page.tsx` |
| 5 | Phase 3: 数値予測の本流化 | `future_simulation_core.py`、`api/routers/analytics.py`、`frontend/src/components/analysis/FutureSimulationPanel.tsx` |

---

## 7. 実装後に残っている限界

1. **数値予測は採点できていない**。将来売上・営業利益の予測は `data/prediction_numeric_forecasts.jsonl` に記録し件数は観測しているが、採点には3〜5期後の実績値が必要で、現在のDBは成約/失注と延滞しか持っていない。レポート側も `calibratable: false` / `calibration_blocker: future_actuals_not_collected` と明示している。実績財務を取り込む経路ができるまでキャリブレーションはしない
2. **スナップショットを取るのは `/api/score/full` の経路だけ**。`api/routers/debate.py`、`api/routers/pipeline_misc.py`、バッチ取り込み経由で作られた案件には事前予測が付かず、結果登録時の逆算にフォールバックする。`prediction_coverage` はこの取りこぼしを含めた実数として読む
3. **confidence はまだ代理指標**。`abs(score - CONDITIONAL_LINE) / 60 + 0.35` は「承認ラインから遠いほど自信がある」以上の意味を持たない。`confidence_basis`（completeness_ratio / used_default_asset_score / quantum_risk 等）を記録しているので、実データが溜まってから式を見直す
4. **キャリブレーションは事前予測が溜まるまで参考値**。`prediction_coverage.trustworthy` が false の間は、先回り提示（Phase 4）も意図的に沈黙する
