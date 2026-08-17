# 予測的フレームワーク導入計画（REV-297）

理由: 「予測」がこのリポジトリに3系統バラバラに存在し、どれも「外れたかどうか」を測られていない。新しい予測器を足すのではなく、既存の3系統を「予測 → 観測 → 誤差 → 更新候補 → 人間レビュー」の一本の輪に繋ぐための計画を正本化する。
適用条件: `api/prediction_error_loop.py` / `timesfm_engine.py` / `future_simulation.py` / `montecarlo.py` / `api/shion_proactive_alert.py` に関わる変更を設計・実装する時。
削除条件: 下記 Phase 1〜4 がすべて完了し、`reports/predictive_framework_latest.md` が日次で安定して出るようになった時（その時点でこの文書は `docs/loop_engineering_map.md` の1行へ縮約する）。

これは**計画文書**であり、実装ではない。この文書の追加だけではスコアリング・API・UI は一切変わらない。

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

想定 PR 分割:

| PR | 内容 | REV |
|---|---|---|
| 1 | この計画文書 | REV-297 |
| 2 | Phase 1: 予測スナップショット | 別 REV |
| 3 | Phase 2: 観測レポート + 日次接続 | 別 REV |
| 4 | Phase 3: 数値予測の API 化 + UI | 別 REV |
| 5 | Phase 4: 先回り提示 | 別 REV |

---

## 7. 未確定事項（着手前に要確認）

1. **接続点**: 予測スナップショットを取るのは `/api/score/calculate`（スコア計算時）か `/api/cases/register`（案件登録時）か。前者は取りこぼしが少ないが試し打ちも全部記録される。後者は綺麗だが登録しない検討案件が落ちる。→ **案件登録時を推奨**（誤差を測れるのは結果登録される案件だけのため）
2. **粒度**: 案件単位のみか、チャット回答（`response_impact_predictions.jsonl`）も同じレポートに含めるか
3. **Phase 3 の優先度**: 数値予測の本流化は工数が最も大きい。Phase 4（先回り提示）を先に出す選択肢もある
