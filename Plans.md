# Plans

## Active Tasks

<!-- タスクをここに追加 -->

---

## Phase 3: AURION CORE — 量子干渉ビジュアライゼーション

作成日: 2026-04-26
出典: AURION_CORE_Visual_Instruction.pdf (v1.0)

### 背景・目的

「データの歪み」を審査担当が直感的に把握できるよう、2変数を **サイン波** として表現し、
その干渉（共鳴 / デコヒーレンス）をリアルタイムアニメーションで可視化する。

- **共鳴 (Coherence)**: 波が重なり輝く → ビジネス実態が健全
- **干渉 (Decoherence)**: 波が打ち消し合い平坦 → 高スコアでも実体が伴わない警告

既存の `quantum_analysis_module.py` のペア定義・スコアを視覚層として昇華させる。

> ターゲット環境: iMac 2019 / Python 3.10+ / Flet + NumPy / 60fps 目標

---

### Phase 3-A: 波形エンジン（コア）

| Task | 内容 | DoD | Depends | Status | 推定 |
|------|------|-----|---------|--------|------|
| AV.1 | `aurion_wave_engine.py` 作成 — NumPy で2変数の位相差からサイン波合成・振幅スコア (0〜1) を返す `compute_wave(v1, v2, freq, t)` | `pytest tests/test_aurion_wave.py` 全 pass・位相差 π で振幅 ≈ 0 になる | - | cc:完了 | small(2h) |
| AV.2 | `aurion_phase_mapper.py` 作成 — 業種コード→干渉ペア自動選択（既存 `PAIR_WEIGHTS` を流用） | `PhaseMapper.get_pair("C")` → `("op_profit","depreciation")` が返る pytest pass | AV.1 | cc:完了 | small(1h) |
| AV.3 | `tests/test_aurion_wave.py` 作成 — エンジン・マッパーの単体テスト | 共鳴・デコヒーレンス境界値・業種マッパー全ケース pass | AV.1, AV.2 | cc:完了 | small(2h) |

---

### Phase 3-B: Flet スタンドアロン版（リアルタイムアニメーション）

| Task | 内容 | DoD | Depends | Status | 推定 |
|------|------|-----|---------|--------|------|
| AV.4 | `flet_aurion_wave.py` 作成 — Flet Canvas でサイン波・合成波をリアルタイム描画（x軸方向に波が流れる） | `python3 flet_aurion_wave.py` でウィンドウ起動・波形アニメーション確認 | AV.1, AV.2 | cc:完了 | medium(4h) |
| AV.5 | スライダー・デバッグ・パネル — 各変数値をスライダーで変更 → 波形リアルタイム更新 | スライダー操作で振幅変化が即時反映される（1フレーム以内） | AV.4 | cc:完了 | medium(3h) |
| AV.6 | デコヒーレンス警告演出 — 位相差 > 閾値（π*0.7）で波が消えて赤い警告テキスト「⚠ データの歪みを検出」を表示 | 閾値超えで視覚的に波形消失・警告表示；閾値以内で波形復活 | AV.4, AV.5 | cc:完了 | small(2h) |

---

### Phase 3-C: Streamlit 埋め込み版（既存 UI 統合）

| Task | 内容 | DoD | Depends | Status | 推定 |
|------|------|-----|---------|--------|------|
| AV.7 | `components/aurion_wave_view.py` 作成 — Plotly で静的波形（1スナップショット）と干渉スコアを表示 | `render_aurion_wave_view()` が現在案件の Q_risk・振幅グラフを描画する | AV.1, AV.2 | cc:完了 | medium(3h) |
| AV.8 | サイドバー登録 — `sidebar.py` の「🔬 実験的機能」に `"⚛️ 量子波形ビジュアル"` を追加し、`lease_logic_sumaho12.py` にルーティング追加 | メニューから遷移して AV.7 画面が表示される | AV.7 | cc:完了 | small(1h) |
| AV.9 | 現在審査案件との連動 — `score_calculation.py` の結果から `aurion_wave_view` へ自動でデータを渡し、Q_risk と波形を並列表示 | 審査後レポート画面でも「量子波形」タブが追加される | AV.7, AV.8 | cc:完了 | medium(3h) |

---

### 推奨着手順

**短期（Phase 3-A）**: AV.1 → AV.2 → AV.3
→ エンジンとテストを固める。TDD で進める。

**中期（Phase 3-B）**: AV.4 → AV.5 → AV.6
→ Flet スタンドアロン版を完成させてデモ可能にする。

**後期（Phase 3-C）**: AV.7 → AV.8 → AV.9
→ 既存 Streamlit UI に統合し、審査担当が日常使いできるようにする。

---



## Phase 2: 量子解析 説明可能性・未知変数探索

作成日: 2026-04-26

### 背景・目的

Q.1〜Q.11（第1フェーズ）完了後の次段階。
審査担当が「なぜこのスコアか」を理解できる説明可能性と、
現在の変数セットで捉えられていない潜在リスクの検出を目指す。

> ⚠️ データ量の注意: 現状 ~40件。分位数表示は「n=XX件（参考値）」を必ず明示すること。

---

### Phase 2-A: 説明可能性の土台（精度土台）

| Task | 内容 | DoD | Depends | Status | 推定 |
|------|------|-----|---------|--------|------|
| EX.1 | `pair_contributions` を `predict()` 戻り値に追加（加法的寄与点数） | `sum(pair_contributions.values())` が `explained_risk` と一致する pytest pass | - | cc:完了 | small(2h) |
| EX.2 | `quantum_explainer.py` 新規作成・`QuantumExplainer` クラス骨格 + `shapley_contributions()` | predict 戻り値を受け取り加法分解 dict を返す; `test_explainer_sums_to_risk` pass | EX.1 | cc:完了 | small(2h) |
| EX.3 | OOD 検出: `QuantumExplainer.ood_check(rec)` — \|z\| > 2.0 で外挿フラグ | fit 済みモデルで極端値（業種平均の10倍）入力 → `True` が返る pytest pass | EX.2 | cc:完了 | small(2h) |
| EX.4 | `predict()` 戻り値に `ood_flags`・`explained_risk`・`entropy_risk` を追加（後方互換） | 既存テスト全 pass・新フィールドが dict に存在する | EX.3 | cc:完了 | small(1h) |
| SC.1 | `quantum_screener.py` 作成 — DB の未使用変数（売上高・総資産・借入金等）と失注の相関を CSV 出力 | `nenshu`, `bank_credit`, `other_assets` 等の相関係数・p値が CSV に出力される | - | cc:完了 | medium(4h) |
| SC.2 | `residual_signal` 計算: エントロピー寄与を分離した 2 軸スコア（`explained_risk` + `entropy_risk`） | predict 戻り値の 2 値の合計が `quantum_risk` と 0.01 以内で一致 | EX.1 | cc:完了 | small(2h) |

---

### Phase 2-B: 説明可能性 UI

| Task | 内容 | DoD | Depends | Status | 推定 |
|------|------|-----|---------|--------|------|
| EX.5 | 反事実説明 `counterfactual(case, var, target_val)` — 1 変数を差し替えて Q_risk 差分を返す | `test_counterfactual_balanced_machines_lowers_risk` pass・注記「数値操作推奨ではない」をUI必須明示 | EX.2 | cc:完了 | medium(4h) |
| EX.6 | 業種内分位数 `industry_percentile(var, value, cases)` — n<5 は None 返却 | n≥5 で 0〜100 の値・n<5 で None; pytest pass | - | cc:完了 | small(2h) |
| EX.7 | 自然言語レポート `build_narrative()` — テンプレートベース（LLM 不要） | 「depreciation が業種平均より X% 乖離。建設業典型リスク。寄与: +18.4点」形式の文字列を返す | EX.1, EX.5, EX.6 | cc:完了 | medium(6h) |
| UI.1 | `report.py` の量子セクションに貢献度横棒グラフ（Plotly）追加 | レポート画面で各ペアの寄与点数が棒グラフで表示される | EX.2 | cc:完了 | medium(4h) |
| UI.2 | `industry_quantum_view.py` に反事実パネル（変数スライダー → Q_risk リアルタイム更新）追加 | スライダー操作で Q_risk が動的に変化する・反事実注記あり | EX.5 | cc:完了 | medium(6h) |
| UI.3 | OOD 警告バッジを `report.py` と `industry_quantum_view.py` に表示 | `ood_flags` に True があると「⚠️ 外挿域」バッジが表示される | EX.4 | cc:完了 | small(2h) |
| UI.4 | 自然言語レポートを `report.py` の量子セクションに統合 | レポート画面に「量子解析コメント」テキストエリアが表示される | EX.7 | cc:完了 | small(2h) |

---

### Phase 2-C: 未知変数探索

| Task | 内容 | DoD | Depends | Status | 推定 |
|------|------|-----|---------|--------|------|
| SC.3 | `components/variable_screener_view.py` 作成 — SC.1 を Streamlit UI として公開 | 画面から「未使用変数スクリーニング実行」ボタン → 相関係数の棒グラフが表示される | SC.1 | cc:完了 | medium(5h) |
| SC.4 | 高残差 + 失注案件への自動フラグ — `residual_signal > 閾値` かつ失注を「未知リスク候補」としてテーブル表示 | `industry_quantum_view.py` のテーブルに「未知リスク候補」列が追加される | SC.2 | cc:完了 | small(2h) |
| SC.5 | スクリーニング結果から新規ペア候補を `quantum_config.json` の `candidate_pairs` セクションに出力 | `python3 quantum_screener.py --suggest` 実行後、config に候補ペアが追記される | SC.1 | cc:完了 | medium(4h) |

---

### 推奨着手順

**短期（2〜3週間）**: EX.1 → EX.2 → EX.3 → EX.4 → SC.2
→ predict() の出力が豊かになり、UI 実装の土台が整う。TDD で進める。

**中期（次の2〜3週間）**: EX.5 → EX.6 → EX.7 → UI.1 → UI.3 → UI.4
→ 審査担当が「なぜか」を画面で読めるようになる。

**後期（SC フェーズ）**: SC.1 → SC.3 → SC.4 → SC.5
→ 「項目外の変数」の候補が可視化され、次の改善サイクルに繋がる。

---

## Backlog（第1フェーズ完了済み）

### 量子解析モジュール改善 Q.1〜Q.11

| Task | 内容 | Status |
|------|------|--------|
| Q.1  | `_fitted` 検証強化 | cc:完了 |
| Q.2  | 閾値を `quantum_config.json` に外部化 | cc:完了 |
| Q.3  | 業種別合成データテンプレート | cc:完了 |
| Q.4  | スコア≥70 拡張 | cc:完了 |
| Q.5  | 業種別ペア重み自動発見 | cc:完了 |
| Q.6  | quantum_risk → スコアフィードバック | cc:完了 |
| Q.7  | 🔁 二次審査推奨バッジ + CSV | cc:完了 |
| Q.8  | エッジケーステスト追加 | cc:完了 |
| Q.9  | モデルバージョン管理 | cc:完了 |
| Q.10 | フィードバックボタン + 学習反映 | cc:完了 |
| Q.11 | ドキュメント補強 | cc:完了 |

---

## Completed

- 業種別量子解析モジュール統合（CSV/レポート/ドキュメント） — commit bf9e6c6
- 量子解析 Q.1〜Q.11 全改善バックログ — commit 2fa94a2
