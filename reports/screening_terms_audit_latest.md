# Screening Terms Audit

- generated_at: `2026-09-17T22:07:07`
- status: `ok`
- scanned_files: `588`
- guardrail: `read_only_terms_audit_no_scoring_or_db_change`

## Counts

- warn: `0`
- review: `33`
- ok: `2523`

## Glossary

- `actual_pd`: pd_percent が明示される場合だけPDとして扱う。欠落・0は未算出扱い。
- `high_risk_similarity`: default_warnings は高リスク格付先との財務類似度。実PDではない。
- `q_risk`: Q_risk / quantum_risk は財務・入力整合性の論点分解センサー。自動減点ではない。
- `score`: スコアは総合判断の入口。PDやQ_riskと同一視しない。

## Warn / Review Findings

- `review` `actual_pd` `api/game_theory/negotiation.py:46` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `default_prob = max(0.0, 1 - (credit_score / 100) * 0.8 - collateral_ratio * 0.2)`
- `review` `actual_pd` `api/game_theory/negotiation.py:47` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `safety_utility = 1.0 - default_prob`
- `review` `actual_pd` `api/outcome_drift_loop.py:7` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `帯ごとの延滞・デフォルト率を集計する`
- `review` `actual_pd` `api/outcome_drift_loop.py:89` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `この集計を見て、「本来低リスクなはずの帯で延滞・デフォルト率が高い」`
- `review` `actual_pd` `api/shion_conscience.py:85` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `cautions.append("デフォルト確率が高い場合は、承認可否より先に返済原資の説明可能性を見る。")`
- `review` `actual_pd` `frontend/src/app/faq/page.tsx:298` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `q: 'PDとAIスコアの関係は？',`
- `review` `score` `frontend/src/app/faq/page.tsx:298` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `q: 'PDとAIスコアの関係は？',`
- `review` `actual_pd` `report_generator.py:17` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `result.time_series_default_prob : 累積デフォルト確率の時系列 (np.ndarray)`
- `review` `actual_pd` `report_generator.py:46` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"""デフォルト確率・財務スコア・リース依存度等を文章化する（テンプレート方式）。"""`
- `review` `score` `report_generator.py:46` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"""デフォルト確率・財務スコア・リース依存度等を文章化する（テンプレート方式）。"""`
- `review` `actual_pd` `report_generator.py:48` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `prob = getattr(result, "default_prob", None)`
- `review` `actual_pd` `report_generator.py:212` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"""モンテカルロシミュレーション結果（デフォルト確率）を文章化する（テンプレート方式）。"""`
- `review` `actual_pd` `report_generator.py:213` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `prob = getattr(result, "default_prob", None)`
- `review` `actual_pd` `report_generator.py:249` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `ts = getattr(result, "time_series_default_prob", None)`
- `review` `actual_pd` `report_generator.py:254` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `f"\n  （参考）シミュレーション期間 {months}ヶ月後の累積デフォルト確率: "`
- `review` `actual_pd` `report_generator.py:396` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `if default_prob < 0.05:`
- `review` `actual_pd` `report_generator.py:398` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `elif default_prob < 0.15:`
- `review` `actual_pd` `report_generator.py:400` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `elif default_prob < 0.30:`
- `review` `actual_pd` `report_generator.py:405` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `risk_level, default_prob = "低リスク", None`
- `review` `actual_pd` `report_generator.py:407` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `risk_level, default_prob = "中リスク", None`
- `review` `actual_pd` `report_generator.py:409` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `risk_level, default_prob = "高リスク", None`
- `review` `actual_pd` `report_generator.py:411` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `risk_level, default_prob = "極高リスク", None`
- `review` `actual_pd` `report_generator.py:425` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `default_prob=default_prob,`
- `review` `actual_pd` `report_generator.py:428` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `time_series_default_prob=None,`
- `review` `actual_pd` `scripts/build_shion_memory_index.py:38` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"PD",`
- `review` `actual_pd` `scripts/build_shion_memory_index.py:45` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"モデル性能、スコア差分、再学習方針、PD表示を確認する時",`
- `review` `score` `scripts/build_shion_memory_index.py:45` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"モデル性能、スコア差分、再学習方針、PD表示を確認する時",`
- `review` `actual_pd` `scripts/cleanup_improvement_reviews_data.json:28` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"REV-061": "PD表示の明確化",`
- `review` `actual_pd` `scripts/cleanup_improvement_reviews_data.json:36` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"REV-085": "PD表示色分け",`
- `review` `actual_pd` `scripts/sync_implemented_to_obsidian.py:32` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"REV-041":             ["PD表示箇所の明確化"],`
- `review` `actual_pd` `shinsa_gunshi_logic.py:508` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"""スコアと PD から事前確率を算出。"""`
- `review` `score` `shinsa_gunshi_logic.py:508` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `"""スコアと PD から事前確率を算出。"""`
- `review` `actual_pd` `shinsa_gunshi_logic.py:661` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい
  - `# ── 状況マッチ：PD高い（要審議・否決圏）→ 逆転・数値証明系を優先 ──`
