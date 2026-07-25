# Bias Extraction Pipeline Plan（協議コメント→人間判断バイアス抽出）

## Purpose

審査員が書く協議コメント（承認/否決の所見、判定を変えた理由）から、人間特有の判断バイアス
（楽観バイアス・ストーリーによる免責・属性フォビア・権威バイアス）とスルーされたリスクを
自動抽出し、既存の「判断資産候補」パイプラインへ接続する。

`planning/data_to_judgment_asset_plan.md` の設計思想（Recall→Transform→Evaluate→Crystallize）
と `data/canonical_judgment_rules.json` の正規スキーマをそのまま踏襲し、新しい
`material_type: "cognitive_bias"` として合流させる。

## Status: Phase A applied（REV-230, PR #632, merged）

- `api/crystallizer/bias_extractor.py`: `judgment_feedback.reason` 等の協議コメント1件を
  Gemini APIに渡し、`overall_sentiment` / `highlighted_factors` / `dismissed_risks` /
  `inferred_biases` を構造化JSONで抽出。`pattern_synthesizer.py`と同じGemini REST呼び出し
  規約、`gunshi_gemini.py`と同型の429リトライ、`api/llm_json_guard`でのJSON抽出を踏襲。
  APIキー未設定・失敗時は捏造しない安全側フォールバック（`bias_type=NONE`）。会社名・個人名の
  露出を減らすベストエフォートのマスク処理も実装済み。
- `scripts/build_bias_extraction_candidates.py`: `judgment_feedback`テーブルをチェックポイント
  方式で増分スキャンし、`dismissed_risk`/`bias_pattern`候補を
  `data/bias_extraction_candidates.jsonl` + `data/bias_extraction_candidate_state.json` に
  出力（`build_autoresearch_judgment_asset_candidates.py`と同型のstate管理・JSONL方式）。

未着手（このPhase Aではスコープ外とした）:
- レビュー用API（Phase B）
- フロントエンドUI（Phase C）
- `shion_screening_reviews.review_text` 等、`judgment_feedback`以外のソースの取り込み

### データソースに関する気づき（2026-07-25, 要検討）

紫苑は審査員が協議コメントを書面化する**前の相談相手**としても使われている。現在Phase Aが
材料にしている `judgment_feedback.reason` は、人間がすでに判断を確定させた後に書く
「決めた後の正当化文」であり、`STORY_EXCUSE`のようなバイアスは相談の途中段階（結論に
染まる前）でこそ素の形で出ている可能性が高い。

→ 紫苑との相談ログ（`AI Chat/Cloud Run Conversation Log`、チャット履歴）を将来的な
データソース候補として検討する価値がある。`reason`（決断後の言い訳）と相談ログ（決断前の
揺れ）は別の性質の材料であり、どちらも判断資産候補の`source_table`として区別して扱う設計に
すること。

## Phase B: レビュー用API（未着手）

着手前に3文計画を提示し承認を得ること（CLAUDE.md Plan-First Checkpoint）。

- `GET /api/bias-extraction-candidates` — `data/bias_extraction_candidates.jsonl`を読んで
  返す。既存の `/api/judgment-asset-candidates/screening`（`api/main.py:12479`）と同構造。
- `POST /api/bias-extraction-candidates/{id}/feedback` — 既存の
  `JudgmentAssetCandidateFeedbackRequest`（`api/main.py:8906`）と同じ
  `Literal["useful","neutral","rejected"]`パターンで、
  `_update_autoresearch_judgment_asset_candidate_feedback`相当の更新関数をこの候補ファイル
  向けに実装。JSONL側の`use_count`等をこのタイミングで直接更新し、バッチ再実行時の
  埋め込みstateフィールドとの不整合を避ける設計にすること（Phase Aでは未解決の既知の制約）。
- 1件オンデマンド分析用に `POST /api/bias-extraction/analyze`（`comment_text`, `case_id`を
  受け取り`extract_bias()`を直接叩く）も追加し、バッチとAPIで処理コアを完全共有する。

## Phase C: フロントエンドUI（未着手）

- `frontend/src/app/judgment-review/page.tsx` に近い一覧＋承認/却下UIを新設
  （例: `bias-review/page.tsx`）。Phase Bの2エンドポイントを`apiClient`経由で呼ぶ。
- スコープが大きいため、Phase B完了後に改めて3文計画→承認のフローで着手する。

## Phase D: リアルタイム相談中のバイアス気づき（将来像・未確定）

現状（Phase A〜C）は、書き終えた協議コメントを事後的に紫苑がレビューして反応する
「後追い型」に留まる。将来的な狙いは、相談がまだ進行中の段階で紫苑が
「今、CF赤字を先行投資という理由で流していませんか」のように、その場でバイアスの兆候に
気づかせる「介入型」への発展。

まだ先の話であり、現時点では設計を確定しない。少なくとも以下がPhase A〜Cとは別の検討が
必要になる:
- 事後バッチ分析（`extract_bias()`を1コメントに対して実行）とは異なり、対話の途中経過に
  対して低レイテンシで動く必要がある（相談を止めない）
- 誤検知時に相談の流れを妨げない出し方（強い断定ではなく「気づき」レベルの投げかけ）
- リアルタイム対話フロー（`lease_intelligence_*.py`系）側への組み込みが前提になるため、
  そちらのPlan-First Checkpointも別途必要

## 対象外・注意点（継続）

- `scoring_core.py`やスコア判定ロジックには一切触れない。
- `constants.APPROVAL_LINE`等の閾値をこの機能内で複製・ハードコードしない。
- PRタイトルには対応するREV番号を含め、`scripts/improvement_ledger.jsonl`に登録すること。
