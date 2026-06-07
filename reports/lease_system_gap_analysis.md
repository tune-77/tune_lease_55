# Lease System Gap Analysis

> Generated: 2026-06-08 07:41 | mode: read-only diagnostics

## Summary
- Total gaps: 8
- Critical: 1 / High: 3 / Medium: 3 / Low: 1

## Recommended First Program Track
1. Data/scoring audit warnings: do not change scores automatically; surface warnings only.
2. RAG eval expansion: add more expected-path cases and run retrieval checks daily.
3. Improvement triage: reduce needs_review backlog into a weekly focus list.

## Gaps

### GAP-004: スコア根拠・データ品質・モデル信頼性の継続監査が不足
- Priority: **critical**
- Category: `risk-scoring`
- Impact: 審査判断の説明性とモデル信頼性を損なう。特に再学習や過去案件比較に影響する。
- Recommended action: スコア計算・学習データ・DB品質を毎回読む監査プログラムを独立運用し、本体画面には警告のみ表示する。
- Suggested program: `scripts/lease_system_gap_analyzer.py の scoring/data checks 拡張`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - Sidecar reports mention test/dummy data, overfitting, or decision logic inversion.
- Source refs:
  - `reports/agent_sidecar_brief.md`

### GAP-001: 改善候補が滞留し、優先順位と着手可否が見えにくい
- Priority: **high**
- Category: `improvement-ops`
- Impact: 重要なUI・データ・審査改善が埋もれ、毎日の改善パイプラインが提案過多になる。
- Recommended action: needs_review を、審査精度・データ品質・RAG・UI・運用に分類し、各カテゴリから最大3件だけ今週のFocusに昇格する。
- Suggested program: `scripts/lease_system_gap_analyzer.py + reports/lease_system_gap_analysis.md`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - reports/latest.json needs_review_count=37
  - 上位候補: ホーム画面のカスタマイズ機能追加 / 知識宇宙マップの立体表示・球体化 / 改善パイプラインログ画面で承認ボタンと一覧表示がない。 / スマホアプリのホーム画面で文字表示がズレている / ニュースデータの審査への活用
- Source refs:
  - `reports/latest.json`

### GAP-002: 高リスク改善項目の扱いが手動確認止まり
- Priority: **high**
- Category: `governance`
- Impact: ポートフォリオ・スコア・データ系の改善が安全ゲート不足で放置されやすい。
- Recommended action: 高リスク項目はSPEC化して、承認ゲート・テスト・ロールバック条件を先に作る。
- Suggested program: `scripts/gen_tests_from_spec.py と連携した SPEC skeleton 生成`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - high-risk needs_review=7
- Source refs:
  - `reports/latest.json`
  - `docs/plan.md`

### GAP-008: フロントエンドの安全性・型安全性レビューが本体改善に未接続
- Priority: **high**
- Category: `frontend-security`
- Impact: AI回答表示や審査画面でXSS・型崩れ・表示不整合のリスクが残る。
- Recommended action: 危険HTML表示、APIレスポンス型、モバイル表示崩れを別トラックで優先修正する。
- Suggested program: `scripts/lease_system_gap_analyzer.py から frontend risk section を出力`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - docs/nextjs_review.md mentions dangerouslySetInnerHTML or any type risks
- Source refs:
  - `docs/nextjs_review.md`

### GAP-003: Sidecarエージェントの監査情報が古い
- Priority: **medium**
- Category: `agent-sidecar`
- Impact: 古い指摘を現在の真実として扱う危険がある。一方で再監査のチェックリストとしては有用。
- Recommended action: 古いレポートは自動で『再確認TODO』へ落とし、最新コードで再検証したものだけ有効扱いにする。
- Suggested program: `scripts/agent_sidecar_reader.py に freshness gate / recheck queue を追加`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - stale sidecar reports=15/15
- Source refs:
  - `reports/agent_sidecar_brief.json`

### GAP-005: RAG評価セットが小さく、検索品質の継続評価が弱い
- Priority: **medium**
- Category: `rag`
- Impact: 検索改善後に、どの質問で悪化したかを検知しにくい。
- Recommended action: 業種別・物件別・補助金・過去案件・否認理由など最低30〜50問へ拡張する。
- Suggested program: `scripts/evaluate_obsidian_rag.py を日次パイプラインに read-only で組み込む`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - rag_eval_set cases=10
- Source refs:
  - `api/knowledge/rag_eval_set.json`
  - `scripts/evaluate_obsidian_rag.py`

### GAP-007: SPECとテストはあるが、改善候補とのトレースが弱い
- Priority: **medium**
- Category: `quality`
- Impact: REV候補が実装された時、どの受入条件を満たしたか追跡しにくい。
- Recommended action: REV-ID / SPEC-ID / test file の対応表を自動生成し、未対応REVを見える化する。
- Suggested program: `scripts/lease_system_gap_analyzer.py に traceability matrix 出力を追加`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - test files=45
  - spec files=22
- Source refs:
  - `tests/`
  - `specs/`
  - `reports/latest.json`

### GAP-009: SQLite WAL/SHMなど実行時ファイルが作業ツリーに残っている
- Priority: **low**
- Category: `repo-hygiene`
- Impact: 誤コミットやバックアップ混乱の原因になる。
- Recommended action: .gitignore確認と、DBバックアップ/実行時ファイルの扱いを明文化する。
- Suggested program: `scripts/backup_case_data.py と repo hygiene check の連携`
- Guardrail: 本体スコア・DB・モデルを直接変更しない。まずレポート化して人間確認。
- Evidence:
  - data/lease_data.db-wal
  - data/lease_data.db-shm
- Source refs:
  - `data/`
