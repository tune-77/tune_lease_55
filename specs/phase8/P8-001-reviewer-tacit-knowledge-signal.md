---
spec_id: P8-001
phase: 8
title: 審査担当者の暗黙知シグナル構造化収集（スコアリング組込みは対象外）
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-08-01
updated: 2026-08-01
depends_on: [P4-001, P4-003]
superseded_by: ""
---

# P8-001 — 審査担当者の暗黙知シグナル構造化収集（スコアリング組込みは対象外）

---

## 1. Goal

「スコアリングは問題ないが、この規模は取れないことが多い」といった、審査担当者がスコアに現れない経験則で下す判断の**理由を構造化タグとして収集**し、`scoring_core.py` への新特徴量導入が妥当かどうかを検証するための土台を作る。本SPECでは収集・集計・統計的検証の仕組みのみを対象とし、**スコアリングロジック自体の変更は行わない**（判断は次フェーズのSPECに委ねる）。

---

## 2. Scope

### In scope
- `judgment_feedback` テーブルへの `reason_category` 列追加（既存の自由記述 `reason` 列は維持、後方互換）
- 事前定義タグ一覧（下記 4章）による選択式入力（自由記述と併用可）
- `learn_from_case_differences.py` の出力（特に `high_score_lost` シグナル）に `reason_category` 別集計を追加する `scripts/summarize_tacit_knowledge_signals.py`（新規）
- タグ別件数・タグと既存定量指標（財務スコア・信用情報・Q_risk等）との相関を出力するレポート（`reports/tacit_knowledge_signal_report_latest.md`）
- 特徴量候補への昇格可否を判定するためのゲート条件（5章 BR）の定義

### Out of scope
- `scoring_core.py` / `total_scorer.py` / `asset_scorer.py` / `quantum_analysis_module.py` / `aurion/` 以下への特徴量追加（**別SPECで扱う。P4-003 BR-424 の設計注記により、審査原入力値相当の特徴量を追加する場合は再学習パイプラインの設計自体の見直しが必要なため**）
- Streamlit / Next.js 審査画面でのタグ入力UI（本SPECはAPI/DB層のみ。UIは別SPEC）
- タグの自動抽出（NLP等）。本フェーズは審査担当者による選択式入力のみ
- 個々の審査担当者の評価・査定に使う目的での集計（あくまでモデル特徴量検証が目的）

---

## 3. Inputs / Outputs

### Inputs（`judgment_feedback` への追加フィールド）

| 項目名 | 型 | 必須/任意 | 説明 | 備考 |
|-------|-----|----------|------|------|
| `reason_category` | str \| None | 任意 | 事前定義タグ（4章参照） | NULL許容。既存レコードは全てNULL |
| `reason_category_confidence` | str | 任意（デフォルト `"reviewer_selected"`） | タグの付与方法 | `"reviewer_selected"` のみ。将来のNLP自動付与と区別するための予約フィールド |

### Outputs（`summarize_tacit_knowledge_signals.py`）

| 項目名 | 型 | 説明 |
|-------|-----|------|
| `tag_counts` | dict[str, int] | タグ別件数（`review_status="approved"` のみ集計） |
| `tag_correlation` | list[dict] | タグと既存定量指標（`total_score`, `asset_score`, `q_risk_score` 等）の相関係数・サンプル数 |
| `promotion_candidates` | list[dict] | 5章のゲート条件を満たしたタグ（特徴量候補として次SPECで検討可能） |
| `insufficient_data` | list[str] | サンプル数不足で判定不能なタグ |

---

## 4. Data Model

### タグ一覧（初期案・Cite the Source: `static_data/leasing_knowhow.json` の `qualitative_appeal` / `financial_improvement` カテゴリ構成、および `static_data/knowledge_base.json` の `scoring_system.quantitative_items` に対応しない定性理由を想定）

```python
from typing import Literal

ReasonCategory = Literal[
    "deal_size_capacity",       # 規模が自社与信枠・営業体制に対して過大
    "guarantor_or_collateral_weak",  # 保証・担保の実質的な弱さ
    "industry_concentration_risk",   # 業種・取引先集中リスク
    "repayment_source_unclear",      # 返済原資の説明が実質的に弱い
    "management_track_record",       # 経営者の実績・信頼性への懸念
    "competitor_pricing_pressure",   # 競合の提示条件による判断
    "other",                          # 上記に当てはまらない（自由記述reasonを参照）
]
```

タグは初期案であり、`status: review` への移行時に人間レビューで確定させる（Freshman Rules「Cite the Source」により、リース業務マニュアル・過去 `judgment_feedback.reason` の頻出語との突合を経てから確定すること。本SPECは仮説段階）。

```python
class TacitKnowledgeSignal(TypedDict):
    reason_category: str
    count: int
    approved_count: int
    correlation_with_existing_features: dict[str, float]  # 例: {"total_score": 0.12, "q_risk_score": 0.08}
    sample_size: int
    promotion_eligible: bool
    promotion_blocked_reason: str | None
```

---

## 5. Business Rules

**BR-801**: タグは既存レビューフローに乗せる
- 条件：`reason_category` が付与されたレコードでも、`review_status="approved"` になるまでは集計対象に含めない
- 処理：`load_explicit_differences()`（`learn_from_case_differences.py`）と同じ `eligible_for_training` 判定ロジックを踏襲する
- 根拠：既存の二段階レビュー設計（P4-001, bias_extraction）との整合性を保ち、未レビューの主観をそのまま学習経路に流さない

**BR-802**: 最小サンプル数ゲート
- 条件：あるタグの `approved_count` が 30件未満
- 処理：`promotion_eligible=False`, `promotion_blocked_reason="insufficient_sample"` とする
- 根拠：`retraining_pipeline.py` の `min_records=50`（BR-421）と同様の趣旨。少数タグでの特徴量化は過学習リスクが高い

**BR-803**: 既存特徴量との独立性チェック
- 条件：タグと既存定量指標（`total_score`, `asset_score`, `q_risk_score`, `competitor_pressure_score`）の相関係数の絶対値が 0.6 以上
- 処理：`promotion_eligible=False`, `promotion_blocked_reason="redundant_with_existing_feature"` とする
- 根拠：既存指標の代理変数に過ぎないタグを新特徴量として追加すると多重共線性・過学習を招く

**BR-804**: レビュアー偏在チェック
- 条件：あるタグの `approved` レコードのうち、単一の審査担当者（`judgment_feedback` に担当者識別情報がある場合）または単一の `source` からの比率が 70% を超える
- 処理：`promotion_eligible=False`, `promotion_blocked_reason="reviewer_bias_concentration"` とする
- 根拠：特定個人の主観がモデル全体の判断基準として固定化されることを防ぐ（観察コメントの「主観が過度に反映されないよう客観指標とのバランス」への直接対応）
- 備考：現行 `judgment_feedback` テーブルに担当者識別列がないため、本チェックの実装には別途 `reviewer_id`（匿名化ID可）列追加が必要。実装不能な場合は `promotion_eligible` を暫定 `False` 固定とし、レビュアー識別列追加を先行課題として明記する

**BR-805**: BR-802〜804を全て満たしたタグのみ `promotion_candidates` に含める
- 条件：`approved_count >= 30` かつ 相関係数 < 0.6 かつ レビュアー集中度 <= 70%
- 処理：`promotion_candidates` に追加し、次SPEC（`scoring_core.py` への組込み検討）の入力とする
- 根拠：3ゲート全通過を「検証済み」とみなす基準とする。これらを満たしても即座に特徴量化を承認するわけではなく、あくまで次SPECでの検討対象になることを意味する

---

## 6. UI / UX

本SPECでは対象外。タグ選択UIは審査結果差分の確認画面（`judgment_feedback` を書き込む既存フロー）に別SPECで追加する。

---

## 7. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| `reason_category` が4章の許容タグ外 | `"other"` として扱い、警告ログを出力 | 例外は発生させない |
| `judgment_feedback` に `reviewer_id` 相当の列がない | BR-804 のチェックをスキップし `promotion_eligible=False` 固定・理由を `"reviewer_id_unavailable"` とする | 先行課題として次SPECで対応 |
| 相関計算時にサンプル数不足（<5件） | 相関係数を `None` とし `insufficient_data` に含める | ZeroDivisionError等を発生させない |

---

## 8. Acceptance Criteria

**AC-801**: `reason_category` 列が既存レコードに影響しない
- Given: 既存の `judgment_feedback` レコード（`reason_category` 列追加前）
- When: マイグレーション（`ALTER TABLE ... ADD COLUMN`）を実行する
- Then: 既存レコードは全て `reason_category=NULL` のまま読み書きでき、`learn_from_case_differences.py` の既存出力（テスト `test_real_case_difference_learning.py`）が変化しない

**AC-802**: サンプル数不足タグは昇格しない
- Given: `deal_size_capacity` タグの `approved_count=20`
- When: `summarize_tacit_knowledge_signals.py` を実行する
- Then: `promotion_eligible=False`, `promotion_blocked_reason="insufficient_sample"`

**AC-803**: 既存指標と強相関のタグは昇格しない
- Given: あるタグと `total_score` の相関係数が 0.75
- When: 集計を実行する
- Then: `promotion_eligible=False`, `promotion_blocked_reason="redundant_with_existing_feature"`

**AC-804**: 全ゲート通過タグのみ候補に入る
- Given: `approved_count=45`、相関係数最大 0.3、レビュアー集中度 40%
- When: 集計を実行する
- Then: `promotion_candidates` に含まれる

**AC-805**: `scoring_core.py` 等の対象外ファイルが本SPEC実装で一切変更されない（回帰）

---

## 9. Non-Functional Requirements

- **後方互換性**: `judgment_feedback` スキーマ変更は追加列のみ（既存クエリ・既存テストを壊さない）
- **既存モジュール不干渉**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/` 以下を変更・importしない
- **監査可能性**: `promotion_blocked_reason` を必ず記録し、なぜ特徴量化を見送ったかを追跡可能にする

---

## 10. Implementation Notes（Codex向け・status: approved 後にのみ着手）

- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/q_risk.py`, `aurion/stealth_competitor.py`, `retraining_pipeline.py`
- **参照すべき既存実装**: `judgment_feedback.py`（テーブル定義）, `scripts/learn_from_case_differences.py`（レビュー済み差分の抽出ロジック）
- **新規ファイル配置**:
  ```
  scripts/summarize_tacit_knowledge_signals.py
  reports/tacit_knowledge_signal_report_latest.md   （生成物）
  tests/spec_phase8/test_P8-001.py
  ```
- **未解決の前提課題**: BR-804 のレビュアー集中度チェックは `judgment_feedback` に担当者識別列が存在しないと実装できない。`status: review` の段階で、担当者識別列追加（匿名化ID）を別途小さなSPECとして切るか、本SPECのスコープに含めるかを人間レビューで判断すること

---

## 11. Test Plan

### 単体テスト（Codexが作成）
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_801 | AC-801 | マイグレーション前後で既存出力が不変 |
| test_802 | AC-802 | サンプル数29件 → insufficient_sample |
| test_803 | AC-803 | 相関0.75 → redundant_with_existing_feature |
| test_804 | AC-804 | 全ゲート通過 → promotion_candidates に含まれる |
| test_805 | AC-805 | 対象外ファイルのdiffが空であることをCI等で確認 |

### 回帰テスト
- `tests/test_real_case_difference_learning.py` が全てpassすること
- `total_scorer.py` の出力が本SPEC実装前後で変化しないこと

### 手動確認（実装後）
- [ ] `judgment_feedback` への `reason_category` 書き込みが既存フローを壊さない
- [ ] `reports/tacit_knowledge_signal_report_latest.md` が生成され、タグ別集計・ゲート判定結果が確認できる
