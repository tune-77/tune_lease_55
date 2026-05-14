# Phase 2 SPEC INDEX — AURION Q_risk

Phase 2 の目的: 財務矛盾検知モジュール `aurion/q_risk.py` を独立モジュールとして新規作成し、APIレスポンスとモバイルUIに参考値として組み込む。既存の RF/LGBM スコアリングロジックには一切触れない。

---

## SPEC一覧

| spec_id | ファイル | タイトル | ステータス | 依存 |
|---------|---------|---------|----------|------|
| P2-001 | [P2-001-aurion-q-risk-module.md](P2-001-aurion-q-risk-module.md) | aurion/q_risk.py — 財務矛盾検知モジュール | draft | なし |
| P2-002 | [P2-002-api-aurion-field.md](P2-002-api-aurion-field.md) | POST /predict — aurion.q_risk フィールド追加 | draft | P2-001 |
| P2-003 | [P2-003-ui-q-risk-display.md](P2-003-ui-q-risk-display.md) | index.html — Q_risk 財務矛盾スコア参考表示 | draft | P2-001, P2-002 |

---

## 依存関係図

```
P2-001 (aurion/q_risk.py モジュール)
  └─ P2-002 (API統合: aurion.q_risk フィールド追加)
       └─ P2-003 (UI表示: Q_risk 参考パネル)
```

## 実装順序

1. P2-001 承認 → Codex が `mobile_app/aurion/q_risk.py` を実装・テスト
2. P2-002 承認 → Codex が `mobile_app/api.py` を修正（aurion.q_risk フィールド追加）
3. P2-003 承認 → Codex が `mobile_app/index.html` を修正（Q_risk パネル追加）

**並行作業ルール**: P2-003 の HTML/JS 実装は P2-001 完了後に着手可能。ただし E2E 動作確認（API から aurion.q_risk を実際に受け取るテスト）は P2-002 完了後まで待つこと。

---

## BR/AC 番号レンジ

| spec_id | BR レンジ | AC レンジ |
|---------|----------|----------|
| P2-001 | BR-201〜BR-208 | AC-401〜AC-415 |
| P2-002 | BR-211〜BR-215 | AC-501〜AC-508 |
| P2-003 | BR-221〜BR-230 | AC-601〜AC-614 |

---

## Phase 2 の前提条件

- Phase 1（P1-001〜P1-003）が `implemented` であること
- `mobile_app/api.py` に `warnings[]` / `rule_check_status` フィールドが実装済みであること（P1-002 完了）

## Phase 2 完了条件

- P2-001〜P2-003 が全て `implemented`（PRマージ済み）
- `mobile_app/aurion/q_risk.py` が存在し、`detect_q_risk()` が呼び出し可能
- `POST /predict` レスポンスに `aurion.q_risk` フィールドが含まれる
- `mobile_app/index.html` に Q_risk パネルが表示される
- 既存スコア（RF/LGBM）の出力値が Phase 1 時点から変化しないこと

---

## 既存モジュールとの関係

| モジュール | Phase 2 での扱い |
|-----------|----------------|
| `quantum_analysis_module.py` | 変更・削除禁止。`aurion/q_risk.py` とは独立した別モジュール |
| `scoring_core.py` / `total_scorer.py` / `asset_scorer.py` | 変更禁止 |
| `mobile_app/lease_rule_checks.py` | 変更禁止 |
| `mobile_app/api.py` | P2-002 のみ変更可（aurion フィールド追加のみ） |
| `mobile_app/index.html` | P2-003 のみ変更可（Q_risk パネル追加のみ） |
