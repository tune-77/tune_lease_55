# Phase 3 SPEC INDEX — AURION ステルス競合推定

Phase 3 の目的: `winning_spread` 乖離を競合圧力シグナルとして活用する独立モジュール `aurion/stealth_competitor.py` を新規作成し、APIレスポンスとモバイルUIに参考値として組み込む。既存の RF/LGBM スコアリングロジックには一切触れない。

---

## SPEC一覧

| spec_id | ファイル | タイトル | ステータス | 依存 |
|---------|---------|---------|----------|------|
| P3-001 | [P3-001-aurion-stealth-competitor-module.md](P3-001-aurion-stealth-competitor-module.md) | aurion/stealth_competitor.py — ステルス競合推定モジュール | draft | P2-001, P2-002 |
| P3-002 | [P3-002-api-competitor-pressure-field.md](P3-002-api-competitor-pressure-field.md) | POST /predict — aurion.competitor_pressure フィールド追加 | draft | P3-001 |
| P3-003 | [P3-003-ui-competitor-pressure-display.md](P3-003-ui-competitor-pressure-display.md) | index.html — ステルス競合圧力スコア参考表示 | draft | P3-001, P3-002 |

---

## 依存関係図

```
P3-001 (aurion/stealth_competitor.py モジュール)
  └─ P3-002 (API統合: aurion.competitor_pressure フィールド追加)
       └─ P3-003 (UI表示: ステルス競合圧力パネル)
```

## 実装順序

1. P3-001 承認 → Codex が `mobile_app/aurion/stealth_competitor.py` を実装・テスト
2. P3-002 承認 → Codex が `mobile_app/api.py` を修正（aurion.competitor_pressure フィールド追加）
3. P3-003 承認 → Codex が `mobile_app/index.html` を修正（ステルス競合圧力パネル追加）

**並行作業ルール**: P3-003 の HTML/JS 実装は P3-001 完了後に着手可能。ただし E2E 動作確認（API から aurion.competitor_pressure を実際に受け取るテスト）は P3-002 完了後まで待つこと。

---

## BR/AC 番号レンジ

| spec_id | BR レンジ | AC レンジ |
|---------|----------|----------|
| P3-001 | BR-301〜BR-304 | AC-701〜AC-717 |
| P3-002 | BR-311〜BR-315 | AC-801〜AC-808 |
| P3-003 | BR-321〜BR-326 | AC-901〜AC-910 |

---

## Phase 3 の前提条件

- Phase 2（P2-001〜P2-003）が `implemented` であること
- `mobile_app/aurion/__init__.py` および `aurion/q_risk.py` が実装済みであること（P2-001 完了）
- `POST /predict` レスポンスに `aurion.q_risk` フィールドが含まれること（P2-002 完了）

## Phase 3 完了条件

- P3-001〜P3-003 が全て `implemented`（PRマージ済み）
- `mobile_app/aurion/stealth_competitor.py` が存在し、`detect_stealth_competitor()` が呼び出し可能
- `POST /predict` レスポンスに `aurion.competitor_pressure` フィールドが含まれる
- `mobile_app/index.html` にステルス競合圧力パネルが表示される
- 既存スコア（RF/LGBM）の出力値が Phase 2 時点から変化しないこと

---

## 既存モジュールとの関係

| モジュール | Phase 3 での扱い |
|-----------|----------------|
| `quantum_analysis_module.py` | 変更・削除禁止。`aurion/stealth_competitor.py` とは独立した別モジュール |
| `aurion/q_risk.py` | 変更禁止（Phase 2 で作成済み） |
| `scoring_core.py` / `total_scorer.py` / `asset_scorer.py` | 変更禁止 |
| `mobile_app/lease_rule_checks.py` | 変更禁止 |
| `mobile_app/api.py` | P3-002 のみ変更可（aurion.competitor_pressure フィールド追加のみ） |
| `mobile_app/index.html` | P3-003 のみ変更可（ステルス競合圧力パネル追加のみ） |
