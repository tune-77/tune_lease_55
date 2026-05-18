# Phase 4 SPEC INDEX — フィードバックループ & データ収集 & モデル再学習

Phase 4 の目的: 審査結果を蓄積し（P4-001）、EDINET API で財務データ取得を自動化し（P4-002）、蓄積データを使ってモデルを継続的に改善する再学習パイプライン（P4-003）を構築する。既存スコアリング（RF/LGBM）・Aurion モジュール・量子リスクモジュールには一切触れない。

---

## SPEC一覧

| spec_id | ファイル | タイトル | ステータス | 依存 |
|---------|---------|---------|----------|------|
| P4-001 | [P4-001-screening-records-feedback.md](P4-001-screening-records-feedback.md) | 支払い実績テーブル連携 — screening_records フィードバックループ | draft | P3-001, P3-002, P3-003 |
| P4-002 | [P4-002-edinet-data-collection.md](P4-002-edinet-data-collection.md) | EDINET データ自動収集 — 法人財務データ取得モジュール | draft | P4-001 |
| P4-003 | [P4-003-model-retraining-pipeline.md](P4-003-model-retraining-pipeline.md) | モデル再学習パイプライン — 自動・手動再学習 | draft | P4-001, P4-002 |

---

## 依存関係図

```
P3-001〜P3-003 (Phase 3: 完了済み)
  └─ P4-001 (screening_records テーブル & フィードバックループ)
       ├─ P4-002 (EDINET データ自動収集 → キャッシュ DB 共有)
       └─ P4-003 (モデル再学習パイプライン)
            └─ P4-002 (DB 共有による初期化順序依存：edinet_cache テーブルが同一 DB に存在するため P4-001 完了後に P4-002 を先行実行すること)
```

## 実装順序

1. **P4-001** 承認 → Codex が `screening_recorder.py` を実装・テスト。Streamlit / Slack Bot から呼び出し連携を追加
2. **P4-002** 承認 → Codex が `edinet_collector.py` を実装・テスト。Streamlit 審査フォームに「EDINET取得」ボタンを追加
3. **P4-003** 承認 → Codex が `retraining_pipeline.py` を実装・テスト。Streamlit 管理画面に再学習 UI を追加

**並行作業ルール**: P4-002 は P4-001 の `edinet_cache` テーブルを同一 DB に作成するため、DB 初期化ロジックを共有する。P4-001 完了後に P4-002 を開始すること。P4-003 は P4-001 の `screening_records` データが必要なため P4-001 完了後に開始すること。P4-002 と P4-003 は P4-001 完了後であれば並行作業可能。

---

## BR/AC 番号レンジ

| spec_id | BR レンジ | AC レンジ |
|---------|----------|----------|
| P4-001 | BR-401〜BR-406 | AC-1001〜AC-1010 |
| P4-002 | BR-411〜BR-416 | AC-1101〜AC-1110 |
| P4-003 | BR-421〜BR-426 | AC-1201〜AC-1212 |

---

## Phase 4 の前提条件

- Phase 3（P3-001〜P3-003）が `implemented` であること
- `mobile_app/aurion/stealth_competitor.py` が存在し `detect_stealth_competitor()` が呼び出し可能であること
- `POST /predict` レスポンスに `aurion.competitor_pressure` フィールドが含まれること
- `data/lease_data.db` が存在するか、アプリ起動時に自動作成される仕組みが整っていること

## Phase 4 完了条件

- P4-001〜P4-003 が全て `implemented`（PRマージ済み）
- `data/lease_data.db` に `screening_records`・`edinet_cache`・`retraining_log` の3テーブルが存在する
- Streamlit 審査フォームに「EDINET取得」ボタンが存在し、財務データが自動入力される
- Streamlit 管理画面に「モデル再学習を実行」ボタンと履歴テーブルが存在する
- 審査実行後に `screening_records` へのレコード INSERT が確認できる
- 既存スコア（RF/LGBM / Aurion / 量子リスク）の出力値が Phase 3 時点から変化しないこと

---

## 既存モジュールとの関係

| モジュール | Phase 4 での扱い |
|-----------|----------------|
| `quantum_analysis_module.py` | 変更・削除禁止 |
| `aurion/q_risk.py` | 変更禁止 |
| `aurion/stealth_competitor.py` | 変更禁止 |
| `scoring_core.py` / `total_scorer.py` / `asset_scorer.py` | 変更禁止 |
| `mobile_app/api.py` | 変更禁止（Phase 3 完了時点の状態を維持） |
| `mobile_app/index.html` | 変更禁止 |
| `train_quantum.py` | 変更禁止 |
| `data/lease_data.db` | P4-001〜P4-003 で新テーブルを追加（既存テーブルは変更しない） |
| `tune_lease_55.py` / `components/` | P4-001〜P4-003 の呼び出し連携のみ追加。既存ロジックは変更しない |
