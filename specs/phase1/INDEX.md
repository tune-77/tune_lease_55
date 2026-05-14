# Phase 1 SPEC INDEX — lease_rule_checks

Phase 1 の目的: リースルールチェック機能（法定耐用年数・期待使用期間・保険）を独立モジュールとして追加し、APIレスポンスとモバイルUIに警告表示を組み込む。スコア計算には影響しない。

---

## SPEC一覧

| spec_id | ファイル | タイトル | ステータス | 依存 |
|---------|---------|---------|----------|------|
| P1-001 | [P1-001-lease-rule-checks-module.md](P1-001-lease-rule-checks-module.md) | lease_rule_checks.py モジュール | draft | なし |
| P1-002 | [P1-002-api-warnings-field.md](P1-002-api-warnings-field.md) | POST /predict warnings[] フィールド追加 | draft | P1-001 |
| P1-003 | [P1-003-ui-warning-banner.md](P1-003-ui-warning-banner.md) | index.html 警告バナーUI | draft | P1-001, P1-002 |

---

## 依存関係図

```
P1-001 (モジュール)
  └─ P1-002 (API統合)
       └─ P1-003 (UI表示)
```

## 実装順序

1. P1-001 承認 → Codex が `mobile_app/lease_rule_checks.py` を実装・テスト
2. P1-002 承認 → Codex が `mobile_app/api.py` を修正
3. P1-003 承認 → Codex が `mobile_app/index.html` を修正

**並行作業ルール**: P1-003 の HTML/JS 実装（フォーム追加・renderWarnings 関数）は P1-001 完了後に着手可能。ただし E2E 動作確認（API から warnings を実際に受け取るテスト）は P1-002 完了後まで待つこと。

---

## BR/AC 番号レンジ

| spec_id | BR レンジ | AC レンジ |
|---------|----------|----------|
| P1-001 | BR-101〜BR-109 | AC-101〜AC-115 |
| P1-002 | BR-111〜BR-115 | AC-201〜AC-208 |
| P1-003 | BR-121〜BR-126 | AC-301〜AC-310 |
