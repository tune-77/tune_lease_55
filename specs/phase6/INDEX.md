# Phase 6 SPEC INDEX — 再帰的自己改善

Phase 6 の目的: 改善候補の抽出・検証・反映・観測・再抽出を閉ループとして扱い、改善結果が次の改善候補を生む再帰構造を定義する。  
改善レポート正本は `reports/latest.json`、再帰レポートは派生物として扱う。

---

## SPEC一覧

| spec_id | ファイル | タイトル | ステータス | 依存 |
|---------|---------|---------|----------|------|
| P6-001 | [P6-001-recursive-self-improvement.md](P6-001-recursive-self-improvement.md) | 再帰的自己改善ループ | implemented | なし |
| P6-002 | [P6-002-loop-engineering-map.md](P6-002-loop-engineering-map.md) | ループエンジニアリング最小観測 | implemented | P6-001 |

---

## 依存関係図

```text
P6-001 (再帰的自己改善ループ)
  └─ P6-002 (ループエンジニアリング最小観測)
```

## 実装順序

1. P6-001 承認 → Codex が `scripts/recursive_self_improvement.py` と関連パイプラインを実装・調整
2. P6-002 承認 → Codex が `docs/loop_engineering_map.md` と読み取り専用メトリクスを追加

**並行作業ルール**: Phase 6 は現時点で単独 SPEC のため並行実装は想定しない。

---

## BR/AC 番号レンジ

| spec_id | BR レンジ | AC レンジ |
|---------|----------|----------|
| P6-001 | BR-601〜BR-606 | AC-601〜AC-607 |
| P6-002 | BR-621〜BR-625 | AC-621〜AC-624 |

---

## Phase 6 の前提条件

- `reports/latest.json` が改善レポートの正本であること
- `~/Library/Logs/tunelease/ledger.jsonl` が台帳の正本であること
- `scripts/build_codex_auto_queue.py` / `scripts/build_wiki_promotion_queue.py` / `scripts/sync_memory_from_daily.py` が既存の正本ループとして維持されていること

## Phase 6 完了条件

- P6-001 が `implemented` になる
- 再帰的自己改善レポートが日次改善パイプラインから生成される
- `repeat_issue_rate` と `noise_rate` が観測可能である
- 再帰レポートが `reports/latest.json` を正本として上書きしない
- ループ地図と読み取り専用メトリクスで、正本・派生物・欠損状態が確認できる

---

## 既存モジュールとの関係

| モジュール | Phase 6 での扱い |
|-----------|----------------|
| `reports/latest.json` | 正本。上書きしない |
| `ledger.jsonl` | 正本。状態履歴の参照先 |
| `scripts/run_daily_improvement_core.sh` | 外側オーケストレーター |
| `scripts/recursive_self_improvement.py` | 再帰的自己改善の観測・派生物生成 |
| `scripts/loop_metrics.py` | ループ健全性の読み取り専用サマリ |
| `scripts/build_codex_auto_queue.py` | 低リスク実行キューの正本 |
| `scripts/build_wiki_promotion_queue.py` | wiki 昇格キューの正本 |
| `scripts/sync_memory_from_daily.py` | 記憶昇格ループの正本 |
