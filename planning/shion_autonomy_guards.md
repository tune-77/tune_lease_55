# 紫苑改善ループ — 自律実行ガードと切り戻し手順

作成日: 2026-07-18
位置づけ: `planning/shion_improvement_loop_plan.md` Phase 4（低リスク自律実行）の前提条件。
backlog（`planning/post_hackathon_shion_backlog.md` §9）の「足りないもの」のうち、
課金・実行回数の制御 / テスト失敗時の停止条件 / ロールバック方針 を実装・明文化する。

## 1. キルスイッチ（即時全停止）

| スイッチ | 効果 | 設定場所 |
|----------|------|---------|
| `CODEX_QUEUE_DISABLED=1` | Codex自動実行キューを一切実行しない（`execute_codex_queue.py` が即終了） | launchd plist の環境変数 |
| `SHION_TRIAGE_QUEUE_MODE=off` | トリアージをキュー・自動承認から完全に切り離す（従来動作へ復帰） | 同上 |

どちらも**変数1つで即時・コード変更不要・再デプロイ不要**。

## 2. 実行回数・課金の制御

- `CODEX_QUEUE_DAILY_LIMIT`（既定 3）: 1日に実行するキューアイテムの上限。
  同日の `reports/codex_queue_result_YYYYMMDD*.json` から実行済み件数を数え、
  超過分は実行せず持ち越す（`guards.carried_over` に記録）。
- 上限を超える再実行が必要な日は、User が明示的に環境変数を上げて1回だけ実行する。

## 3. 失敗時の自動停止

- `CODEX_QUEUE_MAX_CONSECUTIVE_FAILURES`（既定 2）: 連続失敗がこの回数に達したら
  残りのアイテムを実行せず停止する（環境要因の可能性が高いため翌日再試行）。
  停止した事実は結果JSONの `guards.aborted_by_consecutive_failures` に残る。
- 各アイテムの検証は従来どおり `required_checks`（py_compile / targeted_test）を必須とし、
  PR の自動マージは行わない（2026-07 `848813f` で廃止済み・人間承認必須）。

## 4. ロールバック方針

問題の種類ごとに、戻し方を1手に固定する:

| 症状 | 戻し方 |
|------|--------|
| トリアージ反映後のキューがおかしい | `SHION_TRIAGE_QUEUE_MODE=shadow`（比較は続く）または `off`（完全切離し） |
| Codex自動実行が暴れる/課金が不安 | `CODEX_QUEUE_DISABLED=1` |
| 自動実装のPRが壊れていた | PR を close（マージ前）/ `git revert`（マージ後）。台帳は `cleanup_improvement_reviews.py` が rejected を記録 |
| トリアージ記録自体が汚染された | `data/shion_improvement_triage.jsonl` は追記形式なので、末尾に打ち消しの判断を User 確定で追記（最後のエントリが有効）。ファイル削除はしない（監査履歴を残す） |

## 5. User承認なしで動いてよい範囲（現時点）

- 読む・集計する・レポートを書く・提案を記録する（classified_by=llm の提案追記を含む）
- キューの**並び替え・除外**は User 確定トリアージがある場合のみ（live モード時）
- **実装の実行**は従来どおり codex-safe 判定＋required_checks＋人間のPRレビューが前提
- DB / スコアリング / 認証 / デプロイ設定 / 顧客データ / 課金を伴う操作は必ず人間承認（backlog §9.1 権限段階5）
