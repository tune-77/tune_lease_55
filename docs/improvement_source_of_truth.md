# Improvement Source of Truth

このリポジトリの改善・昇格・記憶ループは、役割ごとに正本を分ける。

## 正本

- 改善レポート正本: `reports/latest.json`
- 台帳正本: `~/Library/Logs/tunelease/ledger.jsonl`
- 自動修正キュー正本: `scripts/build_codex_auto_queue.py`
- wiki 昇格キュー正本: `scripts/build_wiki_promotion_queue.py`
- 記憶昇格正本: `scripts/sync_memory_from_daily.py`
- 週次集計正本: `scripts/weekly_self_management.py`
- 再帰的自己改善派生物: `reports/recursive_self_improvement_*.json`
- 再帰的自己改善SPEC: `specs/phase6/P6-001-recursive-self-improvement.md`

## ルール

- `reports/latest.json` は改善パイプラインの唯一の配布先として扱う
- 再帰的自己改善は `reports/latest.json` を読み込むが、既定では書き換えない
- 再帰レポートは観測用の派生物とし、`latest.json` にメタデータを押し戻さない
- `ledger.jsonl` は状態履歴の正本であり、各 queue はそこから重複排除と抑制判定を行う

## 役割の切り分け

- `scripts/run_daily_improvement_core.sh`
  - 日次の統合オーケストレーター
  - 各 queue を順番に回す
- `scripts/recursive_self_improvement.py`
  - 改善レポートと prompt feedback を束ねる観測器
  - 再帰指標を生成するだけで、正本は更新しない
- `specs/phase6/P6-001-recursive-self-improvement.md`
  - 再帰的自己改善の振る舞い・状態遷移・判定ルールの正規定義
  - 実装はこのSPECに合わせる
- `scripts/build_codex_auto_queue.py`
  - needs_review から低リスク候補を切り出す
- `scripts/build_wiki_promotion_queue.py`
  - Obsidian の昇格候補を切り出す
- `scripts/sync_memory_from_daily.py`
  - 日次メモから `MEMORY.md` へ昇格する

## 迷ったら

- 「保存先が正本か、派生か」で判断する
- 正本を増やさず、派生物を増やす
- 変更したいときは先に `ledger.jsonl` と `reports/latest.json` のどちらが正本かを確認する
