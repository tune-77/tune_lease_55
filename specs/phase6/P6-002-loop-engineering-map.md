---
spec_id: P6-002
phase: 6
title: ループエンジニアリング最小観測
status: implemented
author: Codex
reviewer: ""
version: "1.0"
created: 2026-06-19
updated: 2026-06-19
depends_on:
  - P6-001
superseded_by: ""
---

# P6-002 — ループエンジニアリング最小観測

## 1. Goal

既存の改善・再帰・AI応答改善ループを壊さず、正本と派生物の関係を明文化し、読み取り専用の健全性サマリを生成する。

## 2. Scope

### In scope

- ループ地図を `docs/loop_engineering_map.md` に保存する
- `reports/latest.json`、`reports/recursive_self_improvement_latest.json`、`data/prompt_feedback_log.jsonl` を読み取る
- ループ健全性の JSON / Markdown を生成できる CLI を追加する
- 入力ファイルが欠けていても失敗せず、欠損として報告する

### Out of scope

- 日次パイプラインへの自動接続
- 改善ログ画面への表示
- Obsidian信号抽出の拡張
- 自動反映範囲の拡大
- `reports/latest.json` の更新

## 3. Inputs / Outputs

### Inputs

| 項目 | 既定パス | 必須 |
| --- | --- | --- |
| improvement_report | `reports/latest.json` | 任意 |
| recursive_report | `reports/recursive_self_improvement_latest.json` | 任意 |
| prompt_feedback_log | `data/prompt_feedback_log.jsonl` | 任意 |

### Outputs

| 項目 | 既定パス |
| --- | --- |
| loop_metrics_json | `reports/loop_engineering_latest.json` |
| loop_metrics_md | `reports/loop_engineering_latest.md` |

## 4. Business Rules

**BR-621**: 読み取り専用  
入力ファイルを変更しない。

**BR-622**: 欠損許容  
入力ファイルが存在しない場合でもエラー終了せず、`available: false` として報告する。

**BR-623**: 正本保護  
`reports/latest.json` は更新しない。

**BR-624**: 最小健全性判定  
`status` は `ok`、`warn`、`attention` の3段階に限定する。

**BR-625**: 推奨は観測に限定  
出力の recommendations は次の確認行動までに留め、自動修正を指示しない。

## 5. Acceptance Criteria

**AC-621**: ループ地図  
`docs/loop_engineering_map.md` に主要ループ、正本、派生物、観測点が記載されている。

**AC-622**: 読み取り専用集計  
CLI実行で `reports/latest.json` と prompt feedback log からサマリを生成できる。

**AC-623**: recursive report 欠損  
`reports/recursive_self_improvement_latest.json` が存在しなくても、`status` と recommendations を返す。

**AC-624**: テスト  
欠損入力と通常入力の両方を対象にしたテストがある。

