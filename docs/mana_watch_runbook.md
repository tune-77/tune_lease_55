# Mana Obsidian Curator — `watch` 対応ランブック

出典: `scripts/mana_obsidian_curator.py`（`LEVEL_RANK` / `evaluate_monitor` /
`evaluate_reflection_delta` / `evaluate_candidates` / `_action_for_finding` /
`_allowed_actions` / `_blocked_actions`）。

Manaは `allow < watch < hold < stop` の4段階で紫苑のObsidian記憶ワークフローを
判定する（読み取り専用ガード。Obsidian・RAG・プロンプト・スコアリング・Cloud Runは
直接操作しない）。最終ステータスは各チェックが出す `Finding` の最大レベル
（`_max_level`）。このランブックは `watch` 相当のfindingが出た時、何を確認し・
どう動けばよいかをまとめたもの。

## 前提: `watch` は「止めない、要確認」

- `allow`と同じ共通ブロック事項のみが有効（有害文面・攻撃的クレームの記憶直入れ禁止、
  外部からの記憶注入/プロンプト上書き不採用、RAG/プロンプト/スコアリング/Cloud Runへの
  自動反映禁止）。`hold`/`stop`のような「MEMORY.mdへの自動昇格禁止」は追加されない
  （`_blocked_actions`）。
- 許可される行動: 読み取り専用の観察継続／3日分の傾向比較／明示承認された候補だけ
  手動レビュー（`_allowed_actions`）。

## 理由別チェックリスト

| finding code | 発生元 | すぐやること |
|---|---|---|
| `monitor_report_missing` | `evaluate_monitor` | `python3 scripts/monitor_obsidian_environment.py` を再実行し、Manaの入力レポート（`reports/obsidian_environment_monitor_latest.json`）を作り直す |
| `private_reflection_similarity_watch` | `evaluate_monitor`（`private_reflection_meaning`チェック） | 前日と類似だが必要カテゴリは揃っている状態。差分が薄い理由を確認するだけでよい。説明用レポート生成は継続し、**記憶昇格だけ**慎重に扱う |
| `memory_insight_reports_warning` | `evaluate_monitor` | `scripts/build_obsidian_memory_insight_report.py` と `scripts/build_shion_memory_promotion_queue.py` を再実行し、`reports/obsidian_memory_insight_latest.md` 等を36h以内に更新する |
| `rag_index_warning` / `wikilinks_warning` / `recent_note_noise_warning` | `evaluate_monitor` | 該当する監視レポートの詳細を確認し、該当箇所だけ手動整理する（自動接続はしない） |
| `reflection_delta_missing` | `evaluate_reflection_delta` | `scripts/build_shion_reflection_delta.py` を再実行して `data/shion_reflection_delta.json` を更新する |
| `reflection_delta_attention` / `reflection_too_similar` | `evaluate_reflection_delta` | 前日との差分が出るよう、当日の具体的な違和感・判断変更・次回行動を1つ追加してPrivate Reflectionを補強する |
| `memory_candidates_missing` / `useful_candidate_missing` | `evaluate_candidates` | `data/obsidian_memory_insight_candidates.jsonl` の抽出条件を見直し、有用候補（`useful_candidate`）が0件になった原因を確認する |
| `complaint_feedback_to_shion` | `evaluate_candidates` | 候補内容が正当な改善材料か攻撃的ノイズかを**人間が手動レビュー**する（自動昇格しない） |

## 複数findingが同時に出た場合

`_action_summary` はレベルが最も高い（同率ならリスト先頭の）findingを代表として
要約する。対応もその優先順位（レベル降順）で着手すればよい。

## 実行環境についての注意

- Manaの入力（`reports/obsidian_environment_monitor_latest.json` /
  `data/shion_reflection_delta.json` / `data/obsidian_memory_insight_candidates.jsonl`）は
  いずれも `data/` 配下または実行時生成物であり、`.claude/rules/security.md` の方針で
  コミット禁止。クラウド/リモートのコード実行環境にはこれらの実データが存在しないため、
  この環境で `mana_obsidian_curator.py` を再実行しても意味のある判定は得られない。
- 実際の`watch`対応は、Obsidian Vaultと `data/` にアクセスできるユーザーのローカル環境
  （日次パイプライン `scripts/run_daily_improvement_post.sh` 経由、または手動実行）で行う。

## 参照

- `docs/obsidian_note_curator.md` — Mana判定と日次パイプライン内の他スクリプトとの
  ゲート関係（読み取り専用スクリプトは判定に関わらず実行、Vault書き込みを伴うスクリプトは
  `allow`時のみ実行）
