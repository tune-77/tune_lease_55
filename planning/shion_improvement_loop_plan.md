# 紫苑中心・改善ループ移行計画

作成日: 2026-07-17 / 改訂: 2026-07-17（v2）
前提: 紫苑を改善ロググループへ接続済み（対話室で日次改善レポートを相談材料として参照可能）。
本計画は「名実ともに紫苑を改善ループの中心に置く」ための段階移行を定義する。

出典: `planning/post_hackathon_shion_backlog.md` §9（完全自律DevOpsへ向けた段階移行）・§9.1（接続点と権限段階）を実装計画へ落としたもの。

v2改訂の背景: 初版直後、改善PMレポート整形（master直コミット `425fff8`）で
canonical_key ドリフト（削除済みメモの復活リスク）とCI未実行が実際に発生した。
この実例を踏まえ、同定の冗長化・シャドーモード・CI前提条件・コンテキスト予算の
4観点を計画へ組み込んだ。

---

## 1. 現状の実態（2026-07-17 時点）

### パイプライン側（自走している）

`scripts/run_daily_improvement_pipeline.sh` → `run_daily_improvement_core.sh` / `run_daily_improvement_post.sh` が夜間に:

1. 改善候補収集（`extract_obsidian_improvements.py` / `analyze_scoring_drift.py` / `analyze_wizard_inputs.py` / `analyze_rag_feedback.py` / `analyze_error_logs.py` 等）
2. `auto-improvement-pipeline` で整理 → `reports/latest.json`
3. 低リスク候補の自動承認（`auto_approve_safe_recipes.py`）→ 台帳ルール適用（`api/rule_engine/batch_apply.py`）
4. Codex 自動実行キュー生成・実行（`build_codex_auto_queue.py` / `execute_codex_queue.py`）
5. PR マージ後の台帳更新（`cleanup_improvement_reviews.py`、PR タイトルの REV 番号で照合）

### 紫苑側（見ているだけ）

- `api/main.py: _build_dialogue_improvement_report_context()` が `reports/latest.json` の**上位4件＋件数**をシステムプロンプトへ注入
- 対話室（`frontend/src/app/lease-intelligence/page.tsx`）に改善PMレポート表示と Codex 依頼文コピーあり
- 改善PMとしての振る舞い（今日やる／後回し／捨てる、システム監視優先）はプロンプト指示で定義済み（`lease_intelligence_dialogue.py` L482-486）

### ギャップ（「名」はPM、「実」はパイプライン直結）

| # | ギャップ | 影響 |
|---|---------|------|
| G1 | 観測範囲が狭い: 上位4件のみ。`data/pipeline_step_log.jsonl`（ステップ失敗）、Codexキュー実行状況、台帳（`scripts/improvement_ledger.jsonl`）、再帰的自己改善レポートが見えない | 紫苑が「昨夜何が起きたか」を正確に語れない |
| G2 | 紫苑のトリアージ判断がチャット本文のみで、構造化データとして残らない | 判断が翌日に引き継がれず、的中率も測れない |
| G3 | パイプラインが紫苑を素通り: `auto_approve_safe_recipes.py` と `build_codex_auto_queue.py` が紫苑の判断と無関係に優先順位を決める | 中心はパイプラインのままで「実」が伴わない |
| G4 | User承認が手動コピー: 承認→Codex依頼文→手動貼り付け。承認記録も残らない | 承認の履歴・監査ができない |
| G5 | 事後検証の環が閉じていない: `analyze_improvement_quality.py` の品質スコアが紫苑へ戻らず、「効いた／外した」の振り返りがない | 学習が蓄積されない |
| G6 | 権限境界がプロンプト文のみ: 課金上限・テスト失敗時停止・ロールバックのシステム的ガードが未実装 | 自律度を上げる前提が揃わない |

---

## 2. 段階計画

原則（backlog §9.1 踏襲）:
- 接続はまず read-only。権限は機能ではなくリスクで分ける
- 紫苑は実行権限を持っているように装わない
- 目的は User を外すことではなく、User が見るべき判断だけを残すこと

### Phase 0: 観測の拡張（read-only・低リスク・即着手可）

紫苑が改善ループの「今」を全部読める状態にする。
ただし**常時注入はしない**。`/api/chat` は既に改善レポート・記憶・人格指示で注入量が
多く、全会話に詳細を載せると応答品質（トーン・記憶想起）を圧迫するため、
コンテキスト予算を次の2層で管理する:

- **常時**: 異常サマリ1〜2行のみ（例: 「昨夜のパイプラインで2ステップ失敗」）
- **オンデマンド**: 改善相談の intent が立ったときだけ詳細を遅延ロード

- **P0-1 パイプラインヘルスの接続**: `data/pipeline_step_log.jsonl` 直近実行の失敗ステップ要約。常時は失敗件数のみ、詳細は改善相談時に展開
- **P0-2 台帳・キュー状況の接続**: `scripts/improvement_ledger.jsonl` 直近の applied/rejected と、Codex キュー（`reports/codex_auto_queue_*.json`）の実行結果サマリ（オンデマンド）
- **P0-3 再帰的自己改善レポート接続**: `reports/recursive_self_improvement_latest.md` の要点（オンデマンド）

変更対象: `api/main.py`（context builder と改善相談 intent 分岐のみ。既存の `intent` 3経路は壊さない）。DB/スコアリング/デプロイ設定に触れない。

完了条件: 対話室で「昨夜のパイプラインで失敗したステップは？」「今週マージされたREVは？」に紫苑が正答でき、かつ通常会話のプロンプト注入量が増えていない。

### Phase 0.5: ループ土台のCI保証（前提インフラ・Phase 2 までに必須）

検証で発覚した実例: master への直接コミット（Codex 経由の `425fff8` 等）は CI が
走らず、tsc もテストも未実行のまま本番コードに入る。Phase 2 以降はパイプライン
自身のコードを Codex が書き換えるループになるため、土台が未検証で動く状態を先に塞ぐ。

- **P0.5-1**: master への push でも python-syntax / frontend チェックを実行する（workflow の `on: push` 追加）、**または**ブランチ保護で master 直コミットを禁止し全変更をPR経由にする
- **P0.5-2**: どちらを採るかは Codex の運用形態（直コミットを許すか）と合わせて User が決める

完了条件: master に入るすべてのコミットが tsc + テストを通過している状態。

### Phase 1: トリアージの構造化記録（紫苑の判断を資産にする）

- **P1-1 トリアージ保存API**: `POST /api/improvement/triage`。`今日やる / 後回し / 捨てる` と理由1行を `data/shion_improvement_triage.jsonl`（追記形式、最後のエントリ有効 — 台帳と同じ規約）へ記録
  - **同定は冗長に持つ**: `canonical_key` 単独に頼らず、`source_event_id` とタイトルスナップショットも必ず記録し、キー照合失敗時のフォールバック同定を最初から設計に入れる。canonical_key は表示ロジックの変更でドリフトすることが実証済み（`425fff8` の実例。表示タイトルとキー算出は分離済みだが、再発防止は記録側でも担保する）
- **P1-2 分類主体の明確化と対話室UI**: 現在「今日やる/後回し/捨てる」を作っているのは LLM ではなく `page.tsx` の `classifyPmImprovementItems`（決定的ルール）。Phase 1 では**ルール分類をデフォルト、紫苑（LLM）は差分がある時だけ上書き提案**とし、`classified_by: rule | llm | user` を記録に残す。各候補にトリアージボタンを追加し、User が確定する
- **P1-3 翌日レポートへの反映**: 改善レポート文脈に「昨日のトリアージ結果と未処理分」を含め、判断の持ち越しをなくす
- **P1-4 User不在時のデフォルト動作**: 未確定のトリアージは**持ち越しのみ**。日数経過による自動昇格・自動破棄はしない（自動化はPhase 4のガード整備後に検討）

完了条件: トリアージ履歴が翌日の対話に「分類主体つきの判断＋User確定」として表示される。

### Phase 2: トリアージ→パイプライン接続（「実」を移す）

いきなり実キューを切り替えず、**シャドーモードを経由する**（backlog §4 の
Shion-HyDE RAG と同じ方針: 本番導線へ直結しない・shadow mode で比較する）。

- **P2-0 シャドーモード**: `build_codex_auto_queue.py` に `--shadow` を追加し、「従来の並び順」と「トリアージ反映後の並び順」を数日間並記出力する。実キューは従来のまま。乖離の内訳（順位変動・除外候補）を User がレビューし、妥当と確認できてから P2-1 へ進む
- **P2-1 キュー優先度の紫苑化**: `build_codex_auto_queue.py` が triage jsonl を読み、「今日やる（User確定済み）」を優先、「捨てる」を除外する
- **P2-2 自動承認の抑制入力**: `auto_approve_safe_recipes.py` に「捨てると確定した候補は自動承認しない」条件を追加
- **P2-3 承認記録**: User が対話室で承認した時刻・対象を triage レコードへ記録（`approved_at`）。Codex 依頼文生成はこの承認レコードがある候補に限定
- **P2-4 切り戻し手段**: 環境変数またはフラグ1つで従来の並び順へ即時復帰できるようにする（トリアージ品質が落ちた日の安全弁）

完了条件: シャドー期間で乖離が妥当と確認された後、夜間パイプラインの実行順序・除外が「紫苑の分類＋User承認」で決まり、素通りルートが消える。切り戻しフラグが動作する。

### Phase 3: 事後検証ループ（環を閉じる）

- **P3-1 結果の書き戻し**: `cleanup_improvement_reviews.py` の applied/rejected 結果を triage レコードの `outcome` へ反映
- **P3-2 品質フィードバック接続**: `analyze_improvement_quality.py` のスコアを紫苑の朝報告文脈へ注入し、「効いた／微妙／外した」を紫苑が報告・記録する（判断資産の `効いた/外した` 記録型 — backlog §3 と同型）
- **P3-3 的中率レポート**: 紫苑「今日やる」判定のマージ率・効果率を週次で集計し、Weekly Log と同じ形式で CLAUDE.md ではなくレポートへ出力

完了条件: 「紫苑のトリアージ的中率」が数字で追える。

### Phase 4: 低リスク自律実行（ハッカソン後・前提条件が揃ってから）

backlog §9 の「足りないもの」を先に実装しない限り着手しない:

- 課金・実行回数の上限（システム的ガード）
- テスト失敗時の自動停止条件とロールバック方針
- User承認なしで動かしてよい範囲の明文化（README・表示文言・レポート整理のみから開始）

完了条件は backlog §9 の段階7-8 に従う。本計画では**スコープ外**として予約のみ。

---

## 3. ループKPI（Phase 1 以降で計測開始）

| KPI | 定義 | 目的 |
|-----|------|------|
| トリアージ網羅率 | 改善候補のうち分類が確定した割合 | 「中心」の実態確認 |
| リードタイム | 候補発生→トリアージ→PRマージまでの日数 | ループ速度 |
| 的中率 | 「今日やる」→マージ→品質スコア良好の割合（`classified_by` 別に集計） | 紫苑の判断品質。ルール分類とLLM判断を分けて評価する |
| Overrule率 | 紫苑の分類を User が覆した割合 | 迎合・逆張りの検知（backlog §1 と接続） |
| 監視先行率 | パイプライン失敗を Slack 通知より先に紫苑が報告できた割合 | システム監視役の実効性 |

計測の実装: 週次集計スクリプト（例: `scripts/analyze_shion_pm_quality.py`）を
`run_daily_improvement_post.sh` へ**追記のみ・`|| true` 付き**で追加する
（CLAUDE.md 要注意領域: ステップ変更で朝報告を止めない）。
triage jsonl と台帳・品質スコアを突き合わせて `reports/` へ出力する。
監視先行率は triage/報告レコードのタイムスタンプと `notify_pipeline_alerts.py` の
通知時刻の比較で算出する（対話ログの解析はしない）。

---

## 4. 実装順とリスク

| 順 | 項目 | 変更対象 | リスク | 備考 |
|----|------|---------|--------|------|
| 1 | P0-1〜P0-3 | `api/main.py` context builder + 改善相談 intent | 低（read-only） | 1 PR にまとめ可。常時注入は異常サマリのみ |
| 2 | P0.5 | `.github/workflows/` またはブランチ保護 | 低 | Phase 2 までに必須。方式は User 判断 |
| 3 | P1-1〜P1-4 | `api/main.py` + `api/schemas.py` + 対話室 page.tsx | 低〜中 | `/api/chat` の既存 intent 3経路を壊さない（CLAUDE.md 要注意領域）。同定は event_id + タイトルスナップショットで冗長化 |
| 4 | P2-0〜P2-4 | `scripts/build_codex_auto_queue.py` / `auto_approve_safe_recipes.py` | 中 | シャドーモード→切替の順。切り戻しフラグ必須。パイプラインは追記のみ・`\|\| true` 維持（朝報告停止防止） |
| 5 | P3-1〜P3-3 | `scripts/cleanup_improvement_reviews.py` ほか | 中 | 台帳キーは `canonical_key(title)` 形式厳守 |
| 6 | Phase 4 | — | 高 | 前提ガード実装まで凍結 |

各項目は REV 化して通常の改善パイプラインに載せる（PR タイトルに REV 番号必須 — 台帳自動更新のため）。REV 番号は台帳の自動採番に従い、この文書では固定しない。
