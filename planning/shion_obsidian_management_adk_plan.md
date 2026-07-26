# 紫苑 Obsidian Vault 管理 ADK 計画

作成日: 2026-07-25

## 目的

紫苑に Obsidian Vault の中身を管理させる。ただし、Vaultを勝手に書き換える自律エージェントにはしない。

目標は、Vault内の知識・作業ログ・判断資産・プレゼン材料を、紫苑が読み、棚卸しし、整理案を出し、人間承認後に安全に反映すること。

## 結論

ADK は使える。既にこのリポジトリには `api/shion_agent.py` と `api/shion_debate_adk.py` があり、`google.adk` 前提のエージェント構造がある。

ただし、Vault管理ではADKを直接の書き込み主体にしない。

ADKの役割:

- 複数の管理役を束ねる
- 読み取り結果を統合する
- 整理案・分割案・昇格案・リンク案を作る
- 変更計画をJSON/Markdownで出す

実際の書き込み:

- 差分プレビュー
- 人間承認
- バックアップ
- 書き込み
- 検証

の順に限定する。

## 基本思想

Vaultは紫苑の記憶そのものではなく、紫苑が参照する知識宇宙。

生ログ、作業ログ、会話ログ、判断資産候補、正準判断資産、プレゼン材料を混ぜない。

この計画は、以前検討した「知識をより活用するための保存方法」に準じる。

既存方針:

- `docs/okf_subset.md`: RAG、Agent、人間レビューで共有する知識だけを、Markdown + YAML frontmatter の小さなサブセットで保存する。
- `planning/shion_memory_task_agent_plan.md`: AIログは `agent_review_context` という補助知識であり、Userが採用・修正・却下した部分だけ判断資産候補へ進む。
- `planning/memory_effectiveness_layer_plan.md`: 記憶は `保存された -> 想起された -> 使われた -> 人間が採用した -> 結果照合された` の状態で育てる。
- `planning/shion_system_pitch_onepage.md`: 判断資産は単なるメモではなく、`条件 -> 見る論点 -> 稟議に残す一文 -> 使わない条件` の半構造化された判断構文として扱う。

したがって、Vault管理ADKの目的は「保存場所を大きく変えること」ではない。  
既存ノートを壊さず、再利用される知識だけを小さく構造化し、検索・想起・使用・人間評価・結果照合の証跡を付けること。

重要なのは以下の分離:

- raw_log: 生ログ。原則そのまま保持。直接RAGの主材料にしない。
- distilled_note: 要約・決定事項・捨てた案・次の行動。
- agent_review_context: Claude/Codex/他AIの作業・レビュー要約。方針理解や設計相談の補助知識。これ自体は判断資産ではない。
- judgment_asset_candidate: 案件判断・確認条件・稟議文に使える候補。
- canonical_judgment_asset: 人間レビュー済みの判断資産。
- outcome_validated_asset: 実案件結果と照合済みの判断資産。
- public_summary: プレゼンに出せる安全な要約。

## 保存形式の原則

### 1. Rawは動かさない

Claude/Codex作業ログ、AI Chat、Private Reflection、Daily note は原則として raw のまま保持する。

禁止:

- rawログ全文を判断資産扱いする
- rawログをRAG主材料として過信する
- rawログを勝手に書き換える
- private reflectionをpublic summaryへ直接使う

### 2. 再利用する知識だけ OKF 風にする

再利用する価値が確認された知識だけ、`docs/okf_subset.md` に準じて frontmatter を持つMarkdownへ蒸留する。

最低限:

```yaml
type: lease_rule
```

推奨:

```yaml
title: 補助金前提案件の未採択時確認
domain: credit
tags: [補助金, 設備投資, 返済原資, 条件付き承認]
source: user_judgment
confidence: medium
status: draft
updated: 2026-07-25
related:
  - 補助金
  - 条件付き承認
```

本文:

```md
## 要点

## 判断ルール

## 根拠

## 使わない条件

## Related
```

### 3. 判断資産は判断構文で保存する

判断資産候補は、通常メモではなく次の形に寄せる。

```text
条件 -> 見る論点 -> 追加確認 -> 稟議に残す一文 -> 使わない条件
```

例:

```text
条件: 補助金採択前提で設備を導入する
見る論点: 採択前の返済原資と未採択時の代替資金
追加確認: 資金繰り表、つなぎ融資証跡、増産後粗利改善資料
稟議に残す一文: 補助金未採択時も営業CF・自己資金・銀行借入でリース料支払可能な計画確認を条件とする。
使わない条件: 補助金を返済原資に含めず、既存CFだけで支払可能な場合
```

### 4. 記憶状態を持たせる

ノートが存在するだけでは「効いた」と扱わない。

状態:

- dormant: 保存されただけ
- recalled: `knowledge_refs` / `memory_recall.refs` に出た
- used: 回答・確認事項・稟議文面に反映された
- validated: Userが採用、または結果照合で有効性が確認された
- noisy: 想起されるが判断に効かない

ADKは、この状態遷移を管理するための棚卸し役でもある。

### 5. Search-first Wiki に準じる

Vault管理ADKは、保存場所の整理より先に検索性を上げる。

優先するもの:

- `Projects/tune_lease_55/tune_lease_55 Wiki.md`
- `Projects/tune_lease_55/検索語インデックス.md`
- 各主要ノートの `## Related`
- 別名、英語名、略称、API名、ファイル名

禁止:

- 全ノートを一斉変換する
- リンクを大量に貼るだけのグラフ化
- キーワード一致だけでRelatedを増やす

## エージェント構成

### 1. Vault Librarian

役割:

- Vault構造の棚卸し
- 重複ノート検出
- 古いノート・孤立ノート・リンク切れ候補の検出
- フォルダ配置案の作成

権限:

- 読み取りのみ
- 書き込み案を出すだけ

出力:

- `reports/obsidian_librarian_review_latest.md`
- `reports/obsidian_librarian_review_latest.json`

### 2. Judgment Curator

役割:

- Claude/Codex作業ログ、チャットログ、審査メモから判断資産候補を抽出
- 「現場判断」「技術判断」「プレゼン表現」「安全境界」を分類
- 判断資産候補を active store へ入れる前のレビューリストを作る

権限:

- 読み取りのみ
- active store へは直接昇格しない

出力:

- `reports/judgment_asset_candidates_from_vault_latest.md`
- `data/judgment_asset_candidate_queue.jsonl`

### 3. Work Log Distiller

役割:

- Claude/Codex作業ログを短い意思決定ログへ蒸留
- 実装理由、捨てた案、Userが重視した論点、検証結果を抽出
- 紫苑の返答に深みを出す材料へ変換

権限:

- 読み取りのみ
- 蒸留結果は review-only

出力:

- `reports/work_log_distillation_latest.md`
- `reports/work_log_distillation_latest.json`

### 4. Link Gardener

役割:

- 関連ノート候補を提案
- Wikiリンク案を出す
- 検索語インデックスの不足を検出
- 判断資産、作業ログ、プレゼン材料のつながりを見える化

権限:

- 読み取りのみ
- リンク追加は承認後

出力:

- `reports/obsidian_link_suggestions_latest.md`

### 5. Mana Gate

役割:

- 個人情報、攻撃的表現、記憶注入、公開不適切表現を検出
- 長期記憶・判断資産・public summaryへの昇格を止める
- 原文を使わず、カテゴリと理由だけでレビューへ回す

権限:

- 読み取り・ブロック判定
- 自動削除はしない

出力:

- `reports/mana_vault_guard_latest.md`

### 6. Patch Executor

役割:

- 承認済み変更だけをVaultへ反映
- 実行前バックアップ
- Markdown構文検証
- 変更後レポート作成

権限:

- 承認済みコマンドのみ書き込み

出力:

- `reports/obsidian_patch_execution_latest.md`

## ADKでの流れ

```text
User / Scheduler
  ↓
Shion Vault Manager ADK
  ↓
Parallel:
  - Vault Librarian
  - Judgment Curator
  - Work Log Distiller
  - Link Gardener
  - Mana Gate
  ↓
Arbiter
  ↓
Review Plan
  ↓
Human Approval
  ↓
Patch Executor
  ↓
Verify / Report
```

ADKは `ParallelAgent` で複数レビューを並行実行し、`SequentialAgent` で統合・承認待ち・実行へ進める形が合う。

## 実装フェーズ

### Phase 0: Read-only棚卸し

目的:

Vaultを書き換えず、管理対象を見える化する。

実装:

- `scripts/obsidian_vault_management_review.py`
- Vault内の対象ディレクトリを限定して読む
- 生ログ、作業ログ、Private Reflection、判断資産候補、公開材料を分類
- OKF候補、判断構文候補、agent_review_context、検索語不足を分ける

成果物:

- `reports/obsidian_vault_management_review_latest.md`
- `reports/obsidian_vault_management_review_latest.json`

完了条件:

- どのノートが raw / distilled / candidate / canonical / public か見える
- どのノートが OKF化候補か見える
- どの材料が agent_review_context で、どれが User判断候補か分かる
- 重複・孤立・リンク不足が出る
- 書き込みはゼロ

### Phase 1: 作業ログ蒸留

目的:

Claude/Codex作業ログを紫苑の深みへ変換する。

実装:

- `scripts/distill_agent_work_logs.py`
- Claude/Codexログから以下を抽出:
  - 決定事項
  - 捨てた案
  - Userが重視した論点
  - 実装理由
  - 次回から変えること

成果物:

- `reports/work_log_distillation_latest.md`
- `data/work_log_distilled_facts.jsonl`

完了条件:

- 生ログ本文をRAGへ直投入しない
- 1作業ログあたり最大5項目に蒸留
- `source_type=agent_work_log`
- `memory_lane=agent_review_context`
- `judgment_asset_status=not_asset_until_user_selected`
- public summaryへ出せるものと出せないものを分ける

### Phase 2: OKF風ノート候補化

目的:

再利用される知識だけを、`docs/okf_subset.md` に準じた構造化Markdown候補へ変換する。

実装:

- `scripts/build_okf_note_candidates.py`
- rawログから直接作らず、蒸留済み要点・User判断・既存knowledge noteを入力にする
- 10〜30件だけ候補化し、一括変換はしない

成果物:

- `reports/okf_note_candidates_latest.md`
- `reports/okf_note_candidates_latest.json`

完了条件:

- `type`, `domain`, `tags`, `source`, `confidence`, `status`, `updated`, `related` が入る
- 本文に `要点 / 判断ルール / 根拠 / 使わない条件 / Related` がある
- `python scripts/validate_okf_subset.py <candidate-dir>` が通る想定で設計されている
- RAG再indexはまだ行わない

### Phase 3: 判断資産候補化

目的:

Vault内のノートから、案件判断に使える候補だけを抽出する。

実装:

- 既存の `scripts/build_judgment_materials_preview.py`
- 既存の `scripts/build_canonical_judgment_rules.py`
- 新規で work log source を追加

成果物:

- `reports/judgment_materials_preview_latest.md`
- `reports/canonical_judgment_rules_preview_latest.md`

完了条件:

- Claude/Codex作業ログ由来の候補に `source_type=agent_work_log` を付与
- Userが実際に選択・修正・却下したものだけ `source=user_selected` へ進む
- 候補本文は `条件 -> 見る論点 -> 追加確認 -> 稟議に残す一文 -> 使わない条件` を持つ
- active store へ自動昇格しない
- Userレビュー前は `candidate` のまま

### Phase 4: Memory Effectiveness Observation

目的:

Obsidianに保存された知識が、実際に想起・使用・採用されたかを観測する。

Status: initial sidecar implemented on 2026-07-25.

実装:

- `scripts/obsidian_memory_effectiveness_report.py`
- `knowledge_refs`
- `memory_recall.refs`
- `judgment_asset_usage_feedback.jsonl`
- `shion_screening_review` 由来フィードバック

成果物:

- `reports/obsidian_memory_effectiveness_latest.md`
- `data/obsidian_memory_effectiveness.jsonl`

完了条件:

- dormant / recalled / used / validated / noisy の状態が見える
- RAG順位・プロンプト・回答挙動は変えない
- noisy候補を削除せず、レビューへ回す

### Phase 5: Search-first Wiki Maintenance

目的:

保存場所変更より前に、検索語・Related・ハブ導線を整える。

実装:

- `Projects/tune_lease_55/tune_lease_55 Wiki.md` のリンク案
- `Projects/tune_lease_55/検索語インデックス.md` の追加案
- 各主要ノートの `## Related` 追加案

成果物:

- `reports/obsidian_search_index_suggestions_latest.md`
- `reports/obsidian_related_link_suggestions_latest.md`

完了条件:

- Hub -> 検索語インデックス -> 元ノート の導線が見える
- Relatedは5〜8件程度の高信号リンクに限定
- キーワード一致だけでリンクを増やさない

### Phase 6: ADK Orchestrator

目的:

複数のVault管理役を紫苑ADKで束ねる。

実装:

- `api/shion_vault_manager_adk.py`
- `api/shion_vault_manager_tools.py`
- ツールは最初 read-only のみ

ADK構成:

- ParallelAgent:
  - librarian
  - curator
  - distiller
  - link_gardener
  - mana_gate
- Arbiter:
  - 統合レポートを作る
  - 変更案を `proposed_actions` として出す

完了条件:

- APIから read-only review を実行できる
- ADK未導入環境では通常スクリプトへフォールバック
- 失敗してもVaultは変わらない

### Phase 7: 承認付きVault編集

目的:

紫苑が提案した整理案を、人間承認後にだけ反映する。

実装:

- `scripts/apply_obsidian_review_plan.py`
- JSON planを読み、許可された操作だけ実行

許可操作:

- append_section
- create_note
- add_related_links
- add_frontmatter_field
- move_to_archive

禁止操作:

- delete_note
- overwrite_full_note
- raw_log_rewrite
- private_reflection_publication
- active_judgment_asset_auto_promotion
- bulk_okf_conversion
- raw_agent_log_to_rag_promotion

完了条件:

- dry-run差分が見える
- backup作成
- 承認なしでは書かない
- 書き込み後に検証レポートを作る

### Phase 8: 紫苑の反応改善へ接続

目的:

Vault管理結果を、紫苑の回答の深みに戻す。

接続するもの:

- distilled decisions
- user priorities
- rejected alternatives
- judgment asset candidates
- OKF風に構造化された lease_rule / risk_signal / agent_policy
- used / validated 状態になった記憶
- public-safe pitch phrases

接続しないもの:

- 生ログ全文
- private reflection原文
- 個人情報
- 未承認の攻撃的表現
- 機密案件情報

完了条件:

- 紫苑が長くならず、短く鋭くなる
- `memory_debug` に参照元が残る
- 出典が `agent_work_log_distillation` などで追跡できる
- recalled / used / validated の状態がログに残る

## 最初に作るべき最小機能

いきなりADK書き込みまで行かない。

まず作るのはこれ:

```text
Vault Management Review
読み取り専用でVaultを見て、
1. 生ログ
2. 蒸留済みメモ
3. agent_review_context
4. OKF化候補
5. 判断構文候補
6. public-safe候補
7. 要レビュー
を分けるレポート
```

この段階ではADKなしでもよい。

次にADKで複数レビュー役を束ねる。

## プレゼンでの説明

> 紫苑はObsidian Vaultを単なるメモ置き場として扱いません。  
> 審査判断、開発作業ログ、AIとの対話、改善履歴を分類し、判断資産候補として蒸留します。  
> ただし、Vaultの書き換えや判断資産への昇格は自動では行わず、人間承認を挟みます。  
> これにより、人間の判断とAIとの共同作業が、失われず、検索可能で、次の判断に使える知識になります。

## リスクと対策

| リスク | 対策 |
|---|---|
| Vaultを壊す | 初期はread-only。書き込みは承認付きplanのみ |
| 生ログがノイズになる | raw_log と distilled_note を分離 |
| 紫苑が冗長になる | 反応には蒸留済み要点だけ使う |
| 個人情報が混ざる | Mana Gateでpublic/privateを分ける |
| 判断資産が勝手に昇格する | candidate -> review -> active の人間承認を固定 |
| ADK未導入で落ちる | 通常Pythonスクリプトへフォールバック |

## 実装優先順位

1. `scripts/obsidian_vault_management_review.py` をread-onlyで作る
2. `raw_log / agent_review_context / OKF候補 / 判断構文候補 / public-safe / 要レビュー` に分類する
3. `reports/obsidian_vault_management_review_latest.md` を出す
4. Claude/Codex作業ログの蒸留レポートを作る
5. `docs/okf_subset.md` 準拠のOKF風ノート候補を10〜30件だけ作る
6. 判断資産候補へ `source_type=agent_work_log` と `memory_lane=agent_review_context` を足す
7. 記憶効果レポートで dormant / recalled / used / validated / noisy を出す
8. 検索語インデックス・Related案を作る
9. ADK Orchestrator を read-only で作る
10. 承認付きVault編集を作る
11. 紫苑の回答に used / validated / OKF化済みの蒸留結果だけを短く反映する

## 判断

ADKは「自律書き込みエージェント」ではなく、「複数の管理観点を統合する司令塔」として使うのがよい。

紫苑にVaultを管理させるなら、最初の正解は read-only review。  
書き込みは最後でよい。

保存方法の変更は、前に検討したOKF風サブセットに準じて小さく始める。  
Vault全体の再編ではなく、再利用される知識だけを構造化する。
