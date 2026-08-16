# Raw / Wiki / Schema Vault Policy

作成日: 2026-08-13

## Purpose

紫苑の Vault は、情報を集める場所ではなく、判断資産が育つ場所として扱う。

そのため、保存物を `Raw` / `Wiki` / `Schema` の三層に分ける。層を分ける目的は、記録量を増やすことではなく、原文、解釈、機械可読ルールを混ぜないことにある。

## Core Rule

Raw は証拠、Wiki は意味、Schema は運用ルール。

どの層も単独では正解ではない。紫苑は Raw を根拠として残し、Wiki で人間が読める知識にし、Schema で再利用条件を明示する。

## Three Layers

| layer | role | examples | must not |
|---|---|---|---|
| `Raw` | 原文・一次記録を残す | クリップ記事、会話ログ、Cloud Run conversation、営業メモ、OCR結果、ニュース原文 | 要約だけ残して原文を捨てる |
| `Wiki` | 概念・論点・判断観点へ整理する | リース実践知ノート、業種別論点、補助金メモ、物件別ナレッジ、判断事例整理 | 未検証の断定を正本化する |
| `Schema` | 昇格条件・参照条件・状態遷移を定義する | 記憶分類、判断資産スキーマ、改善状態、stale/revised/deprecated、プロンプト注入条件 | 人間承認なしに審査スコアや承認可否へ直結する |

## Default Destinations

既定では通常の iCloud Obsidian Vault を使う。`lease-wiki-vault` は、ユーザーが明示した場合だけ使う。

| layer | normal destination | repository counterpart |
|---|---|---|
| `Raw` | `Projects/tune_lease_55/AI Chat/`, `Daily/`, `Projects/tune_lease_55/Research/`, `Projects/tune_lease_55/News/` | `memory/YYYY-MM-DD.md`, `data/cloudrun_chat_log.jsonl`, `reports/*` |
| `Wiki` | `Projects/tune_lease_55/`, topic notes, search index | `docs/*.md`, `knowledge_base/`, generated knowledge maps |
| `Schema` | policy notes when human-readable | `api/shion_memory_taxonomy.py`, `data/shion_memory_index.json`, `data/canonical_judgment_rules.json`, `specs/*`, `scripts/*_queue.py` |

実装で Vault パスが必要な場合は、必ず `runtime_paths.py` を通す。Vault パスを直接書かない。

## Promotion Flow

1. Capture: Raw に原文・日時・出典・文脈を残す。
2. Extract: 判断材料、論点、再利用できそうな基準を候補化する。
3. Review: 重複、古さ、誤読、個人情報、公開可否を確認する。
4. Wiki: 人間が読めるノートとして整理し、関連ノートへリンクする。
5. Schema: 使う場面、使わない場面、状態、信頼度、出典を構造化する。
6. Use: チャット、審査レビュー、改善候補で短く参照する。
7. Measure: `知識化率`、`再利用率`、`効果率`、`重複・陳腐化率` で棚卸しする。

## Promotion Conditions

Raw から Wiki へ上げてよい条件:

- 同じ論点が複数回出た。
- 1週間以上有効そうな判断基準、運用方針、ユーザーの好みである。
- 審査・知識回答・デモ説明で再利用できる。
- 出典と文脈が残っている。
- 個人情報や機微情報をそのまま露出しない形へ整理できる。

Wiki から Schema へ上げてよい条件:

- 使う場面と使わない場面が書ける。
- 人間レビュー済み、または `candidate` / `accepted_preview` などの検疫状態を持つ。
- 旧ルールと矛盾する場合、削除ではなく `revised` / `supersedes` で履歴を残せる。
- 実案件の承認/否決ではなく、判断材料や確認観点として使える。

## Stop Lines

次はやらない。

- Raw を消して Wiki だけ残す。
- Wiki の文章を、そのまま審査スコアや承認/否決へ接続する。
- 改善ログ、Private Reflection、日次カルテを自動で Schema へ昇格する。
- プラグインや外部知識基盤を必須経路にする。
- 情報源を増やすためだけにパイプラインを広げる。

## Current Strategy

当面は新基盤を導入しない。

既存の Obsidian、Markdown、JSON、RAG、改善パイプラインを使い、三層の名前と境界線を揃える。実装作業は、既存経路が詰まった時だけ小さく追加する。

優先順位:

1. Raw / Wiki / Schema の保存先と昇格条件を守る。
2. `docs/shion_information_health.md` の情報量上限を守る。
3. `docs/shion_memory_architecture.md` の記憶分類と整合させる。
4. 使われたか、効いたか、古くなったかを月次で見る。

## Related

- `OBSIDIAN_WIKI_WORKFLOW.md`
- `docs/self_maintaining_second_brain.md`
- `docs/shion_information_health.md`
- `docs/shion_memory_architecture.md`
- `docs/improvement_source_of_truth.md`
- `docs/knowledge_kpi_template.md`
