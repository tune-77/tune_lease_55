# Self-Maintaining Second Brain

作成日: 2026-08-14
素材: 添付画像「自己メンテ型セカンドブレインの5段階構造」

## Purpose

紫苑の知識基盤は、ノートを増やす場所ではなく、判断資産を腐らせずに使い続ける運用OSとして扱う。

この文書は、画像の5段階構造を tune_lease_55 向けに翻訳し、既存の `Raw / Wiki / Schema` 方針、経験フライホイール、記憶工学レポート、指示負債監査へ接続するための設計メモである。

## Core Problem

Obsidian、Notion、Markdown、JSON、RAG は、入力や更新が止まると急速に陳腐化する。

本質的な問題はツールではなく、次の3点にある。

- 追加した情報が、どの判断に効くか分からない。
- 使われない記憶、重複した記憶、古い指示が残り続ける。
- 人間の継続入力だけを前提にすると、運用が止まった瞬間に知識基盤が腐る。

したがって、紫苑のセカンドブレインは「保存」ではなく「監査、Skill化、構造化、可視化、自動運転」の流れで保つ。

## Five Stage Flow

| stage | image concept | tune_lease_55 mapping | existing entrypoints |
|---|---|---|---|
| 0 | 根本の問題 | 情報追加停止、候補過多、古い指示、未レビュー記憶で判断資産が腐る | `docs/raw_wiki_schema_vault_policy.md`, `reports/memory_engineering_latest.md` |
| 1 | ワークフロー監査 | 日々の会話、改善ログ、審査フィードバックから反復作業と人間判断の必要度を抽出する | `scripts/build_memory_engineering_report.py`, `scripts/build_experience_flywheel_report.py`, `scripts/build_instruction_debt_report.py` |
| 2 | ワークフローをSkill化 | 繰り返し実行する手順だけを `.agents/skills/` や scripts に固定する | `.agents/skills/*/SKILL.md`, `scripts/run_daily_improvement_core.sh` |
| 3 | フォルダ構成を組む | Raw は証拠、Wiki は意味、Schema は運用ルールとして分ける | `docs/raw_wiki_schema_vault_policy.md`, `docs/improvement_source_of_truth.md` |
| 4 | 可視化する | 候補圧、隔離量、再利用率、指示負債、抜けている自動化点を一望する | `reports/memory_engineering_latest.md`, `reports/instruction_debt_latest.md`, `reports/experience_flywheel_latest.md` |
| 5 | デプロイして自動運転させる | 日次パイプラインで観測し、低リスク処理は自動化し、高影響判断は人間レビューへ止める | `scripts/run_daily_improvement_core.sh`, pipeline ledger, review queues |

## Operating Model

### 0. 腐る前提で設計する

情報は追加した瞬間から古くなる。紫苑では、記憶や指示を正解として固定せず、作成日、根拠、適用条件、削除または改訂条件を持つ候補として扱う。

基本姿勢:

- Raw は消さず、証拠として残す。
- Wiki は意味づけであり、未検証の断定にしない。
- Schema は使い方のルールであり、自動承認やスコア変更へ直結しない。
- 古い記憶は即削除せず、`sleeping`、`stale`、`revised`、`deprecated` として状態を持たせる。

### 1. ワークフローを監査する

Claude/Codex が見るべきものは、単なるログ量ではなく反復性である。

監査対象:

- 同じ質問、同じ修正、同じ判断迷いが繰り返されているか。
- その作業は人間判断が必要か、機械的に候補化できるか。
- 候補生成が増えすぎ、レビューできない量になっていないか。
- 実案件、会話、改善ログ、評価セットのどこで再利用されたか。

判定の目安:

- `promote_to_review`: 判断資産候補として人間レビューへ回す。
- `replay_eval`: 回答品質の評価問題へ回す。
- `observe_only`: 記録だけ残し、まだ使わない。
- `quarantine`: ノイズ、私的すぎる内容、誤抽出としてプロンプトに入れない。

### 2. Skill化する

一度きりの作業をすぐSkill化しない。Skill化してよいのは、繰り返し実行され、手順の揺れが事故につながる作業だけである。

Skill化の条件:

- 同じ作業が複数回発生している。
- 手順が説明可能で、入力と出力が明確である。
- 失敗時の止め方が書ける。
- 自動実行してよい部分と、人間確認が必要な部分を分けられる。

Skill化しないもの:

- その場の思いつき。
- 高影響な審査判断そのもの。
- 未検証のプロンプト改善。
- Private Reflection の生ログ。

### 3. フォルダ構成を組む

保存場所は、情報の種類ではなく役割で分ける。

- `Raw`: 原文、会話、OCR、ニュース、操作ログ。
- `Wiki`: 人間が読める意味づけ、判断観点、業種別論点。
- `Schema`: 状態遷移、昇格条件、評価軸、機械可読ルール。

分離の目的は、検索精度よりも責任境界を明確にすることにある。

Raw を Wiki のように読ませると、雑音が判断を汚す。Wiki を Schema のように使うと、未検証の文章が自動判断へ混ざる。Schema を Raw のように増やすと、運用ルールが肥大化する。

### 4. 可視化する

可視化はきれいな図ではなく、次に何を止めるか、何をレビューするかを決めるために使う。

最低限見るもの:

- 候補量: review 待ちが増えすぎていないか。
- 隔離量: quarantine が多い抽出元はないか。
- 再利用: active rule が実際の回答や審査で使われているか。
- 効果: 使った結果、回答や判断が良くなったか。
- 指示負債: 古い指示、理由のない指示、削除条件のない指示が残っていないか。

この段階での出力は、ダッシュボードやレポートで十分。自動削除や自動昇格へ直接つなげない。

### 5. 自動運転させる

自動運転の対象は、観測、抽出、重複排除、レポート生成、低リスクな同期に限る。

人間レビューを必要とするもの:

- 判断資産の active 昇格。
- プロンプトへ常時注入するルール追加。
- 審査スコア、承認、否決に影響する変更。
- 個人情報、公開可否、価値判断を含む整理。

紫苑の自動運転は「勝手に賢くなる」ではなく、「腐りそうな場所を毎日見つけ、人間が判断しやすい形へ畳む」ことを目的にする。

## Practical Daily Loop

日次で見る順番:

1. `reports/memory_engineering_latest.md` の `Daily Review Focus` を見る。
2. review 候補を少数だけ採否する。
3. quarantine が多い抽出元を確認し、抽出条件を弱めるか保留する。
4. sleeping active rule を次案件で試すか、保留する。
5. 指示負債レポートで high guidance が再発していないかを見る。

やらないこと:

- 候補を一括で active にする。
- Raw ログを根拠なしに削る。
- レポートが出たこと自体を改善完了とみなす。
- 自動化ポイントを見つけるたびに新しいパイプラインを増やす。

## Fit With Shion

この構造は、紫苑を単なるRAGではなく、判断資産の運用OSとして扱うために使える。

紫苑にとって重要なのは、知識を検索できることだけではない。どの記憶を残すか、どの候補を寝かせるか、どの指示を消せる状態にするか、どこで人間レビューへ止めるかまで含めて運用することに価値がある。

そのため、5段階構造は次の言い換えで扱う。

- 監査: 何が繰り返され、何が腐り始めているかを見る。
- Skill化: 再現できる手順だけを固定する。
- 構造化: Raw / Wiki / Schema を混ぜない。
- 可視化: 候補圧、再利用、効果、負債を見える状態にする。
- 自動運転: 低リスク作業は回し、高影響判断は人間へ戻す。

## Stop Lines

- この文書を理由に、既存の日次改善パイプラインを拡張しない。
- この文書を理由に、未レビュー候補を自動でプロンプト注入しない。
- この文書を理由に、審査スコアや承認可否へ新しい自動判断を接続しない。
- 画像の一般論を、紫苑の正本ルールとして無批判に採用しない。

## Related

- `docs/raw_wiki_schema_vault_policy.md`
- `docs/improvement_source_of_truth.md`
- `docs/shion_second_brain_demo_talk_track.md`
- `reports/memory_engineering_latest.md`
- `reports/experience_flywheel_latest.md`
- `reports/instruction_debt_latest.md`
