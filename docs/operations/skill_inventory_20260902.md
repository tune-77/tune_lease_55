# Skill Inventory Audit — 2026-09-02

## 結論

- 物理ファイルは **45個**、一意なskill名は **43個**。
- `.claude/skills` は28個、`.agents/skills` は17個、総量は **6,891行**（レビュー修正後の作業ログ工程を含む）。
- 200行超は14個。うち10個は汎用開発skillで、プロジェクト固有ルールや互いの守備範囲と広く重なる。
- Claude Code履歴で明示実行を確認できたローカルskillは5系統だけ。`git-ship`、`restart-api`、`ponytail`、`ponytail-audit`、`ponytail-review`。
- 名前の完全重複は `git-ship` と `re-lease-count`。`git-ship` は本監査後に2定義を同一内容へ同期し、安全上の競合を解消した。
- プロジェクト運用のObsidian系5個の発火境界は整理済み。別に、ファイル形式・CLI用の `json-canvas` / `obsidian-bases` / `obsidian-cli` / `obsidian-markdown` も存在する。`ponytail` / `graft` は一度狭めたが、Userの明示方針によりコード作業では常時適用へ戻し、非コード依頼だけ除外した。残る主な競合候補は改善系3個、調査→審査資産化系3個、汎用開発系10個。
- 現時点では一括削除しない。まず重複定義の同期、発火境界の狭小化、30日利用計測の順で整理する。

## 調査範囲と判定方法

- 対象: `.claude/skills/*/SKILL.md` と `.agents/skills/*/SKILL.md`。
- Claude Code実行履歴: `~/.claude/projects/-Users-kobayashiisaoryou-clawd-tune-lease-55/*.jsonl` 54本。
- 履歴期間: **2026-07-09T13:41:30Z〜2026-08-30T00:25:32Z**。
- 明示実行は `Skill` tool use と `<command-name>` のみを数えた。セッション開始時の `skill_listing` は利用に数えていない。
- `.agents/skills` は主にCodex向けだが、Codex側には同等の完全なローカル呼出履歴が見つからない。このため、関連スクリプト・日次パイプライン・memory記録は「運用証拠」であり、skill自体の実行回数とは区別した。

### 利用区分

| 区分 | 意味 |
|---|---|
| A 明示実行 | Claude履歴でskillまたはslash commandの明示呼出しを確認 |
| B 運用証拠 | skill配下の処理や対応ワークフローの実行・配線は確認できるが、skill呼出し自体は断定できない |
| C 未確認 | 調査範囲内に明示呼出しも十分な運用証拠もない |

## 明示実行サマリー

| skill | `Skill` tool | slash command | 判定 |
|---|---:|---:|---|
| `git-ship` | 5 | 1 | 常用。2定義は2026-09-02に同期済み |
| `restart-api` | 3 | 2 | 常用。維持 |
| `ponytail` | 0 | 4 | 明示モードとして利用あり |
| `ponytail-audit` | 0 | 1 | 単発監査として利用あり |
| `ponytail-review` | 1 | 0 | 専門レビューとして利用あり |
| `graft` | 0 | 0 | skill呼出しは未確認。ただしGraft MCPは15回利用 |

`Skill` toolとslash commandは同じ作業内で重複する可能性があるため、合算して「総利用回数」とは扱わない。

## 45 skill全件一覧

| # | skill / 配置 | 行 | 主な発火条件 | 利用 | 重複・競合 | 棚卸し判断 |
|---:|---|---:|---|---|---|---|
| 1 | `api-and-interface-design` `.claude` | 294 | API、REST/GraphQL、型契約、公開境界 | C | security、frontend、incrementalと広く重なる | 30日計測。未使用なら汎用開発bundleへ統合候補 |
| 2 | `auto-improvement-pipeline` `.agents` | 410 | 「パイプライン実行」「改善を自動化」「リファクタリング」 | B | improvement-list/syncと「改善」で競合 | 日次処理は稼働中。SKILL.mdはreferences/scriptsへ分割候補 |
| 3 | `chat-quality-env-compare` `.agents` | 124 | Cloudflare版対Cloud Run版、memory_debug、RAG品質比較 | B | cloud-run-updateとは目的が別だが連続発火し得る | 固有価値あり。維持 |
| 4 | `cloud-run-update` `.agents` | 60 | Cloud Run更新、再デプロイ、bundle作成 | C | restart-next-cloudflareと「更新/再起動」が近い | 「本番Cloud Run deployのみ」とdescriptionを狭める |
| 5 | `code-review-and-quality` `.claude` | 347 | merge前レビュー、品質全般 | C | ponytail-review、security、TDD、AGENT code-reviewer | 広すぎる。正確性レビュー専用へ縮小かAGENTへ一本化 |
| 6 | `debugging-and-error-recovery` `.claude` | 300 | テスト失敗、build失敗、期待外挙動 | C | TDD、restart-api、build/test AGENT | 30日計測。障害原因調査だけに境界を狭める |
| 7 | `deprecation-and-migration` `.claude` | 206 | 旧機能/API削除、移行、sunset | C | incremental、api design、migration-validator | DB migrationを除外し、廃止判断専用なら維持可能 |
| 8 | `frontend-ui-engineering` `.claude` | 328 | UI、component、layout、state | C | API design、TDD、外部UI/design skill | 汎用すぎる。既存frontend rulesとの差分だけ残す候補 |
| 9 | `git-ship` `.claude` | 108 | ship、add/commit/push/merge | A: tool 5、slash 1 | `.agents/git-ship`と同名・同一内容 | 2026-09-02同期済み。利用あり、維持 |
| 10 | `git-ship` `.agents` | 108 | ship、add/commit/push/merge | C※Codex履歴なし | `.claude/git-ship`と同名・同一内容 | 2026-09-02同期済み。ミラー差分を定期検査 |
| 11 | `graft` `.claude` | 156 | 全コード作業のcontext router | B: MCP 15回 | 汎用開発skillと重なるがUser指定で常時適用 | 非コードを除外し、最初の1呼出しで十分なら追加呼出しを止める |
| 12 | `improvement-list` `.agents` | 50 | 改善リスト、最新改善レポート表示 | C | auto-improvement、syncと「改善」で競合 | 読取専用境界は明瞭。維持し30日計測 |
| 13 | `improvement-report-sync` `.agents` | 45 | 「済にして」「改善済み登録」「レポートへ反映」 | B | auto-improvement-pipelineにも同処理がある | 正本同期として維持。自動/手動入口を明記 |
| 14 | `incremental-implementation` `.claude` | 245 | 複数ファイル変更、大きい実装 | C | planning、TDD、ponytail、AGENTS標準プロトコル | 既存AGENTSと重複大。統合/休眠候補 |
| 15 | `judgment-asset-structurer` `.agents` | 71 | 判断資産化、知識化、候補化 | B | research-insights、source-validatorと連鎖 | 固有価値あり。「入力済み素材の構造化」に限定して維持 |
| 16 | `kb-report` `.agents` | 61 | promoted知識だけを根拠に追跡可能なMarkdownレポートを保存 | B | 2026-09-02に通常検索・保存・Wiki整理を除外 | 根拠付き保存レポート専用として維持 |
| 17 | `lease-source-validator` `.agents` | 60 | ソース確認、信頼性、鮮度、法令・ニュース | B | research-insightsの前段として連続発火 | 固有価値あり。外部情報の検証だけに限定して維持 |
| 18 | `memory-maintenance` `.claude` | 20 | MEMORY棚卸し、heartbeat | C | harness memory、日次自動昇格と競合 | 自動昇格済み。手動棚卸し専用として維持または休眠 |
| 19 | `obsidian` `.agents` | 160 | 通常のVault検索・閲覧・作成・追記・作業ログ | B | 2026-09-02に4用途とのルーティングを明記 | 通常Vault操作専用として維持 |
| 20 | `obsidian-save` `.claude` | 15 | 単に「Obsidian/Vaultへ保存」時の保存先判定 | C | 実書込みはobsidianへ委譲 | 保存先ポリシー専用として維持 |
| 21 | `obsidian-search-rule` `.claude` | 24 | Obsidian参照RAGコードの実装変更だけ | C | 通常のノート検索を明示除外 | コード変更時のみとして維持 |
| 22 | `pdf-table-csv` `.agents` | 253 | PDF分割、OCR、CSV、百万単位、格付変換 | B | PDF/OCR系ツールと隣接するが業務境界は固有 | 維持。手順詳細をreferencesへ分割候補 |
| 23 | `performance-optimization` `.claude` | 350 | 性能要件、回帰、Core Web Vitals、profiling | C | frontend、debugging、benchmark skill | 実測profiling後だけ発火するよう狭める |
| 24 | `planning-and-task-breakdown` `.claude` | 223 | spec分解、大規模タスク、並列化 | C | incremental、AGENTS標準プロトコル | 汎用重複。明示的な計画依頼だけに限定する候補 |
| 25 | `ponytail` `.claude` | 112 | 全コード作業の最小実装規律 | A: slash 4 | 汎用開発skillと重なるがUser指定で常時適用 | 非コードを除外し、コード作業中はfullを既定維持 |
| 26 | `ponytail-audit` `.claude` | 41 | repo全体の過剰設計・bloat監査 | A: slash 1 | ponytail-reviewとの差はrepo全体対diff | 境界明瞭。維持 |
| 27 | `ponytail-debt` `.claude` | 44 | ponytailコメント、先送り一覧 | C | orphan/instruction debtレポートと目的が近い | 30日未使用なら休眠候補 |
| 28 | `ponytail-gain` `.claude` | 50 | ponytail効果・scoreboard | C | 実repo値でなく固定benchmark表示 | 実務価値が薄い。最有力の休眠候補 |
| 29 | `ponytail-help` `.claude` | 71 | ponytail help、コマンド一覧 | C | 本体README/skill一覧で代替可能 | 本体へ統合または休眠候補 |
| 30 | `ponytail-review` `.claude` | 57 | diffの過剰設計レビュー | A: tool 1 | code-review-and-qualityと目的を分けられる | 複雑性専用として維持 |
| 31 | `re-lease-count` `.claude` | 94 | 再リース回数、期待使用期間 | C | `.agents/re-lease-count`と内容完全一致 | pytestによるbyte単位のミラー差分検知を追加済み |
| 32 | `re-lease-count` `.agents` | 94 | 再リース回数、期待使用期間 | C※Codex履歴なし | `.claude/re-lease-count`と内容完全一致 | pytestによるbyte単位のミラー差分検知を追加済み |
| 33 | `research-to-screening-insights` `.agents` | 66 | 調査を審査ポイント・確認質問へ変換 | B | source-validator、judgment-asset-structurerと連鎖 | 固有価値あり。検証済み素材→審査行動の変換に限定 |
| 34 | `restart-api` `.claude` | 70 | 再起動、API落ちた、フロント反映されない | A: tool 3、slash 2 | restart-next-cloudflareと「再起動」が競合 | 「ローカル3000/8000、公開tunnelなし」をdescriptionに明記 |
| 35 | `restart-next-cloudflare` `.agents` | 102 | 再起動＋Cloudflare/public URL | B | restart-apiと「再起動」が競合 | 「公開URL/tunnelが必要な時だけ」に限定して維持 |
| 36 | `scqa-report-writer` `.agents` | 67 | SCQA、報告文、発表、README、Slack | B | kb-reportと報告構造が隣接 | 明示的にSCQA指定された時だけに狭める |
| 37 | `screening-decision-flow-builder` `.agents` | 73 | フロー、決定木、条件分岐、審査プロセス図解 | B | judgment structurer、diagram系skillと隣接 | 意思決定構造専用で境界明瞭。維持 |
| 38 | `defuddle` `.claude` | 41 | URLから本文Markdownを抽出 | C | WebFetchと役割が重なる | Webページ本文抽出専用として維持候補 |
| 39 | `json-canvas` `.claude` | 244 | Obsidian Canvas・マインドマップ・フロー図 | C | diagram系skillと隣接 | `.canvas`形式の作成・編集専用として維持候補 |
| 40 | `obsidian-bases` `.claude` | 499 | Obsidian Basesのview・filter・formula | C | obsidianと単語が重なる | `.base`形式の作成・編集専用として維持候補 |
| 41 | `obsidian-cli` `.claude` | 106 | Obsidian CLIでVaultやpluginを操作 | C | obsidianの通常Vault操作と競合 | CLIまたはplugin/theme操作が必要な時だけ使用 |
| 42 | `obsidian-markdown` `.claude` | 196 | wikilink・embed・callout等のObsidian Markdown | C | obsidianのノート作成と隣接 | Obsidian固有Markdown構文が必要な時だけ使用 |
| 43 | `security-and-hardening` `.claude` | 349 | 入力、認証、保存、外部連携 | C | security-checker AGENT、security rule、API design | skillより専門AGENT/ルールへ一本化候補 |
| 44 | `test-driven-development` `.claude` | 383 | あらゆる実装・bugfix・挙動変更 | C | incremental、debugging、test-runner AGENT | 発火範囲が広すぎる。テスト先行を明示された時だけに狭める |
| 45 | `tune-lease-55-obsidian-wiki` `.agents` | 82 | Wiki hub、検索語索引、Relatedリンク整理 | C | 2026-09-02に単独保存・通常検索を除外 | 既存ノートのWiki構造整理専用として維持 |

## 重複・発火競合マップ

### 1. 完全重複

| 組み合わせ | 問題 | 推奨 |
|---|---|---|
| `.claude/git-ship` ↔ `.agents/git-ship` | 2026-09-02に同一内容へ同期。直接push章を除去 | 今後は内容ハッシュまたはdiffでミラー差分を検知する |
| `.claude/re-lease-count` ↔ `.agents/re-lease-count` | 94行が完全一致 | `tests/test_skill_mirrors.py` でbyte単位の一致を継続検査 |

### 2. 強い発火競合

| グループ | 競合する語・状況 | 望ましいルーティング |
|---|---|---|
| Obsidian運用 | `obsidian` / `obsidian-save` / `obsidian-search-rule` / `kb-report` / `tune-lease-55-obsidian-wiki` | 保存先判定=save、RAGコード変更=search-rule、根拠付き保存レポート=kb-report、リンク整理=wiki、その他Vault操作=obsidian |
| Obsidian形式・CLI | `json-canvas` / `obsidian-bases` / `obsidian-cli` / `obsidian-markdown` | `.canvas`=json-canvas、`.base`=bases、CLI・plugin/theme=cli、Obsidian固有Markdown=markdown |
| 汎用開発 | ponytail、TDD、incremental、planning、debugging、review、security、API、frontend、performance | 明示モードを優先。通常作業はAGENTS標準プロトコルを正本とし、専門条件がある時だけskillを追加 |
| 改善 | auto-improvement-pipeline / improvement-list / improvement-report-sync | 実行=auto、閲覧=list、実装済み状態更新=sync |
| 調査→判断資産 | lease-source-validator / research-to-screening-insights / judgment-asset-structurer | 情報源検査→審査行動化→候補構造化。毎回3つ全部を自動発火させない |
| 再起動 | restart-api / restart-next-cloudflare | ローカルのみ=restart-api、外部公開URLが必要=restart-next-cloudflare |
| 報告 | scqa-report-writer / kb-report | 構造指定=SCQA、Obsidian根拠＋reports保存=kb-report |

## 推奨アクション

### P0: 定義衝突を止める

1. ✅ `git-ship` 2定義の安全手順を同期した（2026-09-02）。
2. ✅ `re-lease-count` と `git-ship` の意図しないミラー差分をpytestで検知するようにした（2026-09-02）。
3. `restart-api` に「公開tunnelなし」、`restart-next-cloudflare` に「公開URLが必要な時だけ」を明記する。

### P1: 自動発火の取り合いを減らす

1. ↩️ `ponytail` は一度明示呼出し専用へ狭めたが、User方針によりコード作業の常時適用へ戻した。非コード依頼は除外する（2026-09-02）。
2. ↩️ `graft` は一度専門探索時だけへ狭めたが、User方針によりコード作業の常時context routerへ戻した。最初の呼出しで十分なら追加呼出しを止める（2026-09-02）。
3. ✅ Obsidian系5個を上記ルーティングへ合わせ、単語だけで競合する状態を解消した（2026-09-02）。
4. `scqa-report-writer` はSCQA明示時だけにする。

### 常時適用のコスト判断

- `ponytail` と `graft` の本文は合計268行・約15.7KBで、完全読込時は概算3,000〜4,000トークンの固定費になり得る。
- Graftが大きなソース読込を1回以上置き換える作業では、固定費を回収しやすい。
- 対象が既知の1ファイル小修正では、Graft読込分が純増になる可能性がある。
- Ponytailは検索コンテキストより、実装行数・説明量・不要な抽象化を減らす効果が中心。
- 現在の採用判断は「コード作業全体では長期的な削減効果を優先し、非コード依頼では読み込まない」。

### P2: 30日観測後に休眠判断

優先観測対象は、明示実行がなく説明が広い汎用開発skill 10個と、`ponytail-gain`、`ponytail-help`、`ponytail-debt`。削除ではなく、まず別ディレクトリへ退避できる候補として扱う。

## 30日利用計測の最小仕様

新しい大規模hookは追加しない。既存のskill呼出ログから、週1回次の項目だけ集計する。

- `skill_name`
- `invoked_at`
- `invocation_type`: `skill_tool` / `slash_command`
- `explicit_or_auto`
- `completed`: true/false
- `user_rework_signal`: 直後にやり直し・別skill指定があったか

30日後の判断基準:

- 明示利用あり、再作業なし → active維持。
- 自動発火あり、別skillへの切替が多い → description修正。
- 利用なしだが壊滅的事故を防ぐ専門skill →休眠させず条件を狭める。
- 利用なし、他skill/AGENT/ルールで完全代替 → archive候補。

## 制約

- この監査はskillの品質評価ではなく、配置、発火条件、利用証拠、重複の監査。
- 「C 未確認」は不要という意味ではない。Claude履歴期間外、Codex、手動スクリプト実行の利用は取りこぼす。
- skill一覧のロードだけでは、実際に指示が適用されたか判定できない。
- 本レポート作成時点ではskill定義の変更・削除・移動は行っていない。
