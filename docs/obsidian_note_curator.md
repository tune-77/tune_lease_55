# Obsidian Note Curator / Taxonomy Audit / Theme Radar / Reflection Journal

内省モードの拡張。4つの独立したread-onlyまたはVault書き込みスクリプトで構成する。
いずれもこのリポジトリ内のコードは書けるが、実行対象のiCloud Obsidian Vaultへは
ユーザーのローカル環境（または日次パイプライン）からのみアクセスできる。

## スクリプト一覧

| スクリプト | 役割 | Vault書き込み | 外部呼び出し |
|---|---|---|---|
| `scripts/obsidian_taxonomy_audit.py` | 既存タグ・フォルダ名の重複検出、命名ルール提案（5個以内） | なし | なし |
| `scripts/obsidian_note_curator.py` | 新規追加ノートへタグ1〜3個・移動先フォルダを提案し関連ノートを提示。`--apply`で実適用 | あり（`--apply`時のみ） | Vertex AI（テキスト生成） |
| `scripts/obsidian_theme_radar.py` | Vault全体の繰り返しテーマ上位3件を抽出し、Vertex AI Google検索グラウンディングで実際の反響（バズ）を確認。一番強いテーマについて、リース審査AI/システム改善の切り口を1つ提案（バズの有無に関わらず必ず提案する） | なし | Vertex AI（Google検索グラウンディング＋テキスト生成） |
| `scripts/obsidian_reflection_journal.py` | `obsidian_theme_radar`の結果をProblem/気づき/解決の3段アウトライン（「解決」はリース審査AI/システム改善の具体アクション）にし、`System Improvement Reflection/`へ日次ノートとして記録 | あり | Vertex AI（テキスト生成） |

共通ヘルパー: `scripts/_obsidian_common.py`（frontmatter解析・Vault走査）、
`scripts/_vertex_text_gen.py`（Vertex AIプレーンテキスト生成。`api/vertex_agent_search.py`の
認証・プロジェクト設定・SDKパターンを再利用し、新しいAPIキー管理は増やしていない）。

## なぜPrivate Reflectionと別フォルダなのか

既存の `Private Reflection/` は紫苑（AI）自身が対話を振り返るための内省ログであり、
`mana_obsidian_curator.py` は「Obsidian本文への自動昇格をしない」ガードを明示的に持つ。
`obsidian_reflection_journal.py` が書くのはリース審査AI/システム改善についての
Problem/気づき/解決メモであり、性質が異なるため `System Improvement Reflection/`
（Vaultルート直下の新規フォルダ）に分離した。このフォルダの内容は、将来的に
`scripts/build_reflection_action_candidates.py` のようなREV改善候補パイプラインへ
人間が取り込むための素材という位置づけで、現時点では改善台帳への自動書き込みは行わない。

## 安全設計

- `obsidian_note_curator.py` はデフォルトでは何も書き込まない。実際にタグ付け・移動を
  行うには明示的に `--apply` を渡す（`cleanup_improvement_reviews.py --apply` などこの
  リポジトリの既存の apply フラグ慣習に合わせている）。
- `--apply` 実行前に対象ファイルを `data/obsidian_note_curator_backup/<run_id>/` へ
  コピーしてから書き換える。
- どのVertex呼び出しも失敗時は例外を投げず `status: unavailable/error` として処理を
  スキップする（`api/vertex_agent_search.py` の既存フォールバック方針と同じ）。
- 処理対象は `Private Reflection` / `Dialogue` / `AI Chat` / `Vertex Distilled` /
  `System Improvement Reflection` など運用・自己生成フォルダを除外し、自己参照ループを避ける
  （`VAULT_SCAN_EXCLUDE_DIR_NAMES`）。
- `obsidian_theme_radar.py` は、バズ確認（Vertex AI Google検索グラウンディング）が
  取れなくても、一番強い繰り返しテーマで必ず切り口を提案する（内省ループが「バズが
  無い日は何も出ない」で止まらないようにするため）。バズの有無自体はレポートに残し、
  切り口生成のプロンプトへ文脈として渡す。

## 状態管理

`obsidian_note_curator.py` は `data/obsidian_note_curator_state.json` に処理済みファイルの
相対パスとmtimeを保持し、次回実行時は新規/変更ファイルだけを対象にする。

## 日次パイプライン

`scripts/run_daily_improvement_post.sh` の末尾（Mana判定が `allow` の場合のみ実行される
ブロック内）に追記のみ・`|| true` 付きで4スクリプトを順番に呼ぶ:
taxonomy_audit → note_curator(--apply) → theme_radar → reflection_journal。

## 環境変数

`api/vertex_agent_search.py` の `VERTEX_AI_SEARCH_PROJECT_ID` 等をそのまま使う。
テキスト生成モデルは `VERTEX_TEXT_GEN_MODEL`（既定 `gemini-2.5-flash`）で上書きできる。
