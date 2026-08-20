# 改善台帳（ledger）とREV管理フロー

「台帳が複雑で分かりにくい」の実体を把握するためのマップ。

## 1. 台帳ファイルは2系統ある

| ファイル | 場所 | 誰が書くか | 内容 |
|---|---|---|---|
| `ledger.jsonl` | `~/Library/Logs/tunelease/ledger.jsonl`（ローカルPC、リポジトリ外） | `pipeline_ledger.py`（`.agents/skills/auto-improvement-pipeline/`） | 日次改善パイプラインが処理した**全アイデア**の履歴（applied/needs_review/parked/rejected/deleted） |
| `improvement_ledger.jsonl` | `scripts/improvement_ledger.jsonl`（リポジトリにコミット、295行） | `.github/workflows/ledger-sync.yml` が PRマージ時に `cleanup_improvement_reviews.py --apply` を実行 | **PRとREV番号の紐付けだけ**を反映したCI台帳 |

どちらも `cleanup_improvement_reviews.py` の同じロジックが書き込むが、宛先は環境変数 `LEDGER_PATH` で切り替わる（ローカル実行時はホームディレクトリ、CI実行時は `scripts/improvement_ledger.jsonl` に上書き）。**両者は自動では同期されない** — CI台帳はPRマージ起点の反映のみ、ローカル台帳は日次パイプラインの全ステータス遷移を持つため、内容が食い違いうる。

⚠️ **REV番号の採番はこの2系統とは別に、さらに `api/rule_engine/ledger_rules.json`（ビジネスルール台帳、後述4節）とも空間を共有している。** `analyze_error_logs.py` 等5スクリプトは `ledger_rules.json` だけを見て次のREV番号を決め、`step1_extract_and_structure.py` は `improvement_ledger.jsonl` だけを見て決めていたため、片方にしか記録が無い番号を「空き番号」と誤認して重複発行する事故があった（REV-292が別々の改善案に2回発行されたケース）。再発防止として `scripts/rev_ledger_utils.py` に `load_all_rev_sources()` を追加し、両ファイルを横断した上で採番するように6スクリプト全てを統一した（対応: 2026-08-20）。新しいREV発行元スクリプトを追加する場合も、独自に片方の台帳だけを読んで採番せず、必ずこの関数を経由すること。

⚠️ **`scripts/improvement_ledger.jsonl` は「CIの内部ファイル」ではなく本番稼働中の読み取り対象。** リネームや構造変更は以下に直接影響する:

- `api/main.py` の `_load_improvement_ledger_summary()`（chat API の応答内容に反映）
- `lease_intelligence_tools.py`（紫苑が「パイプラインの状況は？」に答える際のツール）
- `scripts/build_loop_proof.py` → `reports/loop_proof.html` → `frontend/src/app/loop-proof/page.tsx`（フロントエンド表示ページ）

このファイルに触れる変更は、上記3経路（chat応答・紫苑ツール・loop-proofページ）の実動作確認をセットで行うこと。

## 2. キーが2形式ある（CLAUDE.mdでも警告済み）

- `canonical_key` / `key`: `pipeline_ledger.compute_key()` が `title + description` を正規化してSHA1化した16文字（例: `misc_1a361409652f`）。**タイトルや説明文が少し変わると別キーになる。**
- `rev_id`: `REV-NNN` 形式。人間可読で、PRタイトルに書く番号（`feat: REV-039 ...`）。

同じ改善項目でも、起票時のcanonical_keyと、後から`cleanup_improvement_reviews.py`が再計算するcanonical_keyがズレることがある。実際に **REV-230 / REV-237 で発生**: reconciliation処理が孤立キー（`misc_dfe021de8b2b` / `misc_665e5e075fed`）に"applied"を書き込み、本来のエントリ（`misc_6be43f8b3dce` / `misc_64ce70442596`）は`needs_review`のまま取り残された（`cleanup_improvement_reviews.py` の `_get_rev_id_first_seen_key()` docstring参照）。

この事故の再発防止として「そのREV番号が台帳に最初に出現したときのkeyを本物のcanonical_keyとして扱う」ロジックが追加されているが、根本のズレ自体は解消されていない。

## 3. `cleanup_improvement_reviews.py` の手打ち補正テーブルは外部化済み

過去の不整合を個別に補正するテーブル群は、ロジックから切り離して
`scripts/cleanup_improvement_reviews_data.json` に外部化した（挙動は変更していない。
ロード後は元と同じ型・同じ変数名 `KNOWN_CODE_APPLIED` 等でモジュールから参照できる）。

| テーブル（JSONキー） | 件数 | 役割 |
|---|---|---|
| `known_code_applied` | 36件 | PRを経ずコミット直接実装が確認できた項目 |
| `known_title_applied` | 6件 | canonical_keyがREV番号と繋がらず`needs_review`で滞留する項目を、タイトル一致で強制決着 |
| `known_pr_overrides` | 6件 | PRタイトルにREV番号が無いが確認済みの対応関係 |
| `known_applied_no_pr` | 3件 | PRを経ずコードレビューで実装確認済み |
| `rev_titles` | 48件 | REV番号 → タイトルのマスタ辞書（本来は台帳やreportsにあるはずの情報の写し） |

このテーブル自体は「消えずに増え続ける」性質そのものは変わっていない —
外部化はコードと台帳データの分離であって、根本原因（キー不一致で新しい例外が発生し続けること）の解消ではない。

## 4. 「ledger」という名前が別ドメインでも使われている（紛らわしいが無関係）

REV改善台帳とは別物:

- `api/rule_engine/ledger_rules.json` / `ledger_rules_archive.json` — ビジネスルールの台帳
- `api/shion_action_ledger.py` — 紫苑（AIエージェント）の行動記録
- `scripts/create_judgment_drill_ledger.py` / `scripts/build_agent_action_ledger_report.py` — 判断ドリル・エージェント行動レポート
- `scripts/sync_ledger_to_gcs.py` — 上記いずれかをGCSへ同期（対象は要確認）

`scripts/rev_ledger_utils.py` のdocstringが明言する通り「台帳ファイルは複数存在する」ことは既知の前提。REV採番の重複事故（`step1_extract_and_structure.py`が採番規約に従わず衝突）を機に、REV最大値取得だけは `rev_ledger_utils.max_rev_number()` に一本化されている。

## 5. 変更前に確認すること

- `ledger.jsonl` は追記形式・最後のエントリが有効（CLAUDE.md記載の通り）
- PRタイトルに `REV-NNN` が無いと `cleanup_improvement_reviews.py --apply` が台帳を更新できない
- `LEDGER_PATH` を変更する・台帳フォーマットを変える場合は `.github/workflows/ledger-sync.yml`（PATベースのauto-merge PR作成フロー）と `pipeline_ledger.py` の両方に影響するため、片方だけ直さない
