---
spec_id: P0-003
phase: 0
title: ml_features 週次スナップショット
status: approved
author: Claude Opus
reviewer: human
version: 1.0.0
depends_on: []
---

## 1. Goal
`lease_data.db` の `ml_features` テーブルを毎週日曜00:00（JST）にCSVとしてエクスポートし `data/snapshots/` 配下に保存する。古いスナップショット（90日超過）は自動削除しディスク容量を一定に保つ。

## 2. Scope
### In Scope
- `scripts/snapshot_ml_features.py`の新規作成
- `data/snapshots/ml_features_YYYYMMDD.csv` 形式での書き出し
- 90日超過ファイルの自動削除
- `.github/workflows/weekly_snapshot.yml` によるcron実行（毎週日曜00:00 JST = 土曜15:00 UTC）
- 実行ログの標準出力

### Out of Scope
- `ml_features`以外のテーブルのスナップショット
- S3/GCSへのアップロード
- ドリフト判定

## 3. Inputs / Outputs
| 区分 | 内容 |
|------|------|
| Input | `lease_data.db`（SQLite） |
| Output | `data/snapshots/ml_features_YYYYMMDD.csv`（UTF-8、ヘッダ付き、JSTの実行日） |
| Output | 標準出力ログ（取得件数、ファイルパス、削除件数） |
| Output | `data/snapshots/.gitkeep`（初回作成） |

## 4. Data Model
出力CSV: カンマ区切り、UTF-8 BOM無し、LF改行、1行目ヘッダ

## 5. API / Interface
CLI:
```
python scripts/snapshot_ml_features.py [--db PATH] [--table NAME] [--out-dir PATH] [--retention-days N] [--dry-run]
```
| オプション | 既定値 | 説明 |
|---|---|---|
| --db | lease_data.db | 入力SQLiteファイル |
| --table | ml_features | エクスポート対象テーブル |
| --out-dir | data/snapshots | 出力ディレクトリ |
| --retention-days | 90 | これより古いファイルを削除 |
| --dry-run | False | ログのみ出力 |

GitHub Actions cron: `"0 15 * * 6"` + workflow_dispatch

## 6. Business Rules
- **BR-001**: 入力DBが存在しない場合はexit code 2で異常終了、stderr に `db not found: <path>`
- **BR-002**: テーブルが存在しない場合はexit code 3で異常終了、stderr に `table not found: <name>`
- **BR-003**: 出力先ディレクトリが存在しない場合は自動作成
- **BR-004**: 同日付のCSVが既存の場合は上書き
- **BR-005**: 90日（既定）より古い `ml_features_*.csv` ファイルはmtimeで判定して削除
- **BR-006**: 削除対象は `ml_features_*.csv` パターンのみ（.gitkeep等は削除しない）
- **BR-007**: --dry-run時はファイルI/Oを行わずログのみ
- **BR-008**: タイムゾーンはJST固定（`zoneinfo.ZoneInfo("Asia/Tokyo")`）
- **BR-009**: CSVは1000行チャンクでストリーミング書き込み（メモリ節約）

## 7. UI / UX
CLIログ例:
```
[2026-05-17 00:00:01 JST] snapshot start: db=lease_data.db table=ml_features
[2026-05-17 00:00:02 JST] exporting 12,345 rows to data/snapshots/ml_features_20260517.csv
[2026-05-17 00:00:05 JST] export done (1.2 MB)
[2026-05-17 00:00:05 JST] deleted 1 file(s): ml_features_20260215.csv
[2026-05-17 00:00:05 JST] snapshot complete
```

## 8. Error Handling
| 状況 | 動作 | exit code |
|---|---|---|
| DBファイル不在 | stderr+終了 | 2 |
| テーブル不在 | stderr+終了 | 3 |
| 書き出し権限なし | stderr+終了 | 4 |
| 削除失敗（個別） | warningログ、処理継続 | 0 |
| 想定外例外 | traceback+終了 | 1 |

## 9. Acceptance Criteria
- **AC-001**: Given `ml_features`テーブルにN行存在する状態で、When スクリプトを実行する、Then `data/snapshots/ml_features_<JST日付YYYYMMDD>.csv` が作成されヘッダ1行+データN行を含む
- **AC-002**: Given `data/snapshots/`が存在しない状態で、When スクリプトを実行する、Then ディレクトリが自動作成された上でCSVが生成される
- **AC-003**: Given `data/snapshots/`に100日前mtimeの `ml_features_20260204.csv` が存在する状態で、When スクリプトを実行する、Then 当該ファイルが削除され新規CSVが作成される
- **AC-004**: Given DBファイルが存在しないパスを `--db` に指定した状態で、When スクリプトを実行する、Then exit code 2で終了しstderrに `db not found` を含む
- **AC-005**: Given `--dry-run` を付けて実行した状態で、When スクリプトが完了する、Then `data/snapshots/` 配下に新規ファイルは作成されずstdoutに `[dry-run]` プレフィックス付きのログが出力される
- **AC-006**: Given `.github/workflows/weekly_snapshot.yml` がmasterにマージされた状態で、When workflow_dispatchでワークフローを実行する、Then ジョブがsuccess完了し `data/snapshots/ml_features_<日付>.csv` が新規コミットされる
- **AC-007**: Given 100万行規模の `ml_features` テーブルを用意した状態で、When スクリプトを実行する、Then プロセスの最大RSSが500MBを超えない
- **AC-008**: Given 既に当日付のCSVが存在する状態で、When スクリプトを実行する、Then ファイルは上書きされ警告は出ない

## 10. Non-Functional
- Python: 3.11系
- 依存: 標準ライブラリのみ（sqlite3, csv, argparse, pathlib, datetime, zoneinfo, logging, sys）
- 実行時間: 10万行で30秒以内
- メモリ: 500MB以内

## 11. Implementation Notes（Codex向け）
関数構成:
- `main(argv) -> int`
- `validate_db(db_path: Path) -> None`（BR-001）
- `validate_table(conn, table: str) -> None`（BR-002）
- `export_to_csv(conn, table, out_path, chunk_size=1000, dry_run=False) -> int`（チャンク書き込み、tableは正規表現 `^[A-Za-z_][A-Za-z0-9_]*$` で検証）
- `enforce_retention(out_dir, days, dry_run=False) -> int`（mtimeベース削除）

日付取得: `datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")`
CSV: `csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")`、`open(path, "w", encoding="utf-8", newline="")`
ログ形式: `"[%(asctime)s JST] %(message)s"`
GitHub Actions workflow:
```yaml
name: weekly-ml-features-snapshot
on:
  schedule:
    - cron: "0 15 * * 6"
  workflow_dispatch: {}
jobs:
  snapshot:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    env:
      TZ: Asia/Tokyo
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Run snapshot
        run: python scripts/snapshot_ml_features.py
      - name: Commit snapshot
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/snapshots/
          if ! git diff --cached --quiet; then
            git commit -m "[P0-003] weekly snapshot $(date +%Y-%m-%d)"
            git push
          else
            echo "no changes"
          fi
```
ブランチ: `feature/p0-003-snapshot`、PRタイトル: `[P0-003] add weekly ml_features snapshot`

## 12. Test Plan
`tests/spec_phase0/test_P0-003.py` に以下を自動テストで実装：
- T-001（AC-001）: tmp_pathでダミーDB作成→スクリプト実行→CSV行数確認
- T-002（AC-002）: 存在しないout-dir指定→自動作成確認
- T-003（AC-003）: 100日前mtimeのダミーCSV配置→実行→削除確認
- T-004（AC-004）: 不在DBパス→exit code 2確認
- T-005（AC-005）: --dry-run→ファイル不作成+[dry-run]ログ確認
- T-008（AC-008）: 同日付CSV事前配置→上書き確認
