# Obsidian 運用ガイド (tune_lease_55)

Vault は「審査ナレッジの置き場」であると同時に「RAG の索引元」でもある。
書き込み先と索引先がずれると、アプリは黙って古い知識で回答し続ける。
このドキュメントは**その分裂を起こさないための運用規約**をまとめる。

---

## 1. Vault パスの単一解決窓口

**`runtime_paths.py` が唯一の解決窓口。**新しいコードで Vault パスを直書きしない。
直書きが入ると整合性チェックが error で落ちる。

```python
from runtime_paths import (
    describe_obsidian_vault_resolution,
    resolve_icloud_obsidian_documents,
    resolve_obsidian_vault,
)

vault = resolve_obsidian_vault()               # Path が欲しいだけならこれ
docs = resolve_icloud_obsidian_documents()     # Vault の親（lease-wiki-vault などの隣接ノート用）

resolution = describe_obsidian_vault_resolution()   # 解決元と警告も欲しいとき
print(resolution.path, resolution.source, resolution.warnings)
```

シェルスクリプトからは共有ヘルパー経由で取る:

```bash
source "$(dirname "${BASH_SOURCE[0]}")/resolve_obsidian_vault.sh"
VAULT="$(resolve_obsidian_vault)"   # 解決できないときは空文字（呼び出し側でフォールバック）
```

### 解決順序（`runtime_paths.OBSIDIAN_VAULT_ENV_VARS`）

| 優先 | 解決元 | 備考 |
|---|---|---|
| 1 | `OBSIDIAN_VAULT_PATH` | 正準。plist ではこちらを主に使う |
| 2 | `OBSIDIAN_VAULT` | 後方互換。1 と**必ず同じ値**にする |
| 3 | `Documents/obsidian-vault` | 存在すれば採用（新ディレクトリ名） |
| 4 | `Documents/Obsidian Vault` | 存在すれば採用（旧ディレクトリ名。現行の実体） |
| 5 | 3 のパス | 見つからない場合のフォールバック。警告付き |

`describe_obsidian_vault_resolution()` は例外を投げず**警告を返す**。
夜間ジョブは環境が壊れていても走り切る必要があるが、壊れていることは
レポートに出さなければならない、という方針による。

### 警告が出る条件

- 2つの env が別の値を指している
- 新旧の Vault ディレクトリが両方存在する（＝書き込み先と索引先が分裂しうる）
- 解決したパスが存在しない

> **なぜ env 名が2つあるのか**
> 歴史的に `OBSIDIAN_VAULT_PATH` を先に読む実装（`runtime_paths` / RAG 側）と
> `OBSIDIAN_VAULT` を先に読む実装（環境モニター / 改善パイプライン）が併存し、
> 両方が設定されると参照先がずれていた。現在は順序を上表に統一し、
> plist は両方を同値で設定している。**片方だけ設定するのは禁止。**

---

## 2. launchd ジョブの運用規約

### 実行時刻（衝突しないよう分散させてある）

| 時刻 | 頻度 | ジョブ | Vault |
|---|---|---|---|
| 01:00 | 日曜 | `obsidian-backup` | 読み |
| 01:30 | 日曜 | `case-data-backup` | — |
| 02:00 | 毎日 | `aurion-core-midnight` | 書き |
| 03:00 | 毎日 | `obsidian-reindex` | 読み（ChromaDB 再索引） |
| 04:00 | 毎日 | `run_daily_improvement_pipeline.sh` | 読み書き（launchd 管理外） |
| 05:00 | 日曜 | `weekly-health-check` | 読み |
| 06:00 | 毎日 | `lease-news-collector` | 書き |
| 06:20 | 毎日 | `aurion-core-morning-report` | 読み書き |
| 06:40 | 毎日 | `lease-judgment-autoresearch` | 書き |
| 07:00 | 毎月1日 | `daily-knowledge-feed` | 書き |
| 07:20 | 毎月1日 | `prompt-feedback-monthly` | 書き |

**順序の意図**:
- バックアップ（01:00）は夜間の書き込み群より**前**に取る。書き込み途中の Vault を写さないため
- `aurion-core-midnight`（02:00）→ `obsidian-reindex`（03:00）の順。逆だと当夜の書き込みが索引に入るのが翌日になる
- `lease-news-collector`（06:00）→ `aurion-core-morning-report`（06:20）の順。朝報告が当日ニュースを読めるようにするため

### Python 実行系

**全ジョブがプロジェクトの venv (`.venv/bin/python`) を使う。** 以前は
`.venv/bin/python` / `anaconda3/bin/python` / `anaconda3/bin/python3` の3系統に
分かれており、あるジョブにだけ入っているパッケージに依存したコードが
別ジョブでは ImportError になる状態だった。

- plist の `ProgramArguments` と `PYTHON_BIN` は両方とも venv を指す
- `PATH` は `.venv/bin` を先頭に置く（anaconda はフォールバックとして末尾に残す）
- launchd は `.zshrc` を読まないため `/opt/homebrew/bin`（gcloud 用）も明示する
- シェルラッパーの既定も venv。`PYTHON_BIN` で上書きできる

実行系を変えたら、**Mac の実機で**依存が揃っているか確認する:

```bash
python3 scripts/check_launchd_python_env.py
```

各ジョブのエントリスクリプトのトップレベル import（欠けると起動時点で落ちる）と
遅延 import（欠けてもその経路だけ落ちる）を分けて検証する。

### ジョブを追加・変更するときの規約

1. **Vault を触るジョブは `OBSIDIAN_VAULT_PATH` と `OBSIDIAN_VAULT` を同値で両方設定する**
2. `scripts/check_obsidian_ops_consistency.py` の `VAULT_JOB_LABELS` にラベルを追加する
3. 他の Vault ジョブと **10分以上** 間隔を空ける
4. Python 実行系は venv に揃える
5. 変更後に整合性チェックを回す:

```bash
python3 scripts/check_obsidian_ops_consistency.py
python3 scripts/check_launchd_python_env.py      # Mac 実機で
python3 -m pytest tests/test_obsidian_ops_consistency.py tests/test_launchd_python_env.py -q
```

整合性チェックは日次改善パイプライン（`run_daily_improvement_post.sh`）からも毎朝走り、
結果は `reports/obsidian_ops_consistency_latest.json` に出る。以下は **error** として検出される:

- Vault の env が片方だけ／値が食い違う
- Vault ジョブ同士が同時刻に発火する
- Python 実行系が割れている
- Vault パスを直書きしたファイルが増えた

---

## 3. ノートの構造規約

### ハブとインデックス

1. `Projects/tune_lease_55/tune_lease_55 Wiki.md` をハブとして維持する
2. `Projects/tune_lease_55/検索語インデックス.md` を維持する
3. 主要ノートには `## Related` セクションを置く
4. 同義語・略語・英語名・ファイル名・API 名をノート冒頭付近に書く
5. 1ノート1トピックに保つ
6. 意味のある変更をしたときだけ、サニタイズした日次ログを書く

### リンクの張り方

- ハブ → トピックノート
- トピックノート → ハブ
- トピックノート → 兄弟ノート
- トピックノート → 検索語インデックス

### 検索語として足しておくもの

- 分析: `AUC`, `RandomForest`, `ロジスティック回帰`, `LGBM`, `quantitative`, `単位統一`
- AIエージェント: `軍師AI`, `マルチエージェント`, `石橋`, `風林火山`, `討論型審査`, `multi-agent-screening`
- 業務データ: `成約要因`, `失注要因`, `業種別傾向`, `審査ナレッジ`, `信用リスク群`
- 運用: `uv`, `py_compile`, `tsc`, `Daily Log`

### 監視対象ディレクトリ

`scripts/monitor_obsidian_environment.py` が鮮度を見ているのは以下:

| キー | パス |
|---|---|
| `daily` | `Daily/` |
| `private_reflection` | `Projects/tune_lease_55/Lease Intelligence/Private Reflection/` |
| `dialogue` | `Projects/tune_lease_55/Lease Intelligence/Dialogue/` |
| `cloudrun_conversation` | `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/` |
| `research` | `Projects/tune_lease_55/Research/` |
| `news` | `Projects/tune_lease_55/News/` |

ディレクトリを移動・改名するときは、このマップも合わせて更新する。

---

## 4. トラブルシューティング

### RAG が古い内容を返す

```bash
# 1. どの Vault を見ているか確認
python3 -c "from runtime_paths import describe_obsidian_vault_resolution as d; r=d(); print(r.path, r.source, r.warnings)"

# 2. launchd 側の設定と食い違っていないか確認
python3 scripts/check_obsidian_ops_consistency.py

# 3. 差分更新で足りなければ全件再構築
OBSIDIAN_RAG_FULL_REINDEX=1 .venv/bin/python -m mobile_app.rag_daily_maintenance
```

### 夜間ジョブが動いていない

launchd は `.zshrc` を読まない。`PATH` と Vault の env は plist 側で明示すること
（`gcloud` など外部コマンドも同様）。ログは以下:

- `~/Library/Logs/tune_lease_55_obsidian_reindex.{out,err}.log`
- `~/Library/Logs/tunelease/improvement_YYYYMMDD.log`
- `logs/aurion_daily/`

### バックアップ

```bash
python3 scripts/backup_obsidian_vault.py --dry-run   # 中身を確認
python3 scripts/backup_obsidian_vault.py --keep 14   # 最新14世代を保持
```

保存先は `OBSIDIAN_BACKUP_ROOT`（既定: iCloud Drive の `tune_lease_55_backups/obsidian`）。
