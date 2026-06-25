# Phase B 設計書: GCS ↔ Obsidian Vault 双方向同期

> 作成日: 2026-06-26
> ステータス: **設計書（未実装）**
> 関連チケット: REV-162 / REV-163 / REV-164 / REV-165
> 前工程: [Cloud Run 移行設計書](cloud_run_migration_design.md)（Phase A）

---

## 0. 設計方針

ファイル種別ごとに「書き込み正本」を分離した **非対称双方向同期** を採用する。
競合が起きない構造にするために、両方向同時書き込みを禁止し、正本を一方に固定する。

---

## 1. アーキテクチャ図

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│   ローカル Mac（Obsidian）        │      │   Google Cloud Platform          │
│                                 │      │                                 │
│  iCloud Drive                   │      │  Cloud Storage (GCS)             │
│  ~/Library/Mobile Documents/... │      │  gs://tune-lease-55-vault/       │
│                                 │      │                                 │
│  [タイプ2 正本]                  │      │  ┌──────────────────────────┐    │
│  *.md (ユーザーが書くノート)      │──→──│  │ shion/mind.json          │    │
│  リース審査ナレッジ               │      │  │ shion/world_view.json    │    │
│  討論メモ                        │      │  │ (タイプ1 GCS正本)         │    │
│                                 │      │  │                          │    │
│  [タイプ1 読み取りキャッシュ]     │──←──│  │ obsidian_md/           ← │    │
│  shion/mind.json (コピー)        │      │  │  *.md (読み取りコピー)    │    │
│  shion/world_view.json (コピー)  │      │  └──────────────────────────┘    │
│                                 │      │                                 │
└─────────────────────────────────┘      │  Cloud Run (tune-lease-55-api)   │
                                         │  ・GCS への書き込み権限あり       │
                                         │  ・RAG は GCS の .md から読む     │
                                         └─────────────────────────────────┘

→ : iCloud → GCS（タイプ2: 差分アップロード、毎時バッチ）
← : GCS → iCloud（タイプ1: 一方向ダウンロード、Cloud Run 書き込み後）
```

---

## 2. ファイル種別と同期ルール

### タイプ1: GCS 正本（紫苑が生成・更新するファイル）

| 項目 | 内容 |
|---|---|
| 対象ファイル | `mind.json`、`world_view.json`、紫苑が生成する JSON 系ファイル |
| 書き込み正本 | GCS: `gs://tune-lease-55-vault/shion/` |
| 書き込み権限 | Cloud Run のみ（Service Account に限定） |
| ローカルの位置 | iCloud キャッシュ（読み取り専用として扱う） |
| 同期方向 | **GCS → ローカル** （一方向のみ） |
| 同期タイミング | Cloud Run が GCS へ書き込んだ直後、または定期ポーリング（5分間隔） |
| 競合制御 | GCS 上のロックファイル（`shion/.lock`）で排他制御 |
| 整合性保護 | `mind.json` は最優先。ロック取得失敗時は書き込みをリトライ・アラート |

**同期フロー（タイプ1）:**
```
Cloud Run が mind.json を更新する場合:
  1. GCS: shion/.lock を作成（ttl=60s、内容: {"writer": "cloud-run", "started_at": "..."}）
  2. GCS: shion/mind.json を上書き
  3. GCS: shion/.lock を削除
  4. （非同期）REV-162 スクリプトが GCS → iCloud へダウンロード
```

### タイプ2: iCloud 正本（ユーザーが書く知識ノート）

| 項目 | 内容 |
|---|---|
| 対象ファイル | Obsidian Vault の `*.md` ファイル（リース審査ナレッジ、討論メモ等） |
| 書き込み正本 | iCloud Drive（Obsidian の保存先） |
| GCS の位置 | `gs://tune-lease-55-vault/obsidian_md/` |
| GCS の役割 | Cloud Run の RAG 用「読み取りコピー」（正本ではない） |
| 同期方向 | **iCloud → GCS** （一方向のみ） |
| 同期タイミング | 毎時バッチ（または日中 4 回）、差分のみアップロード |
| バージョンチェック | GCS の `updated_at` メタデータと iCloud のファイル更新日時を比較し、ローカルが新しいファイルのみ上書き |
| 競合制御 | iCloud が常に正本のため GCS 上のファイルは書き換えない（Cloud Run は GCS を読み取るのみ） |

**同期フロー（タイプ2）:**
```
REV-163 スクリプト（毎時 cron）が実行する:
  1. iCloud Vault の *.md をリストアップ
  2. 各ファイルの mtime と GCS オブジェクトの updated メタデータを比較
  3. iCloud の mtime > GCS の updated のファイルのみ gsutil cp でアップロード
  4. 結果ログを ~/Library/Logs/tunelease/gcs_sync_YYYYMMDD.log に記録
  5. 失敗があれば Slack アラートを送信
```

---

## 3. 実装コンポーネント一覧

| コンポーネント | REV | ファイル（案） | 役割 |
|---|---|---|---|
| GCS → ローカル同期スクリプト | REV-162 | `scripts/sync_gcs_to_local.py` | タイプ1ファイルを GCS からダウンロード |
| iCloud → GCS 差分アップロード | REV-163 | `scripts/sync_local_to_gcs.py` | タイプ2 .md を差分で GCS へアップロード |
| GCS 書き込みロック機構 | REV-164 | `api/gcs_lock.py` | Cloud Run からの GCS 書き込み時の排他制御 |
| Cloud Run RAG の GCS 対応 | REV-165 | `api/main.py` + `obsidian_bridge.py` | RAG 参照先を iCloud パスから GCS コピーへ切り替え |
| ローカル LaunchAgent（タイプ2用） | REV-163 に含む | `launchd/com.tunelease.gcs_sync.plist` | macOS で毎時 cron 起動 |

---

## 4. 排他制御（ロック機構）詳細

対象: タイプ1ファイル（`mind.json` 等）の GCS 書き込み

```
GCS バケット構造:
gs://tune-lease-55-vault/
  shion/
    .lock                 # 書き込み中フラグ（ttl=60s）
    mind.json
    world_view.json
  obsidian_md/
    *.md                  # タイプ2 コピー（RAG 用）
```

**ロックファイル仕様:**
```json
{
  "writer": "cloud-run-instance-id",
  "started_at": "2026-06-26T12:00:00Z",
  "ttl_seconds": 60,
  "target_file": "shion/mind.json"
}
```

**ロック取得フロー（`api/gcs_lock.py`）:**
1. `shion/.lock` が存在するか確認
2. 存在する場合: `started_at + ttl_seconds < now` なら期限切れと判断して強制削除（ゾンビロック対策）
3. 存在しない場合: `if_generation_match=0` 条件付きアップロードで競合書き込みを防止
4. ロック取得後: 対象ファイルを書き込み → `.lock` 削除
5. 例外時: `.lock` を finally ブロックで必ず削除、エラーをアラート

---

## 5. バージョン管理とアラート

### バージョンチェック（タイプ2）

```python
# scripts/sync_local_to_gcs.py のロジック概要
for md_file in local_vault_md_files:
    local_mtime = os.path.getmtime(md_file)
    gcs_meta = gcs_client.get_blob_metadata(gcs_path)
    if gcs_meta is None or local_mtime > gcs_meta["updated"]:
        gcs_client.upload(md_file, gcs_path)
        log(f"uploaded: {md_file}")
    else:
        log(f"skipped (GCS is newer or equal): {md_file}")
```

### アラート条件

| 条件 | アラート先 | 内容 |
|---|---|---|
| GCS 書き込みロック取得失敗（リトライ 3 回後） | Slack `#shion-alerts` | mind.json 書き込み失敗の詳細 |
| iCloud → GCS アップロード連続失敗（3 ファイル以上） | Slack `#shion-alerts` | 失敗ファイル一覧 |
| タイプ1ファイルのローカルキャッシュが 10 分以上古い | ログのみ（INFO） | 最終同期時刻 |
| GCS の `mind.json` と `world_view.json` のサイズが 0 | Slack `#shion-alerts` | 整合性エラー（緊急） |

---

## 6. REVチケット一覧

| REV | タイトル | 依存 | 優先度 |
|---|---|---|---|
| REV-162 | GCS → ローカル同期スクリプト実装（mind.json 等、タイプ1） | Phase A（Cloud Run デプロイ完了） | 高 |
| REV-163 | iCloud → GCS 差分アップロードスクリプト実装（.md ファイル、タイプ2） | REV-162 と並行可 | 高 |
| REV-164 | GCS 書き込みロック機構実装 | REV-162 の前提 | 最高（REV-162 の前提） |
| REV-165 | Cloud Run RAG を GCS の .md コピーから読む対応 | REV-163 完了後 | 中 |

---

## 7. 未解決事項・リスク

| 事項 | 詳細 | 対応方針 |
|---|---|---|
| iCloud の同期遅延 | iCloud が macOS でまだ同期中の場合、ローカルファイルが古い可能性がある | `xattr` で iCloud 同期状態を確認してからアップロード |
| GCS バケット名・プロジェクト ID の確定 | Cloud Run 移行設計書に `gen-lang-client-0420497423` とあるが Phase B 用バケット名は未確定 | REV-162 着手前に確認が必要 |
| `mind.json` の JSON スキーマバリデーション | 不正な JSON を GCS に書き込まないための入力チェック | `api/gcs_lock.py` に json.loads バリデーションを含める |
| REV-165 で Obsidian パスの参照が残る場合 | `obsidian_bridge.py` に iCloud パスが散在している | 移行時に旧パス参照を feature flag で切り替え |
