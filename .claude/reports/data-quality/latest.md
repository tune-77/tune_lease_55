---
agent: data-quality-checker
task: lease_data.db 全テーブル監査・データ減少原因特定
timestamp: 2026-03-28 14:55
status: success
reads_from: []
---

## サマリー

`lease_data.db` の `past_cases` テーブルが **本物の審査履歴 33件から手製のダミーデータ 8件に差し替えられている**。
`save_all_cases()` の `DELETE FROM past_cases` → 再INSERT ロジックが、ダミーデータを引数に呼び出されたことが原因と特定。
バックアップ（`data/backups/lease_data.db.20260328_0608`）には本物データ 33件が保存されている。

---

## データ消失の原因特定

### 発生タイムライン

| 時刻 | 状態 |
|------|------|
| 2026-03-27 20:55:35 | バックアップ `20260328_0608` 取得時点：`past_cases` 33件（本物データ） |
| 2026-03-28 09:51:29 | 現在の DB：`past_cases` 8件（ダミーデータに差し替え済み） |

差し替えは **2026-03-27 20:55 〜 2026-03-28 09:51 の間** に発生。

### 根本原因

`data_cases.py:save_all_cases()` 関数（行 188〜236）が以下の動作を持つ：

```python
cursor.execute("DELETE FROM past_cases")   # 全件削除
for data in cases:
    cursor.execute("INSERT INTO past_cases ...")  # 再INSERT
conn.commit()
```

この関数が **ダミーデータ 8件のリスト** を引数に呼び出された結果、本物データが全消去された。

### ダミーデータの特徴

現在の `past_cases` 8件は以下の特徴を持ち、本物の審査データではないと判断される：

- **ID 形式が全く異なる**：本物データは `2026XXXXXXXXXXXXXXXXXX`（20桁・タイムスタンプms）、現在は `202501XXXXXX`（12桁・連番）
- **borrower_name が人工的**：「優良商事」「破綻サービス」「競争敗退運輸」等の説明的な名称
- **`nenshu` の単位が混在**：一部は `1000000000`（1兆円相当、明らかに異常）、一部は欠損
- **タイムスタンプが等間隔**：全件が毎日 09:51:29 固定（実運用ではあり得ない）
- **`result` 構造が極端に簡素**：`{'score': 85.5}` のみ（本物は多数のフィールドを持つ）
- **`data` のキー構成が不統一**：各レコードで保持フィールドがバラバラ

---

## テーブル別品質レポート

### past_cases（審査履歴）
- 総レコード数: 8件（本来 33件以上が正常）
- 問題レコード数: 8件全て
- 欠損フィールド: `nenshu`（2件）、`lease_months`（多数）、`industry_code`、`grade`、`asset_name`
- 異常値:
  - id=202501010001: `nenshu=1000000000`（1兆円超、千円単位換算で1京円相当）
  - id=202501020002: `nenshu=500000000`（500億千万円相当、上限超過）
  - タイムスタンプが全件 09:51:29 固定（人工データの証拠）

### gunshi_cases（軍師AIログ）
- 総レコード数: 3件
- 問題レコード数: 0件（正常範囲）
- `seq=3`（sqlite_sequence）と一致、整合性問題なし

### subsidies（補助金マスタ）
- 総レコード数: 5件
- 問題レコード数: 0件
- `deadline` フォーマット: 全件「通年」で統一（「YYYY-MM-DD」形式は使われていない）
- `active` フラグ: 全件 1（問題なし）
- `max_amount`: 全件 1以上（問題なし）

### subsidy_master（補助金マスタ2）
- 総レコード数: 7件
- 問題レコード数: 0件
- `deadline` フォーマット: 全件「随時」で統一（要件適合）
- `active` フラグ: 全件 1（問題なし）
- 注意: `subsidies` と `subsidy_master` の 2 テーブルが同種データを重複管理している

### phrase_weights
- 総レコード数: 0件
- 未使用テーブル（問題なし）

### DB 容量情報（削除痕跡）
- freelist_count: 94 / 104 ページ（**90.4% が解放済み**）
- これは大量データが削除された痕跡であり、データ消失の物的証拠

---

## 発見された問題

- **[深刻度: HIGH]** `past_cases` 33件の本物審査データが消去され、8件のダミーデータに差し替えられている。スコアリングの比較分析・モデル学習・可視化が全て誤ったデータを参照している
- **[深刻度: HIGH]** `data_cases.py:save_all_cases()` が `DELETE FROM past_cases` → 再INSERT という危険な設計を持ち、誤った引数で呼ばれると全データを消去する。UPSERT（INSERT OR REPLACE）に変更すべき
- **[深刻度: MED]** `past_cases` の `nenshu` フィールドに 1兆円超の異常値が混入（1000000000, 500000000）。千円単位系でこの値は存在し得ない
- **[深刻度: MED]** `subsidies` と `subsidy_master` が重複テーブルとして存在。`ものづくり補助金` の max_amount が前者 1250万円、後者 1250万円（一致）だが `事業再構築補助金` は前者 1500万円、後者 7000万円と不一致
- **[深刻度: LOW]** `goofy-heyrovsky` ワークツリーの `data/` ディレクトリに `lease_data.db` が存在しない（ワークツリー間でデータが共有されていない）
- **[深刻度: LOW]** `lease_logic_sumaho12/.claude/worktrees/lease_logic_sumaho12/data/lease_data.db` が 12KB・`past_cases` 0件で空になっている

---

## バックアップ状況

```
data/backups/lease_data.db.20260328_0608  (2026-03-27 20:55, past_cases: 33件) ← 最新・本物データあり
data/backups/lease_data.db.20260327_0128  (2026-03-26 21:15, past_cases: 34件) ← 34件（さらに多い）
data/backups/lease_data.db.20260326_2056  (2026-03-26 20:51, past_cases: 33件)
data/backups/lease_data.db.20260326_0539  (2026-03-25 15:44, past_cases: 31件)
data/backups/lease_data.db.20260325_0511  (2026-03-24 11:58, past_cases: 29件)
data/backups/lease_data.db.20260324_1922  (2026-03-24 11:58, past_cases: 29件)
data/backups/lease_data.db.20260324_0249  (2026-03-23 08:45, past_cases: 28件)
```

**リストア対象**: `data/backups/lease_data.db.20260327_0128`（34件で最多）または `20260328_0608`（33件）

---

## screening_db.sqlite の状況

`data/screening_db.sqlite` に **153件**の審査記録が存在する（別テーブル `screening_records`）。
このファイルは `lease_data.db` とは独立しており、2026-03-01〜2026-03-28 の実際の審査ログを含む。
スコアリング画面はこちらを使用している可能性が高く、データ消失の影響を受けていない可能性がある。

---

## 課題・リスク

1. **即時リスク**: 現在のアプリが誤ったダミーデータを参照してスコア比較・傾向分析を行っている
2. **再発リスク**: `save_all_cases()` の全削除設計が残る限り、同様の事故が再発し得る
3. **データ二重管理リスク**: `subsidies` vs `subsidy_master` の不一致が補助金スコアリングに影響する可能性
4. **ワークツリー間 DB 同期問題**: 各ワークツリーが独立した DB を参照しており、本番データが特定パスにしか存在しない

---

## 後続エージェントへの申し送り

- **scoring-auditor**: `past_cases` がダミーデータ 8件しかない状態で類似案件スコアリング・比較分析が行われていた可能性を確認してください。`screening_db.sqlite`（153件）との使い分けも確認を
- **migration-validator**: `save_all_cases()` の `DELETE FROM past_cases` → 再INSERT 設計をUPSERT（`INSERT OR REPLACE`）に変更する移行が必要です。データ消失リスクの根本対策
- **復旧手順**（推奨）:
  ```bash
  cp data/backups/lease_data.db.20260327_0128 data/lease_data.db
  ```
  その後、消えた34件目（ID: `20260326211518452246`）は `20260327_0128` バックアップに存在するため完全復旧可能
