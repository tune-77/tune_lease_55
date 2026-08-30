# past_cases JSON内案件IDバックフィル計画

作成日: 2026-08-30

状態: `plan-only / not-executed`

対象: `past_cases.id` は存在するが、`past_cases.data.id` が空の案件

## 目的

DB行の正規案件IDをJSON本文へ補い、`load_all_cases()`、結果登録、追加確認、オントロジー参照で同じ案件IDを使えるようにする。

## 監査時点の対象

- `past_cases`: 2,329件
- JSON内ID一致: 2,190件
- JSON内ID欠損: 139件
- JSON内IDが非空だがDB行IDと不一致: 0件

件数は実行直前に再集計し、この値を固定値として使わない。

## 更新条件

次をすべて満たす行だけを対象にする。

1. `past_cases.id` が空でない。
2. `past_cases.data` が有効なJSONである。
3. `past_cases.data.id` が存在しない、null、または空文字である。
4. JSON内IDが非空の行は更新しない。
5. `excluded_grade_cases` はこの作業の対象外とする。

## 実施前チェック

1. 対象DBの絶対パスとバックエンドがSQLiteであることを確認する。
2. アプリからの書き込みを止め、監査開始後に件数が変化しない状態にする。
3. SQLiteの整合性チェックが `ok` であることを確認する。
4. DB本体と必要なWAL/SHMを一組として、日時付きの復元可能なコピーへ退避する。
5. 次の件数を記録する。
   - 全案件数
   - 有効JSON件数
   - JSON内ID欠損件数
   - JSON内ID不一致件数
   - `load_all_cases()` 相当の可視件数

## Dry-run

更新せず、対象行について次だけを計算する。

- 更新予定件数
- 更新後もJSONとして有効か
- 更新後の `data.id` とDB行IDが一致するか
- 既存キー数とJSONサイズの差分
- 企業名など他フィールドが変化していないかをハッシュで確認

Dry-runの更新予定件数と、実行直前の欠損件数が一致しなければ中止する。

## 更新方法

1トランザクション内で、JSON内IDが空の行だけにDB行IDを設定する。

概念上の更新式:

```sql
UPDATE past_cases
SET data = json_set(data, '$.id', id)
WHERE trim(id) <> ''
  AND json_valid(data)
  AND trim(COALESCE(json_extract(data, '$.id'), '')) = '';
```

実際の実行では、更新前件数を再確認し、`BEGIN IMMEDIATE`、更新件数確認、検証、`COMMIT` の順に行う。検証に失敗した場合は `ROLLBACK` する。

## 合格条件

- 全案件数が更新前と同じ。
- `past_cases.id` の一意件数が更新前と同じ。
- 全行のJSONが有効。
- JSON内ID欠損が0件。
- JSON内IDとDB行IDの不一致が0件。
- `load_all_cases()` 相当の可視件数が全案件数と一致。
- `screening_records`、`screening_outcomes`、レビューの件数が変化していない。
- スコア、判定、企業番号、会社名、物件情報などID以外のJSON値が変化していない。

## ロールバック

更新トランザクション内の検証失敗は `ROLLBACK` する。

コミット後に問題が見つかった場合は、アプリ書き込みを止め、作業前バックアップへDB一式を戻す。個別行の逆UPDATEでIDを消す方法は、更新後に新しい処理がそのIDを参照している可能性があるため使わない。

## 非対象

- Streamlit由来の孤立した `screening_records` 98件
- 借手IDの名称差分
- 個別物件IDの新設
- PostgreSQL / Cloud Run側データ
- スコア、プロンプト、RAG、判断資産の変更

## 実行承認

この文書は計画だけであり、DB更新を許可しない。実行時は、対象DB、バックアップ先、直前件数、停止時間を提示し、明示的な承認を得る。
