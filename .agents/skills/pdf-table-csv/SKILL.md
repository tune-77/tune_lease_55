---
name: pdf-table-csv
description: PDFスキャン表をページ分割し、ページPDFをOCRでCSV化し、リース審査CSV向けに金額を百万単位へ変換し、格付を正規化し、バッチ審査テンプレートCSVへ流し込むスキル。「PDFをページごと」「scan PDFをCSV」「OCRしてCSV」「数字を1000で割る」「百万単位」「格付を変換」「要注意先」「無格付」「batch_shinsa_template_current.csvに入力」「テンプレートに流し込み」「未入力は0」など、一連のスキャン表データ化作業では必ず使用する。
---

# リース審査PDFスキャン取込スキル

## 目的

スキャンPDFの表を、リース審査AIのバッチ審査テンプレートへ取り込めるCSVへ整える。

このSkillは、今回の作業で確立した以下の流れを再利用するためのもの。

1. 複数ページPDFをページごとのPDFに分割する
2. ページPDFを画像化する
3. 表の傾き補正と表領域の切り抜きを行う
4. 罫線からセルを検出する
5. セル単位でOCRしてCSVにする
6. 金額系カラムを1000で割り、百万単位へ変換する
7. `格付`列を業務ルールで正規化する
8. OCR済みCSVをバッチ審査テンプレートへ転記し、対応元がない列は`0`で埋める

## 一本化コマンド

PDFからOCR済みCSVを作り、百万単位化・格付正規化を行い、バッチ審査テンプレートまで流し込む場合は、まずこの`pipeline`を使う。

```bash
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py pipeline \
  /path/to/scan.pdf \
  /path/to/batch_shinsa_template_current.csv \
  --page 1
```

必要に応じて中間CSVや完成テンプレートの出力先を指定できる。

```bash
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py pipeline \
  /path/to/scan.pdf \
  /path/to/batch_shinsa_template_current.csv \
  --page 1 \
  --csv-out /path/to/scan-page-001.csv \
  --out-template /path/to/batch_ready.csv
```

`pipeline` は以下を一括実行する。

1. 指定ページを抽出してOCR
2. OCR前に表の傾き補正と表領域切り抜きを実施
3. 30列のOCR済みCSVを作成
4. 金額系カラムを1000で割って百万単位化
5. `格付`を正規化
6. バッチ審査テンプレート55列へ転記
7. テンプレートにない項目は`0`で補完

## 個別コマンド

`scripts/pdf_table_csv.py` を使う。

```bash
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py split /path/to/input.pdf
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py ocr-page /path/to/page.pdf --out /path/to/output.csv
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py scale-million /path/to/output.csv
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py normalize-rating /path/to/output.csv
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py fill-template /path/to/batch_shinsa_template_current.csv /path/to/source.csv
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py all /path/to/input.pdf --page 1 --out /path/to/output.csv
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py pipeline /path/to/input.pdf /path/to/batch_shinsa_template_current.csv --page 1
```

Downloadsなどワークスペース外へ書く場合は、通常の権限ルールに従って承認を取る。

## 依存関係

ローカルに以下がある前提。

- `pypdf`: PDF分割
- `fitz` / PyMuPDF: PDF画像化
- `Pillow`: 画像加工
- `opencv-python` (`cv2`): 罫線検出
- `ocrmac`: macOS Vision OCR

`ocrmac`/Vision OCRはサンドボックス内で失敗することがある。その場合は同じコマンドを権限昇格で再実行する。

## 標準ヘッダー

リース審査CSVは以下30列を標準にする。

```text
ユーザーコード,発生年月日,部署,商談区分,物件名,業種,銀行貸残,リース残,売上高,売上高総利益,営業利益,経常利益,当期利益,減価償却費,減価償却費(経費),機械・装置,その他有形固定資産,賃借料,賃借料(経費),格付,取引状態区分,契約種類,結果コード,結果,理由,商談ソースコード,商談ソース,取得価額,期間,利回り
```

OCRで表を読むとき、罫線検出列数が30列でない場合は、画像の向き・解像度・余白を確認してから続ける。

## 金額単位変換

次の金額系カラムだけ1000で割る。コード、日付、格付、結果コード、期間、利回りは変換しない。

```text
銀行貸残,リース残,売上高,売上高総利益,営業利益,経常利益,当期利益,減価償却費,減価償却費(経費),機械・装置,その他有形固定資産,賃借料,賃借料(経費),取得価額
```

変換例:

- `379979` -> `379.979`
- `11972858` -> `11972.858`
- `500` -> `0.5`
- `60`（期間） -> 変換しない
- `3.08%`（利回り） -> 変換しない

## 格付変換ルール

`格付`列は以下に正規化する。

- `1-3`系は `2` または `3` として扱う
- `4-6`系は `4`, `5`, `6` として扱う
- `8..`, `8_2`, `8-2`, `82`, `8` など8系は `要注意先`
- `0` と空欄は `無格付`

OCRで `82` と読まれた値は `8_2`相当として `要注意先`にする。

## 個別ワークフロー

### 複数ページPDFを分割

```bash
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py split /Users/.../Downloads/scan-002.pdf
```

出力例:

```text
scan-002-page-001.pdf
scan-002-page-002.pdf
...
```

### 1ページ目をCSV化して整形

```bash
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py all \
  /Users/.../Downloads/scan-002.pdf \
  --page 1 \
  --out /Users/.../Downloads/scan-002-page-001.csv
```

`all` は以下をまとめて実行する。

1. 指定ページを一時PDF/画像化
2. セル単位OCRでCSV化
3. 金額を百万単位へ変換
4. 格付を正規化

### 既存CSVだけ整える

```bash
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py scale-million /path/to/file.csv
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py normalize-rating /path/to/file.csv
```

### バッチ審査テンプレートへ流し込む

```bash
python3 .agents/skills/pdf-table-csv/scripts/pdf_table_csv.py fill-template \
  /Users/.../Downloads/batch_shinsa_template_current.csv \
  /Users/.../Downloads/scan-002-page-001.csv
```

`fill-template` は、テンプレート側のヘッダーを維持して、OCR済み30列CSVから対応できる項目だけを転記する。対応元がないテンプレート列は`0`で埋める。

主な対応:

```text
取引先ID <- ユーザーコード
審査日 <- 発生年月日（YYYY-MM-DDへ変換）
業種大分類 <- 業種
取引区分 <- 取引状態区分
営業担当部署 <- 部署
紹介元 <- 商談ソース（銀行なら銀行紹介、それ以外はその他）
検収時期(年) <- 発生年月日の年
リース期間(月) <- 期間
取得価格(百万円) <- 取得価額
物件名（任意） <- 物件名
契約種別 <- 契約種類（一般/自動車/割賦）
売上高・利益・資産・借入・リース残高 <- 対応する財務列
格付 <- 格付
最終結果 <- 結果（当社受注は成約、失注は失注）
獲得レート(%) <- 利回り（%を外す）
失注理由/結果登録メモ <- 理由
```

テンプレート上にあるがOCR済みCSVにない列（企業名、業種小分類、物件ID、物件スコア、競合情報、純資産、総資産、契約件数、定性スコア、日付詳細、基準金利、承認条件など）は`0`にする。

## OCR品質確認

OCR結果は完全ではない。完了後に必ず以下を確認する。

前処理後のグリッド検出では、通常は30列になる。ページ内に列名がなくても、列順が標準ヘッダーと同じなら問題ない。行数が明らかに多すぎる、先頭に空行が入る、ユーザーコードや日付の左端が欠ける場合は、傾き補正または表領域切り抜きが効いていない可能性が高い。

```bash
python3 - <<'PY'
import csv, collections
p = "/path/to/output.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))
print(len(rows), "rows")
print(collections.Counter(r["格付"] for r in rows))
for r in rows[:3]:
    print(r["ユーザーコード"], r["発生年月日"], r["物件名"], r["銀行貸残"], r["売上高"], r["取得価額"], r["期間"], r["利回り"])
PY
```

テンプレート流し込み後は、55列が維持されていることと主要列を確認する。

```bash
python3 - <<'PY'
import csv, collections
p = "/path/to/batch_shinsa_template_current.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))
print(len(rows), "rows", len(rows[0]), "cols")
print(collections.Counter(r["最終結果"] for r in rows))
for r in rows[:3]:
    print(r["取引先ID"], r["審査日"], r["物件名（任意）"], r["取得価格(百万円)"], r["格付"], r["獲得レート(%)"])
PY
```

日本語の細かい文字（部署、理由、業種、物件名）はOCR誤りが残りやすい。金額・期間・利回り・格付の整合性を優先して検査する。

## 注意

- 上書き前にはバックアップを作る。スクリプトはCSV整形系コマンドで自動的に `*_before_<operation>.csv` を作る。
- 元PDFがテキストPDFなら、まず `pypdf` の `extract_text()` を確認する。0文字ならスキャン画像としてOCRする。
- 表が横向きの場合はスクリプトが読みやすい向きへ回転して処理する。
