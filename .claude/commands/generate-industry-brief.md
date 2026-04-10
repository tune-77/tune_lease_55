# /generate-industry-brief — 業界動向レポート生成

## 使い方
```
/generate-industry-brief <業種名> [--cached] [--list]
```
- `<業種名>`: 業種の名称または業種コード（例: `製造業` / `13`）
- `--cached`: API呼び出しなし、既存キャッシュから表示のみ
- `--list`: 生成済みレポートの業種一覧を表示

**例:**
```
/generate-industry-brief 製造業
/generate-industry-brief 医療・福祉 --cached
/generate-industry-brief --list
```

## 処理手順

### `--list` モード

```bash
python3 -c "
import sys, json, os; sys.path.insert(0, '.')

output_file = 'industry_reports_a4.json'
if not os.path.exists(output_file):
    print('❌ industry_reports_a4.json が見つかりません')
    print('  /generate-industry-brief <業種名> で生成してください')
    exit(0)

with open(output_file, encoding='utf-8') as f:
    reports = json.load(f)

print(f'=== 生成済み業界レポート ({len(reports)}業種) ===\n')
for industry, data in sorted(reports.items()):
    ts = data.get('generated_at', '不明')[:10] if isinstance(data, dict) else '?'
    length = len(str(data.get('report', data))) if isinstance(data, dict) else len(str(data))
    print(f'  {industry:<25} {ts}  ({length}文字)')
print()
print('表示: /generate-industry-brief <業種名> --cached')
"
```

### `--cached` モード（APIなし）

```bash
python3 -c "
import sys, json, os; sys.path.insert(0, '.')

industry = '<引数の業種名>'
output_file = 'industry_reports_a4.json'

if not os.path.exists(output_file):
    print(f'❌ キャッシュなし: {output_file}')
    exit(1)

with open(output_file, encoding='utf-8') as f:
    reports = json.load(f)

# 部分一致検索
match = next((k for k in reports if industry in k or k in industry), None)
if not match:
    print(f'❌ \"{industry}\" のレポートが見つかりません')
    print(f'利用可能: {list(reports.keys())[:10]}')
    exit(1)

data = reports[match]
report_text = data.get('report', data) if isinstance(data, dict) else data
ts = data.get('generated_at', '不明') if isinstance(data, dict) else '不明'

print(f'=== 業界動向レポート: {match} ===')
print(f'生成日時: {ts}\n')
print(report_text)
"
```

### 新規生成モード（`<業種名>` のみ）

```bash
python3 -c "
import sys, json, os; sys.path.insert(0, '.')
from scripts.generate_industry_reports import (
    load_json, generate_report_for_industry, OUTPUT_FILE, JSIC_FILE
)

industry = '<引数の業種名>'

# 既存データ読み込み
jsic = load_json(JSIC_FILE)

# 業種のベース情報を取得（部分一致）
basic = next(
    (v for k, v in jsic.items() if industry in k or k in industry),
    f'{industry}に関するデータが登録されていません。'
)

print(f'🔄 \"{industry}\" のレポートを生成中（Gemini API使用）...\n')
try:
    report = generate_report_for_industry(industry, basic, '')
    print(report)

    # キャッシュ保存
    import datetime
    existing = load_json(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else {}
    existing[industry] = {
        'report': report,
        'generated_at': datetime.datetime.now().isoformat()
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f'\n✅ キャッシュ保存完了: {OUTPUT_FILE}')
except Exception as e:
    print(f'❌ 生成エラー: {e}')
    print('Gemini APIキーを確認してください（GEMINI_API_KEY または secrets.toml）')
"
```

### 出力フォーマット例

```
=== 業界動向レポート: 製造業 ===

【2025年 製造業 業界動向・課題・見通し】

1. 業界概況
   国内製造業は設備投資回復局面にあり...

2. 主要リスク要因
   原材料価格の高止まりと人手不足が...

3. リース需要見通し
   工作機械・産業用ロボットへの需要は...

4. 審査上の留意点
   財務安全性よりも受注残・稼働率を重視...
```

## 注意事項
- 新規生成は **Gemini API** を使用（`GEMINI_API_KEY` が必要）
- APIキーがない場合は `--cached` オプションでキャッシュを参照
- 生成結果は `industry_reports_a4.json` にキャッシュされる（再生成は上書き）
- `industry_trends_jsic.json` に業種ベース情報がない場合は汎用プロンプトで生成
- PDF 出力は Streamlit UI の「業種レポート」ページから行う
