# /batch-export — バッチ審査エクスポート

## 使い方
```
/batch-export [--template] [--check <CSVファイルパス>] [--sample]
```
- 引数なし: バッチ審査の概要と必要なCSVカラムを表示
- `--template`: CSV テンプレートを `/tmp/batch_template.csv` に出力
- `--check <path>`: 指定CSVの形式検証（スコアリングは行わない）
- `--sample`: サンプルCSVで5件のバッチスコアリングを試行

## 処理手順

### デフォルトモード（引数なし）

**必要CSVカラムを表示する:**
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from components.batch_scoring import _CSV_COLUMNS, _CSV_SAMPLE

print('=== バッチ審査 CSVフォーマット ===\n')
print('必須カラム:')
for i, col in enumerate(_CSV_COLUMNS, 1):
    print(f'  {i:2d}. {col}')
print(f'\nサンプルデータ: {len(_CSV_SAMPLE)}件')
print('コマンド: /batch-export --template でテンプレートCSV出力')
print('コマンド: /batch-export --sample  でサンプル5件スコアリング試行')
"
```

### `--template` モード

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from components.batch_scoring import _CSV_SAMPLE

_CSV_SAMPLE.to_csv('/tmp/batch_template.csv', index=False, encoding='utf-8-sig')
print('✅ テンプレートCSV出力完了: /tmp/batch_template.csv')
print(f'行数: {len(_CSV_SAMPLE)}件（サンプルデータ入り）')
print('このファイルを編集して /batch-export --check で検証してください')
"
```

### `--check <path>` モード

```bash
python3 -c "
import sys, pandas as pd; sys.path.insert(0, '.')
from components.batch_scoring import _CSV_COLUMNS

path = '<引数のパス>'
try:
    df = pd.read_csv(path, encoding='utf-8-sig')
except Exception as e:
    df = pd.read_csv(path, encoding='utf-8')

missing = [c for c in _CSV_COLUMNS if c not in df.columns]
extra   = [c for c in df.columns if c not in _CSV_COLUMNS]

print(f'=== CSV検証: {path} ===')
print(f'行数: {len(df)}件  カラム数: {len(df.columns)}')
print()
if missing:
    print(f'❌ 不足カラム ({len(missing)}件):')
    for c in missing: print(f'   - {c}')
else:
    print('✅ 必須カラム: すべて存在')
if extra:
    print(f'ℹ️  追加カラム ({len(extra)}件): {extra}（無視されます）')

# NULL チェック
null_counts = df[_CSV_COLUMNS[:10]].isnull().sum()
problems = null_counts[null_counts > 0]
if not problems.empty:
    print(f'\n⚠️  NULL 値あり:')
    for col, cnt in problems.items():
        print(f'   {col}: {cnt}件')
else:
    print('✅ NULL値: なし')
print(f'\n{'✅ 検証OK — スコアリング実行可能' if not missing else '❌ フォーマットエラー — 修正が必要'}')
"
```

### `--sample` モード

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from components.batch_scoring import _CSV_SAMPLE, _score_batch_row
from data_cases import get_effective_coeffs, get_score_weights

coeffs  = get_effective_coeffs()
weights = get_score_weights()

print('=== サンプル5件 バッチスコアリング ===\n')
results = []
for _, row in _CSV_SAMPLE.iterrows():
    try:
        r = _score_batch_row(row, coeffs, weights)
        results.append(r)
        grade_icon = {'S': '🟣', 'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'}.get(r.get('grade', ''), '⚪')
        print(f\"{grade_icon} {r.get('company_name', row.get('業種小分類','?'))[:20]:<20} スコア:{r.get('score', 0):5.1f} グレード:{r.get('grade','?')}\")
    except Exception as e:
        print(f'❌ エラー: {e}')

print(f'\n集計: {len(results)}件処理  平均スコア:{sum(r.get(\"score\",0) for r in results)/len(results):.1f}')
print('本番実行は Streamlit UI の「バッチ審査」ページで行ってください')
"
```

## 注意事項
- このコマンドは **検証・サンプル実行のみ**（本番バッチは Streamlit UI）
- CSV は `utf-8-sig`（BOM付き）または `utf-8` エンコーディングを推奨
- 金額カラムは「万円」単位で入力（内部で千円換算）
- `物件ID` カラムは空欄可（空欄時はカテゴリ「その他」で計算）
- 出力ファイルは `/tmp/` に保存（コミット対象外）
