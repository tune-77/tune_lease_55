# /financial-forecast — 財務予測・3期分析

## 使い方
```
/financial-forecast [--check] [--summary <案件ID>]
```
- 引数なし: `financial_analysis.py` の機能概要とバックエンド接続状況を確認
- `--check`: バックエンド API（FastAPI / TimesFM）への疎通確認
- `--summary <案件ID>`: 登録済み案件の財務サマリーを表示

**例:**
```
/financial-forecast --check
/financial-forecast --summary case_20240310_001
```

## 処理手順

### デフォルトモード（引数なし）

```bash
python3 -c "
import sys, os; sys.path.insert(0, '.')

# バックエンドURL確認
backend_url = os.environ.get('FINANCIAL_BACKEND_URL', 'http://localhost:8000')
print('=== 財務予測モジュール概要 ===\n')
print(f'バックエンドURL: {backend_url}')
print()
print('機能:')
print('  1. 3期分の財務データ入力（売上・営業利益・純資産）')
print('  2. YoY成長率・営業利益率・純資産比率の自動計算')
print('  3. TimesFM による将来予測（FastAPI 経由）')
print('  4. Gemini API による3行審査コメント生成')
print('  5. Plotly グラフ（実績+予測）の可視化')
print()
print('Streamlit UI で使用: 「📊 3期財務分析」ページ')
print()
print('疎通確認: /financial-forecast --check')
"
```

### `--check` モード

```bash
python3 -c "
import sys, os, requests; sys.path.insert(0, '.')

backend_url = os.environ.get('FINANCIAL_BACKEND_URL', 'http://localhost:8000')
results = {}

# FastAPI バックエンド確認
try:
    r = requests.get(f'{backend_url}/health', timeout=5)
    results['FastAPI'] = ('✅', f'HTTP {r.status_code}')
except Exception as e:
    results['FastAPI'] = ('❌', str(e)[:60])

# Gemini API 確認
try:
    import google.generativeai as genai
    results['Gemini API'] = ('✅', 'パッケージ利用可能')
except ImportError:
    results['Gemini API'] = ('⚠️', 'google-generativeai 未インストール')

# TimesFM 確認
try:
    import timesfm
    results['TimesFM'] = ('✅', 'パッケージ利用可能')
except ImportError:
    results['TimesFM'] = ('⚠️', 'timesfm 未インストール（予測機能は無効）')

print('=== 財務予測 依存サービス確認 ===\n')
for svc, (icon, msg) in results.items():
    print(f'{icon} {svc:<20} {msg}')

all_ok = all(r[0] == '✅' for r in results.values())
print(f\"\n{'✅ すべて正常' if all_ok else '⚠️  一部サービスに問題あり — /check-health で詳細確認'}\")
"
```

### `--summary <案件ID>` モード

```bash
python3 -c "
import sys, json; sys.path.insert(0, '.')
from data_cases import load_all_cases

case_id = '<引数のID>'
cases = load_all_cases()
case = next((c for c in cases if c.get('case_id') == case_id), None)

if not case:
    print(f'❌ 案件 {case_id} が見つかりません')
    exit(1)

inp = case.get('inputs', {})
print(f'=== 財務サマリー: {case_id} ===\n')

# 基本財務指標
metrics = [
    ('売上高', inp.get('nenshu'), '万円'),
    ('営業利益', inp.get('op_profit'), '万円'),
    ('経常利益', inp.get('ord_profit'), '万円'),
    ('純利益', inp.get('net_income'), '万円'),
    ('純資産', inp.get('jiko_shihon'), '万円'),
    ('総資産', inp.get('total_assets'), '万円'),
]
for label, val, unit in metrics:
    if val is not None:
        print(f'  {label:<12}: {val:>10,.0f} {unit}')

# 計算指標
nenshu = inp.get('nenshu', 0) or 1
op = inp.get('op_profit', 0) or 0
jiko = inp.get('jiko_shihon', 0) or 0
total = inp.get('total_assets', 0) or 1
print()
print(f'  営業利益率  : {op/nenshu*100:6.1f}%')
print(f'  自己資本比率: {jiko/total*100:6.1f}%')

# スコア
result = case.get('result', {})
if result.get('score'):
    print(f\"\n  総合スコア  : {result['score']:.1f}点  グレード: {result.get('grade', '?')}\")
"
```

## 注意事項
- 財務予測機能は **Streamlit UI「📊 3期財務分析」ページ** がメイン
- このコマンドは確認・デバッグ用途（予測グラフの生成は行わない）
- TimesFM が未インストールの場合、予測機能は無効になる（入力・計算は動作する）
- `FINANCIAL_BACKEND_URL` 環境変数でバックエンドURLを変更可能
- Gemini APIキーは `.streamlit/secrets.toml` または環境変数 `GEMINI_API_KEY` で設定
