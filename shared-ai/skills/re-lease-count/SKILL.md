---
name: re-lease-count
description: 期待使用期間DATAから再リース回数を計算するスキル。「再リース回数」「再リース何回」「期待使用期間から計算」「re-lease count」などのキーワードが含まれる場合に使用する。スコアリングロジックへの組み込みやStreamlit UIウィジェット追加も対象。
---

# 再リース回数計算スキル

## データ構造

`期待使用期間.json` の `usage_period_data` に機種ごとの期待使用期間が格納されている。
`periods` の値は **期待使用期間（月数）** — リース期間別に異なる。

```json
{
  "item_name": "電子計算機",
  "legal_useful_life": 4,
  "periods": {
    "3y": 46,
    "4y": 66,
    "5y": 83,
    "6y": 95,
    "7y": 103,
    "8y": 110,
    "9y": 115,
    "10y_plus": 120
  }
}
```

例: 電子計算機（パソコン）をリース5年で契約 → `periods["5y"] = 83ヶ月 ≈ 7年` まで使われる可能性がある。

## 計算式

```python
import math

def _lease_months_to_period_key(lease_months: int) -> str:
    lease_years = math.ceil(lease_months / 12)
    if lease_years >= 10:
        return "10y_plus"
    return f"{lease_years}y"

# リース期間に対応する期待使用期間（月）を取得
period_key = _lease_months_to_period_key(lease_months)
expected_usage_months = periods_data.get(period_key)

# 再リース回数 = ceil((期待使用期間月 - リース期間月) / 12)
re_lease_count = max(0, math.ceil((expected_usage_months - lease_months) / 12))
```

**単位**: 年単位・端数切り上げ  
**最小値**: 0（期待使用期間 ≤ リース期間の場合）

## 計算例

| 機種 | リース期間 | periods値 | 再リース回数 |
|---|---|---|---|
| 電子計算機 | 5年(60ヶ月) | 83ヶ月 | 2回 `ceil((83-60)/12)` |
| 電子計算機 | 3年(36ヶ月) | 46ヶ月 | 1回 `ceil((46-36)/12)` |
| 通信機器   | 5年(60ヶ月) | 106ヶ月 | 4回 `ceil((106-60)/12)` |

## 実装済み箇所

`expected_usage_period.py` の `calc_lease_period_fit_score()` 返り値に `re_lease_count` フィールドが追加済み。

```python
result = calc_lease_period_fit_score(asset_name, lease_months)
re_lease_count = result["re_lease_count"]  # 取得方法
```

フォールバック（機種データなし or periodsキー未マッチ）は `re_lease_count = 0`。

## Streamlit UIへの追加

```python
import streamlit as st
from expected_usage_period import calc_lease_period_fit_score

def show_re_lease_count(asset_name: str, lease_months: int):
    result = calc_lease_period_fit_score(asset_name, lease_months)
    count = result["re_lease_count"]
    st.metric(
        label="再リース回数（見込み）",
        value=f"{count} 回",
        help="期待使用期間DATAから算出（リース期間別）"
    )
    return count
```

## 注意事項

- `periods` の値は**月数**（スコアではない）
- `legal_useful_life`（法定耐用年数）とは別の概念
- 数値単位: スコアリングモジュール内は「月」管理 → `lease_months / 12.0` で年換算
