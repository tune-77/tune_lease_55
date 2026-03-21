"""
IV（情報価値）分析スクリプト
==============================
past_cases.jsonl の成約/失注データを使って
各変数のWoE（証拠の重み）とIV（情報価値）を計算し、
変数の予測力をランキング表示する。

使い方:
    python scripts/iv_analysis.py

IVの目安:
    < 0.02  : 使えない変数
    0.02-0.1: 弱い予測力
    0.1-0.3 : 中程度の予測力
    > 0.3   : 強い予測力
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ==============================
# データ読み込み
# ==============================
DATA_PATH = Path(__file__).parent.parent.parent / "past_cases.jsonl"
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent.parent / "past_cases.jsonl"

cases = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]
cases = [c for c in cases if c.get("final_status") in ("成約", "失注")]

n_events    = sum(1 for c in cases if c["final_status"] == "成約")
n_nonevents = sum(1 for c in cases if c["final_status"] == "失注")

print(f"分析対象: {len(cases)}件  成約:{n_events}件  失注:{n_nonevents}件")
if len(cases) < 30:
    print(f"⚠ 件数が少ないためIVの絶対値は参考程度。100件以上で信頼度が上がります。")


# ==============================
# DataFrameに展開
# ==============================
def build_df(cases):
    rows = []
    for c in cases:
        inp = c.get("inputs", {})
        res = c.get("result", {})
        qs  = inp.get("qualitative_scoring", {})
        rows.append({
            # ターゲット
            "target": 1 if c["final_status"] == "成約" else 0,
            # カテゴリ変数
            "industry_major": c.get("industry_major", "")[:3],
            "customer_type":  c.get("customer_type", ""),
            "main_bank":      c.get("main_bank", ""),
            "competitor":     c.get("competitor", ""),
            "grade":          inp.get("grade", ""),
            "contract_type":  inp.get("contract_type", ""),
            "deal_source":    inp.get("deal_source", ""),
            "lease_asset":    inp.get("lease_asset_name", ""),
            # 数値変数
            "nenshu":           inp.get("nenshu", 0),
            "op_profit":        inp.get("op_profit", 0),
            "ord_profit":       inp.get("ord_profit", 0),
            "bank_credit":      inp.get("bank_credit", 0),
            "lease_credit":     inp.get("lease_credit", 0),
            "acquisition_cost": inp.get("acquisition_cost", 0),
            "lease_asset_score":inp.get("lease_asset_score", 0),
            "contracts":        inp.get("contracts", 0),
            # スコア変数
            "score":          res.get("score", 0),
            "score_borrower": res.get("score_borrower", 0),
            "pd_percent":     res.get("pd_percent", 0),
            "qual_score":     qs.get("weighted_score", 0),
        })
    return pd.DataFrame(rows)


df = build_df(cases)

CAT_COLS = [
    "industry_major", "customer_type", "main_bank", "competitor",
    "grade", "contract_type", "deal_source", "lease_asset",
]
NUM_COLS = [
    "nenshu", "op_profit", "ord_profit", "bank_credit", "lease_credit",
    "acquisition_cost", "lease_asset_score", "contracts",
    "score", "score_borrower", "pd_percent", "qual_score",
]


# ==============================
# WoE / IV 計算
# ==============================
def calc_woe_iv(series, target, n_bins=4):
    """
    連続変数はn_bins分位でビニング、カテゴリ変数はそのままグループ化して
    WoE（各ビン）とIV（変数全体）を返す。
    """
    col = series.copy()
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if is_numeric:
        try:
            col = pd.qcut(col, q=n_bins, duplicates="drop")
        except Exception:
            col = pd.cut(col, bins=n_bins, duplicates="drop")

    grouped = (
        pd.DataFrame({"bin": col, "target": target})
        .groupby("bin", observed=True)["target"]
    )
    events    = grouped.sum()
    nonevents = grouped.count() - events

    iv = 0.0
    rows = []
    for b in events.index:
        e  = max(float(events[b]),    0.5)
        ne = max(float(nonevents[b]), 0.5)
        pct_e  = e  / n_events
        pct_ne = ne / n_nonevents
        woe = np.log(pct_ne / pct_e)
        iv += (pct_ne - pct_e) * woe
        rows.append({
            "bin":       str(b),
            "events":    int(events[b]),
            "nonevents": int(nonevents[b]),
            "pct_e":     round(pct_e, 3),
            "pct_ne":    round(pct_ne, 3),
            "WoE":       round(woe, 3),
        })
    return iv, rows


def iv_label(iv):
    if iv < 0.02:  return "× 使えない"
    if iv < 0.1:   return "△ 弱い"
    if iv < 0.3:   return "○ 中程度"
    return "◎ 強い"


# ==============================
# 全変数に対してIV計算
# ==============================
results    = []
detail_map = {}

for col in CAT_COLS + NUM_COLS:
    iv, rows = calc_woe_iv(df[col], df["target"])
    results.append({
        "変数": col,
        "IV":   round(iv, 4),
        "種別": "カテゴリ" if col in CAT_COLS else "数値",
    })
    detail_map[col] = rows

iv_df = (
    pd.DataFrame(results)
    .sort_values("IV", ascending=False)
    .reset_index(drop=True)
)
iv_df["判定"] = iv_df["IV"].apply(iv_label)


# ==============================
# 結果表示
# ==============================
print("\n" + "=" * 62)
print("  IV（情報価値）ランキング")
print("=" * 62)
print(f"{'順位':>4} {'変数':<22} {'IV':>8}  {'種別':>6}  判定")
print("-" * 62)
for i, row in iv_df.iterrows():
    print(f"{i+1:>4} {row['変数']:<22} {row['IV']:>8.4f}  {row['種別']:>6}  {row['判定']}")

print()
print("IVの目安: < 0.02 使えない / 0.02-0.1 弱い / 0.1-0.3 中程度 / > 0.3 強い")

# 上位5変数の詳細WoE
print("\n" + "=" * 62)
print("  上位5変数の詳細WoE（ビン別）")
print("=" * 62)
top5 = iv_df.head(5)["変数"].tolist()
for col in top5:
    iv_val = iv_df[iv_df["変数"] == col]["IV"].values[0]
    print(f"\n【{col}】  IV={iv_val:.4f}  {iv_label(iv_val)}")
    print(f"  {'ビン':<32} {'成約':>5} {'失注':>5} {'WoE':>8}")
    print("  " + "-" * 54)
    for r in detail_map[col]:
        print(f"  {r['bin']:<32} {r['events']:>5} {r['nonevents']:>5} {r['WoE']:>8.3f}")
