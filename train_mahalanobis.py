import sys
import os
import pandas as pd
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

try:
    from data_cases import load_all_cases
    from mahalanobis_engine import MahalanobisScorer
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

FEATURES = [
    "nenshu", "op_profit", "ord_profit", "net_income", "net_assets", "total_assets", "contracts",
    "dept_utsunomiya", "dept_oyama", "dept_ashikaga", "dept_saitama",
]

SYNTH_MEANS = {
    "nenshu": 500_000, "op_profit": 20_000, "ord_profit": 18_000,
    "net_income": 10_000, "net_assets": 150_000, "total_assets": 600_000, "contracts": 3,
    "dept_utsunomiya": 0.25, "dept_oyama": 0.25, "dept_ashikaga": 0.25, "dept_saitama": 0.25,
}
SYNTH_STDS = {
    "nenshu": 300_000, "op_profit": 15_000, "ord_profit": 13_000,
    "net_income": 8_000, "net_assets": 100_000, "total_assets": 400_000, "contracts": 2,
    "dept_utsunomiya": 0.43, "dept_oyama": 0.43, "dept_ashikaga": 0.43, "dept_saitama": 0.43,
}


_DEPT_MAP = {
    "dept_utsunomiya": "宇都宮営業部",
    "dept_oyama":      "小山営業部",
    "dept_ashikaga":   "足利営業部",
    "dept_saitama":    "埼玉営業部",
}


def _extract_val(c: dict, key: str) -> float:
    if key in _DEPT_MAP:
        sd = c.get("sales_dept") or (c.get("inputs") or {}).get("sales_dept", "未設定")
        return 1.0 if str(sd) == _DEPT_MAP[key] else 0.0

    fin = c.get("result", {}).get("financials", {})
    val = (
        c.get(key)
        or fin.get(key)
        or (fin.get("assets") if key == "total_assets" else None)
    )
    if key == "contracts" and (val is None or val == 0):
        return 1.0
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _synth_df(n: int = 80, rng: np.random.Generator | None = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(42)
    data = {}
    for f in FEATURES:
        vals = rng.normal(SYNTH_MEANS[f], SYNTH_STDS[f], n)
        data[f] = np.maximum(vals, 0)
    return pd.DataFrame(data)


def train() -> None:
    print("データ読込中...")
    all_cases = load_all_cases()
    success = [c for c in all_cases if c.get("final_status") == "成約"]
    print(f"成約案件: {len(success)} 件")

    rows = [
        {f: _extract_val(c, f) for f in FEATURES}
        for c in success
    ]
    df_real = pd.DataFrame(rows) if rows else pd.DataFrame(columns=FEATURES)

    if len(df_real) < 20:
        print(f"件数不足({len(df_real)}) → 合成データ80件で補強")
        df_synth = _synth_df(80)
        df = pd.concat([df_real, df_synth], ignore_index=True)
    else:
        df = df_real

    print(f"学習開始: {len(df)} 件 / 7次元")
    scorer = MahalanobisScorer(FEATURES)
    scorer.fit(df)

    os.makedirs("data", exist_ok=True)
    scorer.save("data/mahalanobis_model.joblib")
    print(f"保存完了: data/mahalanobis_model.joblib  (推定器={'MinCovDet' if len(df)>=30 else 'EmpiricalCovariance'})")


if __name__ == "__main__":
    train()
