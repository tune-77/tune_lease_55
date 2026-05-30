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

# net_assets / total_assets / contracts はDBに実データなし → 除外
# dep_expense(94%) / machines(88%) を追加
FEATURES = [
    "nenshu", "op_profit", "ord_profit", "net_income",
    "bank_credit", "dep_expense", "machines",
    "dept_utsunomiya", "dept_oyama", "dept_ashikaga", "dept_saitama",
]

SYNTH_MEANS = {
    "nenshu": 500_000, "op_profit": 20_000, "ord_profit": 18_000,
    "net_income": 10_000, "bank_credit": 50_000,
    "dep_expense": 5_000, "machines": 30_000,
    "dept_utsunomiya": 0.25, "dept_oyama": 0.25, "dept_ashikaga": 0.25, "dept_saitama": 0.25,
}
SYNTH_STDS = {
    "nenshu": 300_000, "op_profit": 15_000, "ord_profit": 13_000,
    "net_income": 8_000, "bank_credit": 40_000,
    "dep_expense": 4_000, "machines": 25_000,
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

    # inputs から取得（メイン経路）
    inp = c.get("inputs") or {}
    val = inp.get(key)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # フォールバック: トップレベル / result.financials
    fin = c.get("result", {}).get("financials", {})
    val = c.get(key) or fin.get(key)
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


def _calibrate_scale(scorer: MahalanobisScorer, df: pd.DataFrame) -> float:
    """
    学習データの距離分布から score_scale を自動設定。
    75パーセンタイル距離が score=50 になるよう調整。
    典型的な成約案件が 50pt 以上、外れ値が下回るスケール感。
    """
    import pandas as _pd
    scores_d = []
    for _, row in df.iterrows():
        _, d, _, _ = scorer.get_analysis(_pd.DataFrame([row.to_dict()]))
        scores_d.append(d)
    d75 = float(np.percentile(scores_d, 75))
    if d75 < 1e-6:
        return 5.0
    # score = 100 * exp(-d75 / scale) = 50  →  scale = d75 / ln(2)
    return round(d75 / np.log(2), 2)


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

    # 非ゼロ充填率を確認
    nonzero = (df_real != 0).mean()
    print("特徴量充填率:")
    for f in FEATURES:
        print(f"  {f}: {nonzero.get(f, 0):.0%}")

    if len(df_real) < 20:
        print(f"件数不足({len(df_real)}) → 合成データ80件で補強")
        df_synth = _synth_df(80)
        df = pd.concat([df_real, df_synth], ignore_index=True)
    else:
        df = df_real

    print(f"\n学習開始: {len(df)} 件 / {len(FEATURES)}次元")
    scorer = MahalanobisScorer(FEATURES)
    scorer.fit(df)

    # score_scale を自動キャリブレーション
    scale = _calibrate_scale(scorer, df.head(200))
    scorer.score_scale = scale
    print(f"score_scale 自動設定: {scale} (75パーセンタイル距離から算出)")

    os.makedirs("data", exist_ok=True)
    scorer.save("data/mahalanobis_model.joblib")
    print(f"保存完了: data/mahalanobis_model.joblib  (推定器={'MinCovDet' if len(df)>=30 else 'EmpiricalCovariance'})")

    # 簡易バックテスト
    _backtest(scorer, all_cases)


def _backtest(scorer: MahalanobisScorer, all_cases: list) -> None:
    import pandas as _pd
    won   = [c for c in all_cases if c.get("final_status") == "成約"][:400]
    lost  = [c for c in all_cases if c.get("final_status") == "失注"][:400]

    def get_score(cases):
        out = []
        for c in cases:
            row = {f: _extract_val(c, f) for f in FEATURES}
            s, d, _, _ = scorer.get_analysis(_pd.DataFrame([row]))
            out.append(s)
        return np.array(out)

    w_scores = get_score(won)
    l_scores = get_score(lost)

    print("\n=== バックテスト ===")
    print(f"成約  mean={w_scores.mean():.1f}  median={np.median(w_scores):.1f}")
    print(f"失注  mean={l_scores.mean():.1f}  median={np.median(l_scores):.1f}")
    print(f"差分  {l_scores.mean() - w_scores.mean():+.1f}  (正=失注が高い=逆方向、負=失注が低い=正常)")

    # pred_proba との相関（ml_featuresから）
    try:
        import sqlite3, json
        conn = sqlite3.connect(os.path.join(_DIR, "data", "lease_data.db"))
        ml_rows = conn.execute('''
            SELECT mf.pred_proba_v3, pc.data
            FROM ml_features mf JOIN past_cases pc ON mf.case_id=pc.id
            WHERE mf.pred_proba_v3 IS NOT NULL AND pc.data IS NOT NULL
            LIMIT 600
        ''').fetchall()
        conn.close()
        proba_list, score_list = [], []
        for proba, raw in ml_rows:
            c = json.loads(raw)
            row = {f: _extract_val(c, f) for f in FEATURES}
            s, _, _, _ = scorer.get_analysis(_pd.DataFrame([row]))
            proba_list.append(float(proba))
            score_list.append(s)
        r = float(np.corrcoef(proba_list, score_list)[0, 1])
        print(f"\nMahalanobis vs pred_proba_v3 相関: r={r:.3f}")
        print(f"  (目標: r < -0.20, 現在の旧実装: r=-0.159)")
        for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            p = np.array(proba_list); sc = np.array(score_list)
            mask = (p >= lo) & (p < hi)
            if mask.sum():
                print(f"  pred_proba {lo:.2f}-{hi:.2f}: score mean={sc[mask].mean():.1f} n={mask.sum()}")
    except Exception as e:
        print(f"pred_proba 相関チェック スキップ: {e}")


if __name__ == "__main__":
    train()
