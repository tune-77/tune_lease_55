# -*- coding: utf-8 -*-
"""
timesfm_engine.py
=================
TimesFM（Google Research 時系列 Foundation Model）ラッパー。
未インストール時は GBM（幾何ブラウン運動）にフォールバックして動作継続する。

ユースケース:
  1. forecast_financial_paths()  — モンテカルロパス生成（montecarlo.py の _gbm_paths 代替）
  2. forecast_company_score()    — 同一企業の過去スコア時系列から将来スコアを予測
  3. forecast_industry_trend()   — 業種内スコア月次集計からトレンドを予測
  4. forecast_final_rate()       — 成約案件の金利時系列から将来金利帯を予測
  5. forecast_base_rate()        — 基準金利マスタ（期間別9列）の時系列から将来金利を予測
  6. forecast_base_rate_all()    — 全9期間列の一括予測
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

# ── TimesFM インポート試行 ────────────────────────────────────────────────────
try:
    import timesfm
    _TFM = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=128,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m",
        ),
    )
    TIMESFM_AVAILABLE = True
except Exception:
    _TFM = None
    TIMESFM_AVAILABLE = False


# ── 内部 GBM ヘルパー ─────────────────────────────────────────────────────────

def _gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    n_periods: int,
    n_paths: int,
    dt: float = 1 / 12,
    seed: int = 42,
) -> np.ndarray:
    """GBM で shape (n_paths, n_periods+1) のパスを生成する（フォールバック用）。"""
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_periods + 1))
    paths[:, 0] = S0
    for t in range(1, n_periods + 1):
        z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
        )
    return paths


def _timesfm_point_forecast(
    context: list[float],
    horizon: int,
) -> np.ndarray:
    """
    TimesFM で中央値予測を返す。
    返値: shape (horizon,) の予測値配列
    エラー時は空配列を返す。
    """
    if _TFM is None or not context:
        return np.array([])
    try:
        import pandas as pd
        df = pd.DataFrame({"unique_id": ["x"] * len(context),
                           "ds": range(len(context)),
                           "y": context})
        forecast_df, _ = _TFM.forecast_on_df(
            inputs=df,
            freq="M",
            value_name="y",
            num_jobs=1,
        )
        # 中央値列 (timesfm-1.0 では "timesfm" 列)
        col = [c for c in forecast_df.columns if "timesfm" in c.lower()]
        if col:
            return forecast_df[col[0]].values[:horizon]
        return np.array([])
    except Exception:
        return np.array([])


# ── ユースケース 1: モンテカルロパス生成 ──────────────────────────────────────

def forecast_financial_paths(
    historical_values: list[float],
    n_periods: int,
    n_paths: int = 10_000,
    fallback_mu: float = 0.02,
    fallback_sigma: float = 0.15,
    dt: float = 1 / 12,
) -> np.ndarray:
    """
    TimesFM の点予測を中央として不確実性帯からモンテカルロパスを生成する。
    TimesFM 未インストール時は GBM にフォールバック。

    Args:
        historical_values: 過去の財務指標値リスト（売上高など）
        n_periods:          予測期間（月数）
        n_paths:            生成パス数
        fallback_mu:        GBM ドリフト（フォールバック時）
        fallback_sigma:     GBM ボラティリティ（フォールバック時）
        dt:                 時間刻み（月次=1/12）

    Returns:
        np.ndarray: shape (n_paths, n_periods+1)
    """
    S0 = historical_values[-1] if historical_values else 1.0

    # TimesFM による点予測
    median_forecast = np.array([])
    if TIMESFM_AVAILABLE and len(historical_values) >= 6:
        median_forecast = _timesfm_point_forecast(historical_values, n_periods)

    if len(median_forecast) == n_periods:
        # TimesFM 成功: 各時点の予測値を中心に正規分布でパスをサンプリング
        # ボラティリティは歴史値から推定
        log_returns = np.diff(np.log(np.clip(historical_values, 1e-6, None)))
        sigma_est = float(np.std(log_returns)) if len(log_returns) > 1 else fallback_sigma

        rng = np.random.default_rng(42)
        paths = np.zeros((n_paths, n_periods + 1))
        paths[:, 0] = S0
        for t in range(1, n_periods + 1):
            center = median_forecast[t - 1]
            noise = rng.normal(0, sigma_est * math.sqrt(dt), n_paths)
            paths[:, t] = center * np.exp(noise)
        return paths

    # GBM フォールバック
    return _gbm_paths(S0, fallback_mu, fallback_sigma, n_periods, n_paths, dt)


# ── ユースケース 2: 企業の将来スコア予測 ──────────────────────────────────────

def forecast_company_score(
    case_history: list[dict],
    horizon_months: int = 12,
) -> dict:
    """
    同一企業の過去案件リスト（timestampでソート済み）からスコア時系列を構築し、
    TimesFM で将来スコアを予測する。

    Args:
        case_history: [{"timestamp": "2025-01-15T...", "score": 75.2, ...}, ...]
        horizon_months: 予測月数

    Returns:
        {
            "score_history":   list[float],   # 過去スコア
            "score_forecast":  list[float],   # 将来スコア (horizon_months 点)
            "band_low":        list[float],   # 予測下限 (-1σ相当)
            "band_high":       list[float],   # 予測上限 (+1σ相当)
            "trend":           str,           # "up" / "down" / "stable"
            "method":          str,           # "timesfm" / "linear_extrapolation"
        }
    """
    scores = [c["score"] for c in case_history if c.get("score") is not None]
    if not scores:
        return {"error": "スコアデータがありません"}

    method = "linear_extrapolation"
    forecast: list[float] = []
    band_low: list[float] = []
    band_high: list[float] = []

    if TIMESFM_AVAILABLE and len(scores) >= 4:
        raw = _timesfm_point_forecast(scores, horizon_months)
        if len(raw) == horizon_months:
            forecast = raw.tolist()
            method = "timesfm"

    if not forecast:
        # 線形外挿フォールバック
        if len(scores) >= 2:
            slope = (scores[-1] - scores[0]) / max(len(scores) - 1, 1)
        else:
            slope = 0.0
        forecast = [scores[-1] + slope * (i + 1) for i in range(horizon_months)]

    # 不確実性帯（過去スコア標準偏差を利用）
    sigma = float(np.std(scores)) if len(scores) > 1 else 2.0
    band_low  = [max(0.0, f - sigma) for f in forecast]
    band_high = [min(100.0, f + sigma) for f in forecast]

    # トレンド判定
    if forecast[-1] > scores[-1] + 1:
        trend = "up"
    elif forecast[-1] < scores[-1] - 1:
        trend = "down"
    else:
        trend = "stable"

    return {
        "score_history":  scores,
        "score_forecast": forecast,
        "band_low":       band_low,
        "band_high":      band_high,
        "trend":          trend,
        "method":         method,
    }


# ── ユースケース 3: 業種別トレンド予測 ────────────────────────────────────────

def forecast_industry_trend(
    industry: str,
    all_cases: list[dict],
    horizon_months: int = 24,
) -> dict:
    """
    業種内の全案件スコアを月次集計 → TimesFM で業種トレンドを予測する。

    Args:
        industry:      業種名（industry_sub）
        all_cases:     load_all_cases() の全案件リスト
        horizon_months: 予測月数

    Returns:
        {
            "months_history":  list[str],    # 過去月ラベル (YYYY-MM)
            "avg_score_hist":  list[float],  # 月次平均スコア
            "months_forecast": list[str],    # 予測月ラベル
            "avg_score_fore":  list[float],  # 予測スコア
            "risk_signal":     str,          # "positive" / "neutral" / "negative"
            "method":          str,
        }
    """
    import datetime

    # 対象業種の案件フィルタ
    industry_cases = [
        c for c in all_cases
        if (c.get("industry_sub") or "").startswith(industry[:4])
        and c.get("score") is not None
        and c.get("timestamp")
    ]
    if not industry_cases:
        return {"error": f"業種「{industry}」の案件データがありません"}

    # 月次集計
    monthly: dict[str, list[float]] = {}
    for c in industry_cases:
        try:
            dt = datetime.datetime.fromisoformat(c["timestamp"])
            key = dt.strftime("%Y-%m")
            monthly.setdefault(key, []).append(float(c["score"]))
        except Exception:
            continue

    sorted_months = sorted(monthly.keys())
    avg_scores = [float(np.mean(monthly[m])) for m in sorted_months]

    method = "linear_extrapolation"
    forecast_scores: list[float] = []

    if TIMESFM_AVAILABLE and len(avg_scores) >= 4:
        raw = _timesfm_point_forecast(avg_scores, horizon_months)
        if len(raw) == horizon_months:
            forecast_scores = raw.tolist()
            method = "timesfm"

    if not forecast_scores:
        slope = (avg_scores[-1] - avg_scores[0]) / max(len(avg_scores) - 1, 1) if len(avg_scores) >= 2 else 0
        forecast_scores = [avg_scores[-1] + slope * (i + 1) for i in range(horizon_months)]

    # 予測月ラベル生成
    if sorted_months:
        last_dt = datetime.datetime.strptime(sorted_months[-1], "%Y-%m")
        forecast_months = [
            (last_dt + datetime.timedelta(days=31 * (i + 1))).strftime("%Y-%m")
            for i in range(horizon_months)
        ]
    else:
        forecast_months = [f"M+{i+1}" for i in range(horizon_months)]

    # リスクシグナル
    if forecast_scores[-1] > avg_scores[-1] + 2:
        risk_signal = "positive"
    elif forecast_scores[-1] < avg_scores[-1] - 2:
        risk_signal = "negative"
    else:
        risk_signal = "neutral"

    return {
        "months_history":  sorted_months,
        "avg_score_hist":  avg_scores,
        "months_forecast": forecast_months,
        "avg_score_fore":  forecast_scores,
        "risk_signal":     risk_signal,
        "method":          method,
    }


# ── ユースケース 4: 成約金利の将来予測 ────────────────────────────────────────

def forecast_final_rate(
    all_cases: list[dict],
    industry: str = "",
    horizon_months: int = 6,
) -> dict:
    """
    成約案件の `final_rate` 時系列から将来の適正金利帯を予測する。

    Args:
        all_cases:      load_all_cases() の全案件
        industry:       業種フィルタ（空文字なら全業種）
        horizon_months: 予測月数

    Returns:
        {
            "rate_history":  list[float],  # 過去成約金利
            "rate_forecast": float,        # 将来中央予測金利
            "band_low":      float,        # 予測下限
            "band_high":     float,        # 予測上限
            "method":        str,
        }
    """
    import datetime

    def _get_rate(c: dict) -> Optional[float]:
        data = c.get("data") or {}
        if isinstance(data, str):
            import json
            try:
                data = json.loads(data)
            except Exception:
                return None
        return data.get("final_rate") or data.get("pricing", {}).get("final_rate")

    filtered = [
        c for c in all_cases
        if c.get("final_status") == "成約"
        and _get_rate(c) is not None
        and (not industry or (c.get("industry_sub") or "").startswith(industry[:4]))
    ]

    rates = []
    for c in sorted(filtered, key=lambda x: x.get("timestamp", "")):
        r = _get_rate(c)
        if r is not None:
            rates.append(float(r))

    if not rates:
        return {"error": "成約案件の金利データがありません"}

    method = "mean_extrapolation"
    forecast_vals: list[float] = []

    if TIMESFM_AVAILABLE and len(rates) >= 4:
        raw = _timesfm_point_forecast(rates, horizon_months)
        if len(raw) == horizon_months:
            forecast_vals = raw.tolist()
            method = "timesfm"

    if not forecast_vals:
        # 直近3件の平均でフォールバック
        mean_rate = float(np.mean(rates[-3:]))
        forecast_vals = [mean_rate] * horizon_months

    sigma = float(np.std(rates)) if len(rates) > 1 else 0.1
    rate_forecast = forecast_vals[-1]

    return {
        "rate_history":  rates,
        "rate_forecast": round(rate_forecast, 3),
        "band_low":      round(max(0.0, rate_forecast - sigma), 3),
        "band_high":     round(rate_forecast + sigma, 3),
        "method":        method,
        "horizon_forecast": [round(v, 3) for v in forecast_vals],
    }


# ── ユースケース 5: 基準金利マスタ時系列予測 ────────────────────────────────────

TERM_COLS = ["r_2y", "r_3y", "r_4y", "r_5y", "r_6y", "r_7y", "r_8y", "r_9y", "r_over9y"]
TERM_LABELS = {
    "r_2y": "2年以内", "r_3y": "3年以内", "r_4y": "4年以内",
    "r_5y": "5年以内", "r_6y": "6年以内", "r_7y": "7年以内",
    "r_8y": "8年以内", "r_9y": "9年以内", "r_over9y": "9年超",
}


def forecast_base_rate(
    term_col: str = "r_5y",
    horizon_months: int = 6,
) -> dict:
    """
    基準金利マスタ（base_rate_master）の指定期間列時系列から
    将来の基準金利帯を予測する。

    Args:
        term_col:       列名 ("r_2y"〜"r_over9y")
        horizon_months: 予測月数

    Returns:
        {
            "term_col":        str,
            "term_label":      str,
            "months_history":  list[str],   # "YYYY-MM" 古い順
            "rate_history":    list[float],
            "months_forecast": list[str],
            "rate_forecast":   float,       # 最終月の中央予測
            "band_low":        float,
            "band_high":       float,
            "horizon_forecast": list[float],
            "method":          str,
        }
    """
    import datetime
    from base_rate_master import list_base_rates

    if term_col not in TERM_COLS:
        return {"error": f"無効な期間列: {term_col}"}

    records = list_base_rates(limit=60)
    # 古い順にソート
    records = sorted(records, key=lambda r: r["month"])

    months_history: list[str] = []
    rate_history: list[float] = []
    for rec in records:
        v = rec.get(term_col)
        if v is not None:
            months_history.append(rec["month"])
            rate_history.append(float(v))

    if len(rate_history) < 2:
        return {"error": "基準金利データが不足しています（初期データ投入をお試しください）"}

    # 予測月ラベル生成
    last_month = months_history[-1]
    y, m = int(last_month[:4]), int(last_month[5:7])
    months_forecast: list[str] = []
    for i in range(1, horizon_months + 1):
        m2 = m + i
        y2 = y + (m2 - 1) // 12
        m2 = (m2 - 1) % 12 + 1
        months_forecast.append(f"{y2:04d}-{m2:02d}")

    # 予測実行
    method = "linear_extrapolation"
    forecast_vals: list[float] = []

    if TIMESFM_AVAILABLE and len(rate_history) >= 4:
        raw = _timesfm_point_forecast(rate_history, horizon_months)
        if len(raw) == horizon_months:
            forecast_vals = [max(0.0, float(v)) for v in raw]
            method = "timesfm"

    if not forecast_vals:
        # 線形回帰フォールバック
        n = len(rate_history)
        x = np.arange(n, dtype=float)
        coeffs = np.polyfit(x, rate_history, 1)
        for i in range(1, horizon_months + 1):
            val = float(np.polyval(coeffs, n - 1 + i))
            forecast_vals.append(max(0.0, val))
        method = "linear_extrapolation"

    sigma = float(np.std(rate_history[-12:])) if len(rate_history) >= 3 else 0.1
    rate_forecast = forecast_vals[-1]

    return {
        "term_col":         term_col,
        "term_label":       TERM_LABELS.get(term_col, term_col),
        "months_history":   months_history,
        "rate_history":     [round(v, 3) for v in rate_history],
        "months_forecast":  months_forecast,
        "rate_forecast":    round(rate_forecast, 3),
        "band_low":         round(max(0.0, rate_forecast - sigma), 3),
        "band_high":        round(rate_forecast + sigma, 3),
        "horizon_forecast": [round(v, 3) for v in forecast_vals],
        "method":           method,
    }


def forecast_base_rate_all(horizon_months: int = 6) -> dict:
    """全9期間列を一括予測して返す。"""
    results = {}
    for col in TERM_COLS:
        res = forecast_base_rate(col, horizon_months)
        if "error" not in res:
            results[col] = res
    return {
        "term_cols":    TERM_COLS,
        "term_labels":  TERM_LABELS,
        "forecasts":    results,
        "horizon_months": horizon_months,
    }
