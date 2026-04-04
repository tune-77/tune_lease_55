# -*- coding: utf-8 -*-
"""
backend.py
==========
FastAPI バックエンド: 3期財務データから TimesFM による12ヶ月予測を提供する。

起動方法:
    uvicorn backend:app --reload --port 8000

エンドポイント:
    GET  /health   — サービス稼働確認
    POST /forecast — 年次3点 → 月次補完 → TimesFM 12ヶ月予測
"""
from __future__ import annotations

import math
from datetime import datetime, date
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ── TimesFM エンジン（既存モジュールを再利用） ────────────────────────────────
# timesfm_engine.py が同一ディレクトリにある前提
from timesfm_engine import _timesfm_point_forecast, TIMESFM_AVAILABLE

# ── アプリ初期化 ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="リース審査 TimesFM API",
    description="3期財務データから月次補完 + 12ヶ月予測を返す",
    version="1.0.0",
)

# CORS: Streamlit (localhost:8501) からのリクエストを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── 業種別季節性インデックス ──────────────────────────────────────────────────
# 12ヶ月分の係数（合計 = 12.0）。各値は "月平均の何倍か" を表す。
# 例: 建設業の係数 1.8（3月）= その月は年間平均の1.8倍の売上が集中する
SEASONAL_INDICES: dict[str, list[float]] = {
    "建設業":       [0.6, 0.7, 1.8, 0.8, 0.9, 0.9, 0.8, 0.9, 1.0, 0.9, 1.0, 1.7],  # 3月・12月に年度末集中
    "小売業":       [0.9, 0.8, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 0.9, 1.0, 1.1, 1.6],  # 12月（年末商戦）集中
    "製造業":       [0.9, 0.9, 1.1, 1.0, 1.0, 1.0, 1.0, 0.9, 1.1, 1.0, 1.0, 1.1],  # 年度末・下期に若干集中
    "卸売業":       [0.9, 0.9, 1.2, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.1],  # 製造業に準じる
    "医療・福祉":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 安定
    "飲食・宿泊業": [0.8, 0.8, 1.0, 1.0, 1.1, 1.1, 1.3, 1.2, 1.0, 1.0, 0.9, 0.8],  # 夏季（7-8月）集中
    "サービス業":   [0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1],  # ほぼ平坦
    "不動産業":     [0.8, 0.9, 1.4, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 1.3],  # 3月（引越し期）集中
    "情報通信業":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 安定
    "運輸・物流":   [0.9, 0.9, 1.1, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2],  # 年末・3月にやや集中
}
# 未登録業種のデフォルト（均等分布）
_DEFAULT_SEASONAL = [1.0] * 12


# ── スキーマ定義 ──────────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    """予測リクエスト"""
    sales:      list[float]  # 売上高（千円）: [3期前, 2期前, 直近期]
    profit:     list[float]  # 営業利益（千円）: [3期前, 2期前, 直近期]
    net_assets: list[float]  # 純資産（千円）: [3期前, 2期前, 直近期]
    industry:   str = "サービス業"  # 業種カテゴリ

    @field_validator("sales", "profit", "net_assets")
    @classmethod
    def must_be_three_values(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("3期分（3要素）のリストを指定してください")
        return v


class ForecastResponse(BaseModel):
    """予測レスポンス"""
    months_history:      list[str]    # 過去36ヶ月のラベル（YYYY-MM）
    sales_history:       list[float]  # 補完後の月次売上（千円）
    profit_history:      list[float]  # 補完後の月次利益（千円）
    net_assets_history:  list[float]  # 補完後の月次純資産（千円）
    months_forecast:     list[str]    # 予測12ヶ月のラベル（YYYY-MM）
    sales_forecast:      list[float]  # 予測売上（千円）
    profit_forecast:     list[float]  # 予測利益（千円）
    net_assets_forecast: list[float]  # 予測純資産（千円）
    timesfm_available:   bool         # TimesFM が利用可能かどうか


# ── ヘルパー関数 ──────────────────────────────────────────────────────────────

def _annual_to_monthly(annual_values: list[float], industry: str) -> list[float]:
    """
    年次3点を36ヶ月の擬似月次データに変換する。

    アルゴリズム:
        1. 業種別の季節性インデックス（12要素、合計=12.0）を取得する
        2. 各年の年間値を月平均（÷12）に変換し、季節係数を掛ける
        3. 3年分を結合して36要素のリストを返す

    Args:
        annual_values: [3期前, 2期前, 直近期] の年次値（千円）
        industry:      業種カテゴリ文字列

    Returns:
        36要素の月次値リスト（千円/月）
    """
    idx = SEASONAL_INDICES.get(industry, _DEFAULT_SEASONAL)
    monthly: list[float] = []
    for annual in annual_values:
        # 月平均に変換してから季節係数を乗じる
        monthly_mean = annual / 12.0
        for season in idx:
            monthly.append(monthly_mean * season)
    return monthly  # 36要素


def _gbm_forecast(history: list[float], horizon: int = 12) -> list[float]:
    """
    TimesFM が利用不可のときのフォールバック。
    GBM（幾何ブラウン運動）の期待値パスで単純予測する。

    Args:
        history: 過去の月次値リスト
        horizon: 予測月数（デフォルト12）

    Returns:
        horizon 要素の予測値リスト
    """
    if not history or history[-1] <= 0:
        return [0.0] * horizon

    # 対数収益率からドリフト・ボラティリティを推定
    arr = np.array(history, dtype=float)
    arr = np.clip(arr, 1e-6, None)
    log_r = np.diff(np.log(arr))
    mu = float(np.mean(log_r)) if len(log_r) > 0 else 0.01
    S0 = history[-1]
    dt = 1.0 / 12

    # 期待値パス（ランダム要素なし）
    forecast = [S0 * math.exp(mu * (t + 1) * dt) for t in range(horizon)]
    return forecast


def _run_forecast(monthly_history: list[float], horizon: int = 12) -> list[float]:
    """
    TimesFM または GBM フォールバックで予測値を返す。

    Args:
        monthly_history: 36ヶ月の月次値リスト
        horizon:         予測月数

    Returns:
        horizon 要素の予測値リスト
    """
    if TIMESFM_AVAILABLE and len(monthly_history) >= 6:
        # TimesFM による予測（_timesfm_point_forecast は timesfm_engine.py の内部関数）
        result = _timesfm_point_forecast(monthly_history, horizon)
        if len(result) == horizon:
            return [float(v) for v in result]

    # フォールバック: GBM 期待値パス
    return _gbm_forecast(monthly_history, horizon)


def _make_month_labels(start_year: int, start_month: int, count: int) -> list[str]:
    """
    YYYY-MM 形式の月ラベルリストを生成する。

    Args:
        start_year:  開始年
        start_month: 開始月（1-12）
        count:       生成する月数

    Returns:
        ["YYYY-MM", ...] の形式で count 要素のリスト
    """
    labels = []
    y, m = start_year, start_month
    for _ in range(count):
        labels.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return labels


# ── エンドポイント ────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """サービス稼働確認 + TimesFM インストール状態を返す"""
    return {"status": "ok", "timesfm": TIMESFM_AVAILABLE}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    """
    3期分の年次財務データから12ヶ月予測を生成して返す。

    処理フロー:
        1. 年次3点 → 業種別季節性を使って36ヶ月の月次データに補完
        2. 36ヶ月を TimesFM（または GBM）に渡して12ヶ月先を予測
        3. 過去36ヶ月・予測12ヶ月のラベルと値をまとめて返す
    """
    # ── 月次補完 ─────────────────────────────────────────────────────────────
    sales_hist      = _annual_to_monthly(req.sales,      req.industry)
    profit_hist     = _annual_to_monthly(req.profit,     req.industry)
    net_assets_hist = _annual_to_monthly(req.net_assets, req.industry)

    # ── 月ラベル生成 ─────────────────────────────────────────────────────────
    # 直近期を「今年度末」とみなし、36ヶ月前を起点にする
    today = date.today()
    # 直近期の末月 = 今年の3月（日本標準の3月決算を仮定）
    # ※ 汎用性のため「今月を基準に36ヶ月前から開始」とする
    hist_start_year  = today.year - 3
    hist_start_month = today.month
    history_labels   = _make_month_labels(hist_start_year, hist_start_month, 36)
    forecast_labels  = _make_month_labels(today.year, today.month, 12)

    # ── TimesFM / GBM 予測 ───────────────────────────────────────────────────
    HORIZON = 12
    sales_fore      = _run_forecast(sales_hist,      HORIZON)
    profit_fore     = _run_forecast(profit_hist,     HORIZON)
    net_assets_fore = _run_forecast(net_assets_hist, HORIZON)

    return ForecastResponse(
        months_history      = history_labels,
        sales_history       = sales_hist,
        profit_history      = profit_hist,
        net_assets_history  = net_assets_hist,
        months_forecast     = forecast_labels,
        sales_forecast      = sales_fore,
        profit_forecast     = profit_fore,
        net_assets_forecast = net_assets_fore,
        timesfm_available   = TIMESFM_AVAILABLE,
    )
