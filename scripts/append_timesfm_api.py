import os

# The code to append
code_to_append = """

# =============================================================================
# 高度分析 API (Phase 15.2: TimesFM / 3期財務分析)
# =============================================================================

from pydantic import field_validator

# ── 3期財務分析 (financial) 関連 ────────────────────────────────────────────────

SEASONAL_INDICES = {
    "建設業":       [0.6, 0.7, 1.8, 0.8, 0.9, 0.9, 0.8, 0.9, 1.0, 0.9, 1.0, 1.7],
    "小売業":       [0.9, 0.8, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 0.9, 1.0, 1.1, 1.6],
    "製造業":       [0.9, 0.9, 1.1, 1.0, 1.0, 1.0, 1.0, 0.9, 1.1, 1.0, 1.0, 1.1],
    "卸売業":       [0.9, 0.9, 1.2, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.1],
    "医療・福祉":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "飲食・宿泊業": [0.8, 0.8, 1.0, 1.0, 1.1, 1.1, 1.3, 1.2, 1.0, 1.0, 0.9, 0.8],
    "サービス業":   [0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1],
    "不動産業":     [0.8, 0.9, 1.4, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 1.3],
    "情報通信業":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "運輸・物流":   [0.9, 0.9, 1.1, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2],
}
_DEFAULT_SEASONAL = [1.0] * 12

class ForecastRequest(BaseModel):
    sales:      list[float]
    profit:     list[float]
    net_assets: list[float]
    industry:   str = "サービス業"

    @field_validator("sales", "profit", "net_assets")
    @classmethod
    def must_be_three_values(cls, v: list[float]) -> list[float]:
        if len(v) != 3: raise ValueError("3 elements required")
        return v

class ForecastResponse(BaseModel):
    months_history:      list[str]
    sales_history:       list[float]
    profit_history:      list[float]
    net_assets_history:  list[float]
    months_forecast:     list[str]
    sales_forecast:      list[float]
    profit_forecast:     list[float]
    net_assets_forecast: list[float]
    timesfm_available:   bool

def _annual_to_monthly(annual_values: list[float], industry: str) -> list[float]:
    idx = SEASONAL_INDICES.get(industry, _DEFAULT_SEASONAL)
    monthly = []
    for annual in annual_values:
        monthly_mean = annual / 12.0
        for season in idx:
            monthly.append(monthly_mean * season)
    return monthly

def _run_forecast(monthly_history: list[float], horizon: int = 12) -> list[float]:
    import math
    import numpy as np
    from timesfm_engine import _timesfm_point_forecast, TIMESFM_AVAILABLE
    if TIMESFM_AVAILABLE and len(monthly_history) >= 6:
        result = _timesfm_point_forecast(monthly_history, horizon)
        if len(result) == horizon:
            return [float(v) for v in result]
    # Fallback to GBM
    if not monthly_history or monthly_history[-1] <= 0: return [0.0] * horizon
    arr = np.clip(np.array(monthly_history, dtype=float), 1e-6, None)
    log_r = np.diff(np.log(arr))
    mu = float(np.mean(log_r)) if len(log_r) > 0 else 0.01
    S0 = monthly_history[-1]
    dt = 1.0 / 12
    return [S0 * math.exp(mu * (t + 1) * dt) for t in range(horizon)]

def _make_month_labels(start_year: int, start_month: int, count: int) -> list[str]:
    labels = []
    y, m = start_year, start_month
    for _ in range(count):
        labels.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12: m = 1; y += 1
    return labels

@app.post("/api/forecast", response_model=ForecastResponse)
def api_forecast(req: ForecastRequest):
    from timesfm_engine import TIMESFM_AVAILABLE
    from datetime import date
    sales_hist = _annual_to_monthly(req.sales, req.industry)
    profit_hist = _annual_to_monthly(req.profit, req.industry)
    net_assets_hist = _annual_to_monthly(req.net_assets, req.industry)

    today = date.today()
    history_labels = _make_month_labels(today.year - 3, today.month, 36)
    forecast_labels = _make_month_labels(today.year, today.month, 12)

    return ForecastResponse(
        months_history=history_labels,
        sales_history=sales_hist,
        profit_history=profit_hist,
        net_assets_history=net_assets_hist,
        months_forecast=forecast_labels,
        sales_forecast=_run_forecast(sales_hist, 12),
        profit_forecast=_run_forecast(profit_hist, 12),
        net_assets_forecast=_run_forecast(net_assets_hist, 12),
        timesfm_available=TIMESFM_AVAILABLE,
    )

# ── TimesFM 時系列予測 (timesfm) 関連 ──────────────────────────────────────────

class TfmCompanyScoreRequest(BaseModel):
    company_name: str
    horizon_months: int = 12

@app.post("/api/timesfm/company_score")
def api_tfm_company_score(req: TfmCompanyScoreRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_company_score
    cases = load_all_cases()
    company_cases = sorted(
        [c for c in cases if (c.get("company_name") or c.get("inputs", {}).get("company_name", "")) == req.company_name],
        key=lambda x: x.get("timestamp", ""),
    )
    if not company_cases:
        raise HTTPException(status_code=404, detail="Company not found")
    result = forecast_company_score(company_cases, horizon_months=req.horizon_months)
    import numpy as np
    # Convert numpy values to standard python floats for JSON serialization
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [float(x) if not isinstance(x, str) else x for x in v]
    return result

class TfmIndustryTrendRequest(BaseModel):
    industry: str
    horizon_months: int = 24

@app.post("/api/timesfm/industry_trend")
def api_tfm_industry_trend(req: TfmIndustryTrendRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_industry_trend
    cases = load_all_cases()
    result = forecast_industry_trend(req.industry, cases, horizon_months=req.horizon_months)
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [float(x) if not isinstance(x, str) else x for x in v]
    return result

class TfmFinalRateRequest(BaseModel):
    industry: str = ""
    horizon_months: int = 6

@app.post("/api/timesfm/final_rate")
def api_tfm_final_rate(req: TfmFinalRateRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_final_rate
    cases = load_all_cases()
    result = forecast_final_rate(cases, industry=req.industry, horizon_months=req.horizon_months)
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [float(x) if not isinstance(x, str) else x for x in v]
    return result

class TfmFinancialPathsRequest(BaseModel):
    company_name: str
    n_periods: int = 12

@app.post("/api/timesfm/financial_paths")
def api_tfm_financial_paths(req: TfmFinancialPathsRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_financial_paths, TIMESFM_AVAILABLE
    import numpy as np
    cases = load_all_cases()
    company_cases = sorted(
        [c for c in cases if (c.get("company_name") or c.get("inputs", {}).get("company_name", "")) == req.company_name],
        key=lambda x: x.get("timestamp", ""),
    )
    revenues = []
    for c in company_cases:
        inp = c.get("inputs", {})
        if isinstance(inp, str):
            import json
            try: inp = json.loads(inp)
            except: inp = {}
        v = inp.get("nenshu", inp.get("revenue"))
        if v:
            try: revenues.append(float(v))
            except: pass
            
    if not revenues: revenues = [10_000_000.0]
    
    # 200 paths, downsampled to 50 for frontend drawing to save bandwidth
    gbm_paths = forecast_financial_paths(revenues, req.n_periods, n_paths=200)[:50].tolist()
    gbm_median = np.median(forecast_financial_paths(revenues, req.n_periods, n_paths=200), axis=0).tolist()
    
    tfm_paths = []
    tfm_median = []
    if TIMESFM_AVAILABLE:
        raw_tfm = forecast_financial_paths(revenues, req.n_periods, n_paths=200)
        tfm_paths = raw_tfm[:50].tolist()
        tfm_median = np.median(raw_tfm, axis=0).tolist()
        
    return {
        "gbm_paths": gbm_paths,
        "gbm_median": gbm_median,
        "tfm_paths": tfm_paths,
        "tfm_median": tfm_median,
        "revenues": revenues,
        "timesfm_available": TIMESFM_AVAILABLE
    }
"""

with open('/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/api/main.py', 'a') as f:
    f.write(code_to_append)
print("Appended API endpoints to main.py")
