"""
TimesFM 時系列予測 API ルーター (REV-234)
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/timesfm", tags=["timesfm"])


class TfmCompanyScoreRequest(BaseModel):
    company_name: str
    horizon_months: int = 12


@router.post("/company_score")
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
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [float(x) if not isinstance(x, str) else x for x in v]
    return result


class TfmIndustryTrendRequest(BaseModel):
    industry: str
    horizon_months: int = 24


@router.post("/industry_trend")
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


@router.post("/final_rate")
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
    current_revenue: Optional[float] = None
    current_revenue_unit: str = "thousand_yen"


@router.post("/financial_paths")
def api_tfm_financial_paths(req: TfmFinancialPathsRequest):
    import json
    import numpy as np
    from data_cases import load_all_cases
    from timesfm_engine import forecast_financial_paths, TIMESFM_AVAILABLE

    def _to_thousand_yen(value, unit: str = "thousand_yen") -> Optional[float]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v) or v <= 0:
            return None
        normalized_unit = (unit or "thousand_yen").lower()
        if normalized_unit in {"million_yen", "million", "m_yen"}:
            return v * 1000.0
        return v

    cases = load_all_cases()
    company_cases = sorted(
        [c for c in cases if (c.get("company_name") or c.get("inputs", {}).get("company_name", "")) == req.company_name],
        key=lambda x: x.get("timestamp", ""),
    )
    revenues: list[float] = []
    for c in company_cases:
        inp = c.get("inputs", {})
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except Exception:
                inp = {}
        v = inp.get("nenshu", inp.get("revenue"))
        if v:
            parsed = _to_thousand_yen(v)
            if parsed is not None:
                revenues.append(parsed)

    current_revenue = _to_thousand_yen(req.current_revenue, req.current_revenue_unit)
    if current_revenue is not None:
        if not revenues or abs(revenues[-1] - current_revenue) > max(1.0, current_revenue * 0.001):
            revenues.append(current_revenue)

    if not revenues:
        revenues = [200_000.0]

    raw_gbm = forecast_financial_paths(revenues, req.n_periods, n_paths=200)
    gbm_paths = raw_gbm[:50].tolist()
    gbm_median = np.median(raw_gbm, axis=0).tolist()

    tfm_paths: list = []
    tfm_median: list = []
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
        "timesfm_available": TIMESFM_AVAILABLE,
    }


class TfmBaseRateRequest(BaseModel):
    term_col: str = "r_5y"
    horizon_months: int = 6


class TfmBaseRateAllRequest(BaseModel):
    horizon_months: int = 6


@router.post("/base_rate")
def api_tfm_base_rate(req: TfmBaseRateRequest):
    from timesfm_engine import forecast_base_rate
    try:
        return forecast_base_rate(req.term_col, req.horizon_months)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/base_rate_all")
def api_tfm_base_rate_all(req: TfmBaseRateAllRequest):
    from timesfm_engine import forecast_base_rate_all
    try:
        return forecast_base_rate_all(req.horizon_months)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
