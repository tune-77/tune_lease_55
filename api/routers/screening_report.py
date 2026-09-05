"""
api/routers/screening_report.py

審査分析画面（審査結果 result）から、lieflat-charts skill 骨格の
単体HTMLレポートをエクスポートするエンドポイント。

Endpoints:
  POST /api/screening/report – render_screening_report
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api.reporting.screening_html_report import render_screening_report_html

router = APIRouter(tags=["screening-report"])


class ScreeningReportRequest(BaseModel):
    result: dict[str, Any]
    case_label: str = ""


@router.post("/api/screening/report", response_class=HTMLResponse)
def render_screening_report(payload: ScreeningReportRequest) -> HTMLResponse:
    """審査結果(result)をそのまま受け取り、単体HTMLレポートを返す。

    result は審査分析画面がスコアリング呼び出しで既に取得している
    オブジェクトをそのまま渡す想定。ここでは新規のスコア計算・DB参照は行わない。
    """
    html = render_screening_report_html(payload.result, case_label=payload.case_label)
    return HTMLResponse(content=html)
