"""Read-only lease-news endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.knowledge.news_classifier import (
    build_classified_news_summary_from_vault,
    load_latest_classified_news_summary,
)
from api.knowledge.news_vertex_summary import build_vertex_assisted_news_trend_summary
from api.lease_news_presenters import (
    lease_news_actions_to_dict,
    lease_news_brief_to_dict,
    lease_news_focus_to_dict,
)
from lease_news_digest import (
    build_daily_news_digest,
    build_lease_news_brief,
    find_vault,
    get_latest_lease_news_actions,
    get_latest_lease_news_focus,
)

router = APIRouter(prefix="/api/lease-news", tags=["lease-news"])


@router.get("/focus")
def get_lease_news_focus_api():
    """ホーム画面とAICHATで共通利用する最新ニュースの注目論点を返す。"""
    try:
        return lease_news_focus_to_dict(get_latest_lease_news_focus())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/brief")
def get_lease_news_brief_api(prefecture: str = "", industry: str = ""):
    """AICHATとホームで共通利用する、全国+地域のニュースブリーフを返す。"""
    try:
        return lease_news_brief_to_dict(build_lease_news_brief(prefecture=prefecture, industry=industry))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/actions")
def get_lease_news_actions_api():
    """日次ニュースを審査アクションへ変換した一覧を返す。"""
    try:
        return lease_news_actions_to_dict(get_latest_lease_news_actions())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/daily-digest")
def get_lease_news_daily_digest_api(limit: int = 3):
    """Obsidianの日次ニュースを、対話室の朝報向けに短く返す。"""
    try:
        return build_daily_news_digest(limit=max(1, min(int(limit), 5)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/classified-summary")
def get_lease_news_classified_summary_api(limit: int = 30, days: int = 14, refresh: bool = False):
    """ニュースを業種別・社会情勢・金融情報の軸で束ね、審査示唆つきで返す。"""
    try:
        summary = build_classified_news_summary_from_vault(
            find_vault(),
            limit=max(1, min(int(limit), 80)),
            days=max(1, min(int(days), 60)),
        )
        if summary.get("available") or refresh:
            return summary
        latest = load_latest_classified_news_summary()
        return latest if latest.get("available") else summary
    except Exception as exc:
        if refresh:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latest = load_latest_classified_news_summary()
        if latest.get("available"):
            return latest
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/trend-summary")
def get_lease_news_trend_summary_api(
    limit: int = 30,
    days: int = 14,
    refresh: bool = False,
    use_vertex: bool = True,
):
    """分類済みニュースから、Vertex補助つきの傾向・要約・注意点を返す。"""
    try:
        summary = build_classified_news_summary_from_vault(
            find_vault(),
            limit=max(1, min(int(limit), 80)),
            days=max(1, min(int(days), 60)),
        )
        if not summary.get("available") and not refresh:
            latest = load_latest_classified_news_summary()
            if latest.get("available"):
                summary = latest
        return build_vertex_assisted_news_trend_summary(summary, use_vertex=use_vertex)
    except Exception as exc:
        if refresh:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return build_vertex_assisted_news_trend_summary(
            load_latest_classified_news_summary(),
            use_vertex=use_vertex,
        )
