"""紫苑タスク・挨拶ルーター (REV-234 Phase3)"""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["shion"])

class ShionTaskCreateRequest(BaseModel):
    title: str
    due_at: str = ""
    note: str = ""
    source: str = "chat"
    reminder: bool = False
    tags: list[str] = Field(default_factory=list)


class ShionTaskUpdateRequest(BaseModel):
    title: Optional[str] = None
    due_at: Optional[str] = None
    note: Optional[str] = None
    source: Optional[str] = None
    reminder: Optional[bool] = None
    tags: Optional[list[str]] = None


class ShionTaskStatusRequest(BaseModel):
    status: Literal["open", "done", "cancelled"]


@router.get("/api/shion/daily-greeting")
def get_shion_daily_greeting():
    """紫苑コンシェルジュの毎日変わる一言挨拶を返す。外部通信なしの安定版。"""
    try:
        import datetime as _dt
        from zoneinfo import ZoneInfo as _dt_zoneinfo

        now = _dt.datetime.now(_dt_zoneinfo("Asia/Tokyo"))
        today = now.date()
        opening = _daily_greeting_opening(now)
        git_summary = _daily_greeting_git_summary()
        yesterday_note = _daily_greeting_yesterday_note()
        anniversary = _daily_greeting_anniversary()
        news_thought = _daily_greeting_news_thought()
        yesterday = git_summary or yesterday_note or "昨日の作業ログは薄めです。今日は入口で状況を整理してから始めます。"
        suggestion = "まず紫苑の予測を見て、審査入力・外部調査・チャットのどこから始めるかを選びましょう。"
        if opening["time_band"] == "evening":
            suggestion = "今日は広げすぎず、審査入力・Research・チャットのうち、残す判断材料を一つ回収しましょう。"
        elif opening["time_band"] == "late_night":
            suggestion = "今は作業を増やすより、明日すぐ再開できる入口を一つだけ決めましょう。"
        return {
            "date": today.isoformat(),
            "opening": opening["text"],
            "time_band": opening["time_band"],
            "time_note": opening["mood"],
            "yesterday": yesterday,
            "anniversary": anniversary,
            "thought": news_thought,
            "suggestion": suggestion,
            "source": {
                "git": bool(git_summary),
                "yesterday_daily": bool(yesterday_note),
                "anniversary": "data/shion_anniversaries.json",
                "news": "data/lease_news_actions_latest.json",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/shion/tasks")
def get_shion_tasks(status: Literal["open", "done", "cancelled", "all"] = "open", limit: int = 50) -> dict:
    """紫苑のタスク台帳を返す。記憶・判断資産とは分離した読み取り専用一覧。"""
    from api.shion_tasks import list_tasks

    tasks = list_tasks(status=status, limit=limit)
    return {
        "count": len(tasks),
        "status_filter": status,
        "ledger": "data/shion_tasks.jsonl",
        "separation_policy": "tasks_are_not_memory_or_judgment_assets",
        "tasks": tasks,
    }


@router.get("/api/shion/proactive-alert")
def get_shion_proactive_alert() -> dict:
    """紫苑からの能動的な割り込み発言があるか判定する。エラーログの急増だけを見る軽量版。"""
    from api.shion_proactive_alert import check_shion_proactive_alerts

    return check_shion_proactive_alerts()


@router.get("/api/shion/latent-need-alert")
def get_shion_latent_need_alert(prefecture: str = "", industry: str = "") -> dict:
    """案件の業種・地域に基づき、紫苑から能動的に伝えられる気づき（業界動向・審査上の注意点）があるか判定する。"""
    from api.shion_proactive_alert import check_shion_latent_need_alert

    return check_shion_latent_need_alert(prefecture=prefecture, industry=industry)


@router.get("/api/shion/memory-lanes")
def get_shion_memory_lanes(
    include_private: bool = False,
    include_sensitive_personal: bool = False,
    sample_limit: int = 5,
) -> dict:
    """普通の記憶エージェントとしての記憶棚を分離して返す。

    ここではプロンプト注入や判断資産昇格はしない。見える化だけ。
    """
    from api.shion_memory_lanes import build_memory_lanes

    return build_memory_lanes(
        include_private=include_private,
        include_sensitive_personal=include_sensitive_personal,
        sample_limit=max(1, min(int(sample_limit or 5), 20)),
    )


@router.post("/api/shion/tasks")
def post_shion_task(req: ShionTaskCreateRequest, background_tasks: BackgroundTasks) -> dict:
    """紫苑タスクを追記型台帳へ追加する。通知は別系統で、ここでは管理だけ行う。"""
    from api.shion_tasks import create_task

    try:
        task = create_task(
            title=req.title,
            due_at=req.due_at,
            note=req.note,
            source=req.source,
            reminder=req.reminder,
            tags=req.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_task_created",
        surface="shion_tasks",
        payload={**task, "schema_version": 1},
    )
    return {"status": "ok", "task": task}


@router.patch("/api/shion/tasks/{task_id}")
def patch_shion_task(task_id: str, req: ShionTaskUpdateRequest, background_tasks: BackgroundTasks) -> dict:
    from api.shion_tasks import update_task

    try:
        task = update_task(
            task_id,
            title=req.title,
            due_at=req.due_at,
            note=req.note,
            source=req.source,
            reminder=req.reminder,
            tags=req.tags,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_task_updated",
        surface="shion_tasks",
        payload={**task, "schema_version": 1},
    )
    return {"status": "ok", "task": task}


@router.post("/api/shion/tasks/{task_id}/status")
def post_shion_task_status(task_id: str, req: ShionTaskStatusRequest, background_tasks: BackgroundTasks) -> dict:
    from api.shion_tasks import set_task_status

    try:
        task = set_task_status(task_id, req.status)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_task_status_changed",
        surface="shion_tasks",
        payload={**task, "schema_version": 1},
    )
    return {"status": "ok", "task": task}


def _lease_news_brief_to_dict(brief):
    if not brief or not getattr(brief, "available", False):
        return {"available": False}
    return {
        "available": True,
        "prefecture": getattr(brief, "prefecture", ""),
        "region": getattr(brief, "region", ""),
        "geo_context": getattr(brief, "geo_context", ""),
        "national_headline": getattr(brief, "national_headline", ""),
        "national_focus_lines": list(getattr(brief, "national_focus_lines", ()) or ()),
        "regional_available": getattr(brief, "regional_available", False),
        "regional_title": getattr(brief, "regional_title", ""),
        "regional_summary_lines": list(getattr(brief, "regional_summary_lines", ()) or ()),
        "regional_usage_memo": getattr(brief, "regional_usage_memo", ""),
        "regional_tags": list(getattr(brief, "regional_tags", ()) or ()),
        "regional_source": getattr(brief, "regional_source", ""),
        "opening_line": getattr(brief, "opening_line", ""),
        "question_line": getattr(brief, "question_line", ""),
        "note_date": getattr(brief, "note_date", ""),
        "note_path": getattr(brief, "note_path", ""),
    }

