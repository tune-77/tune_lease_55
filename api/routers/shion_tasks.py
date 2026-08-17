"""紫苑タスク・挨拶ルーター (REV-234 Phase3)"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from api.cloudrun_writeback import record_cloudrun_input_event

router = APIRouter(tags=["shion"])
_REPO_ROOT = Path(__file__).resolve().parents[2]

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


class MemoryReviewRequest(BaseModel):
    decision: Literal["adopted", "revised", "held", "rejected"]
    note: str = ""
    edited_claim: str = ""


class ShionHydeDebugRequest(BaseModel):
    message: str
    industry: str = ""
    asset_name: str = ""
    score_band: str = ""
    known_risk_tags: list[str] = Field(default_factory=list)
    limit: int = 5


class CounterHypothesisCardRequest(BaseModel):
    case_summary: str
    industry: str = ""
    asset_name: str = ""
    score_band: str = ""
    human_discomfort: str = ""
    risk_tags: list[str] = Field(default_factory=list)


def _daily_greeting_read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _daily_greeting_git_summary() -> str:
    try:
        proc = subprocess.run(
            ["git", "log", "--since=yesterday", "--pretty=format:%s", "-4"],
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if lines:
            return " / ".join(lines[:2])
    except Exception:
        pass
    return ""


def _daily_greeting_yesterday_note() -> str:
    try:
        import datetime as _dt

        vault_raw = os.environ.get("OBSIDIAN_VAULT_PATH") or os.environ.get("OBSIDIAN_VAULT") or ""
        if not vault_raw:
            try:
                from lease_news_digest import find_vault

                found = find_vault()
                vault_raw = str(found) if found else ""
            except Exception:
                vault_raw = ""
        if not vault_raw:
            return ""
        yesterday = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
        path = Path(vault_raw) / "Daily" / f"{yesterday}.md"
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [
            re.sub(r"^[#>*\-\s]+", "", line).strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("```")
        ]
        return next((line for line in lines if len(line) >= 18), "")[:120]
    except Exception:
        return ""


def _daily_greeting_anniversary() -> dict:
    try:
        import datetime as _dt

        key = _dt.date.today().strftime("%m-%d")
        data = _daily_greeting_read_json(_REPO_ROOT / "data" / "shion_anniversaries.json")
        item = data.get(key) or {}
        if item:
            return {"date_key": key, **item}
    except Exception:
        pass
    return {
        "date_key": "",
        "name": "小さな兆候を見る日",
        "note": "今日は、数字やニュースの端に出る小さな違和感を拾ってから判断します。",
    }


def _daily_greeting_news_thought() -> str:
    try:
        actions = _daily_greeting_read_json(_REPO_ROOT / "data" / "lease_news_actions_latest.json")
        summary = str(actions.get("summary") or "").strip()
        action_items = actions.get("action_items") or []
        if action_items:
            checks = action_items[0].get("recommended_checks") or []
            if checks:
                return str(checks[0]).strip()[:160]
        if summary:
            return f"ニュースからは「{summary[:80]}」が見えています。今日はこれを審査条件に直結させすぎず、確認観点として扱います。"
    except Exception:
        pass
    return "ニュースはまだ薄めです。今日は外部情報より、前回の作業と手元の案件条件を優先して見ます。"


def _daily_greeting_opening(now) -> dict:
    hour = int(getattr(now, "hour", 12))
    if 5 <= hour < 10:
        return {
            "text": "おはようございます。",
            "mood": "朝なので、昨日の続きと今日の最初の一手を短く整理します。",
            "time_band": "morning",
        }
    if 10 <= hour < 17:
        return {
            "text": "こんにちは。",
            "mood": "日中なので、今すぐ進める作業順に並べます。",
            "time_band": "daytime",
        }
    if 17 <= hour < 22:
        return {
            "text": "こんばんは。",
            "mood": "夕方以降なので、今日の判断材料を回収しながら進めます。",
            "time_band": "evening",
        }
    return {
        "text": "夜更かしですね。",
        "mood": "深い時間なので、無理に広げず、次に残す判断だけ整えます。",
        "time_band": "late_night",
    }


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


@router.get("/api/shion/judgment-prediction-alert")
def get_shion_judgment_prediction_alert(limit: int = 3) -> dict:
    """繰り返し外している前提を、案件を開いた時点で先回りして返す。

    日次レポート（reports/predictive_framework_latest.json）を読むだけで、
    スコアリング・判断資産は変更しない。該当が無ければ has_alert=False を返す。
    """
    from api.shion_proactive_alert import check_judgment_prediction_alert

    return check_judgment_prediction_alert(limit=limit)


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


@router.get("/api/shion/memory-engineering-report")
def get_shion_memory_engineering_report() -> dict:
    """最新のMemory Engineeringレポートを返す（読み取り専用）。"""
    from runtime_paths import get_data_dir

    bundle_root = Path(os.environ.get("CLOUDRUN_BUNDLE_DIR") or (_REPO_ROOT / ".cloudrun_bundle"))
    report_paths = [
        get_data_dir() / "memory_engineering_latest.json",
        bundle_root / "data" / "memory_engineering_latest.json",
        bundle_root / "reports" / "memory_engineering_latest.json",
        _REPO_ROOT / "reports" / "memory_engineering_latest.json",
    ]
    report_path = next((path for path in report_paths if path.exists()), report_paths[-1])
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "generated": False,
            "mode": "read_only_memory_engineering_observation",
            "guardrail": "no_memory_delete_no_promotion_no_prompt_or_scoring_change",
            "summary": {},
            "write_path": {"sources": []},
            "promotion_path": {},
            "read_path": {},
            "maintenance_path": {"forgetting_review_sample": []},
            "hardware_pressure_proxy": {},
            "recommendations": [],
        }
    if isinstance(payload, dict):
        return {"generated": True, **payload}
    return {
        "generated": False,
        "mode": "read_only_memory_engineering_observation",
        "guardrail": "report_payload_invalid",
        "summary": {},
        "write_path": {"sources": []},
        "promotion_path": {},
        "read_path": {},
        "maintenance_path": {"forgetting_review_sample": []},
        "hardware_pressure_proxy": {},
        "recommendations": [],
    }


@router.get("/api/shion/post-hackathon-status")
def get_shion_post_hackathon_status() -> dict:
    """ハッカソン後バックログの実装状態を集約して返す。"""
    from api.shion_post_hackathon import build_post_hackathon_status

    return build_post_hackathon_status()


@router.get("/api/shion/memory-health")
def get_shion_memory_health() -> dict:
    """記憶インデックスの健康診断を読み取り専用で返す。"""
    from api.shion_post_hackathon import build_memory_health_payload

    return build_memory_health_payload()


@router.post("/api/debug/shion-hyde-rag")
def post_shion_hyde_rag_debug(req: ShionHydeDebugRequest) -> dict:
    """Shion-HyDE検索のdebug比較。本回答・RAG索引・プロンプトは変更しない。"""
    from api.shion_post_hackathon import build_shion_hyde_debug_payload

    return build_shion_hyde_debug_payload(
        message=req.message,
        industry=req.industry,
        asset_name=req.asset_name,
        score_band=req.score_band,
        known_risk_tags=req.known_risk_tags,
        limit=req.limit,
    )


@router.post("/api/shion/counter-hypothesis-card")
def post_counter_hypothesis_card(req: CounterHypothesisCardRequest) -> dict:
    """違和感抽出用の反証可能な次善仮説カードを返す。通常回答には混ぜない。"""
    from api.shion_post_hackathon import build_counter_hypothesis_card

    return build_counter_hypothesis_card(
        case_summary=req.case_summary,
        industry=req.industry,
        asset_name=req.asset_name,
        score_band=req.score_band,
        human_discomfort=req.human_discomfort,
        risk_tags=req.risk_tags,
    )


@router.get("/api/shion/memory-review-inbox")
def get_shion_memory_review_inbox(
    status: str = "candidate",
    source: str = "all",
    q: str = "",
    limit: int = 30,
    offset: int = 0,
) -> dict:
    """記憶候補のレビュー受け箱を返す。判断結果は別stateで重ねる。"""
    from api.memory_review_inbox import list_inbox

    allowed_statuses = {"all", "candidate", "adopted", "revised", "held", "rejected"}
    if status not in allowed_statuses:
        raise HTTPException(status_code=422, detail="invalid status")
    return list_inbox(status=status, source=source, q=q, limit=limit, offset=offset)


@router.post("/api/shion/memory-review-inbox/{inbox_id}/review")
def post_shion_memory_review(inbox_id: str, req: MemoryReviewRequest, background_tasks: BackgroundTasks) -> dict:
    """記憶候補の人間判断を保存する。元候補ファイルは変更しない。"""
    from api.memory_review_inbox import review_candidate

    try:
        item = review_candidate(
            inbox_id,
            decision=req.decision,
            note=req.note,
            edited_claim=req.edited_claim,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="memory review candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="memory_review_inbox_reviewed",
        surface="shion_memory_system",
        payload={
            "schema_version": 1,
            "inbox_id": inbox_id,
            "decision": req.decision,
            "source": item.get("source"),
            "source_item_id": item.get("source_item_id"),
        },
    )
    return {"status": "ok", "item": item}


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
