"""Lease-news collection, presentation, and feedback endpoints."""
from __future__ import annotations

import ipaddress
import os
import socket
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from api.background_executor import background_executor
from api.cloudrun_writeback import record_cloudrun_input_event, record_lease_news_usage_feedback_event
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
from api.lease_news_summary_render import (
    parse_recent_news_note,
    recent_news_dedupe_key,
    render_news_obsidian_note,
    render_news_summary,
)
from api.llm_json_guard import extract_candidate_text, parse_or_recover_json, with_retry_tokens
from api.schemas import LeaseNewsSummarizeRequest
from lease_news_digest import (
    build_daily_news_digest,
    build_lease_news_brief,
    find_vault,
    get_latest_lease_news_actions,
    get_latest_lease_news_focus,
    record_lease_news_collection,
    record_lease_news_usage_feedback,
)
from runtime_paths import resolve_obsidian_vault

router = APIRouter(prefix="/api/lease-news", tags=["lease-news"])

_NEWS_OBSIDIAN_DIR = "05-クリップ_記事/業界リスクニュース"
_NEWS_OBSIDIAN_DIR_ALIASES = (
    "05-クリップ_記事/業界リスクニュース",
    "業界リスクニュース",
    "05-クリップ_記事/リースニュース",
    "リースニュース",
)


def _gemini_generate_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _news_vault_root() -> Path | None:
    vault = find_vault()
    if vault and vault.is_dir():
        return vault
    fallback = resolve_obsidian_vault()
    return fallback if fallback.is_dir() else None


def _lease_news_dir(vault: Path, create: bool = False) -> Path | None:
    """Return the folder containing industry-risk news notes."""
    for rel in _NEWS_OBSIDIAN_DIR_ALIASES:
        candidate = vault / rel
        if candidate.exists():
            return candidate
    if create:
        candidate = vault / _NEWS_OBSIDIAN_DIR
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    return None


def _validate_public_http_url(url: str) -> None:
    """Reject URLs that could reach local/private infrastructure."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="URLは http/https のみ指定できます")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="認証情報を含むURLは指定できません")
    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="URLのホスト名が不正です")
    if host.lower() in {"localhost", "metadata.google.internal"} or host.lower().endswith(".local"):
        raise HTTPException(status_code=400, detail="ローカル/内部ホストは指定できません")
    try:
        addresses = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail=f"URLの名前解決に失敗: {exc}") from exc
    for addr in addresses:
        ip = ipaddress.ip_address(addr[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HTTPException(status_code=400, detail="ローカル/内部ネットワーク宛のURLは指定できません")


def _fetch_url_text(url: str) -> str:
    """Fetch public HTML, validating every redirect before connecting to it."""
    current_url = url
    response = None
    for _ in range(6):
        _validate_public_http_url(current_url)
        response = requests.get(
            current_url,
            timeout=15,
            headers={"User-Agent": "TuneLeaseBot/1.0"},
            allow_redirects=False,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            if not location:
                raise HTTPException(status_code=400, detail="リダイレクト先が不正です")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()
        break
    else:
        raise HTTPException(status_code=400, detail="リダイレクト回数が上限を超えました")

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self._parts.append(stripped)

    parser = _TextExtractor()
    parser.feed(response.text)
    return "\n".join(parser._parts)[:6000]


def _summarize_news_with_gemini(text: str, source: str) -> dict:
    from api.chat_memory import _get_gemini_api_key as _chat_get_gemini_api_key

    api_key = _chat_get_gemini_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini APIキーが未設定です")

    prompt = f"""あなたはリース審査担当向けに、顧客業界・物件・市況ニュースを分類するアシスタントです。
以下のニュース記事を読み、短い構造JSONだけを出力してください。説明文は不要です。

{{
  "title": "ニュースタイトル（15文字〜30文字）",
  "summary_codes": ["CAPEX/RATE/REGULATION/MARKET/RISK/TECH/ASSET から最大3件"],
  "key_phrases": ["記事内の重要語句を最大5件、各40文字以内"],
  "usage_codes": ["PROPOSAL_TIMING/RATE_EXPLAIN/RISK_CHECK/ASSET_MATCH/INDUSTRY_TALK/FOLLOW_UP から最大2件"],
  "tags": ["タグ1", "タグ2", "タグ3"],
  "region": "国内/米国/欧州/アジア のいずれか1つ",
  "importance": "高/中/低"
}}

タグは顧客業界（製造、建設、運送等）、物件（工作機械、建機、車両等）、トピック（金利動向、倒産、設備投資、補助金、中古価格等）から選んでください。
リース会社そのもののニュースではなく、借手の返済力、設備稼働、投資回収、物件価値に影響する論点を優先してください。
regionは記事の主な対象地域を判定してください。日本国内のニュースは「国内」、米国は「米国」、欧州は「欧州」、中国・東南アジア等は「アジア」。複数地域にまたがる場合は主な地域を1つ選んでください。
summary_codes と usage_codes は必ず上記の英字コードだけを返してください。

ニュース記事:
{text[:4000]}
"""
    defaults = {
        "title": "業界リスクニュース",
        "summary_lines": [
            "ニュース本文の自動要約が一部不完全です。",
            "原文を確認して営業活用可否を判断してください。",
            f"情報源: {source[:80] or '不明'}",
        ],
        "usage_memo": "要約の自動生成が不完全なため、原文確認後に提案材料として扱ってください。",
        "summary_codes": ["MARKET"],
        "usage_codes": ["INDUSTRY_TALK"],
        "key_phrases": [],
        "tags": ["要確認"],
        "region": "国内",
        "importance": "中",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }

    result = defaults
    finish_reason = ""
    for current_payload in (payload, with_retry_tokens(payload, 2048)):
        response = requests.post(
            _gemini_generate_url(),
            json=current_payload,
            headers={"x-goog-api-key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        raw, finish_reason = extract_candidate_text(response.json())
        result, recovered = parse_or_recover_json(
            raw,
            defaults=defaults,
            string_fields={"title", "usage_memo", "region", "importance"},
            array_fields={"summary_codes", "key_phrases", "usage_codes", "summary_lines", "tags"},
        )
        if not recovered and finish_reason != "MAX_TOKENS":
            break
        if current_payload["generationConfig"]["maxOutputTokens"] >= 2048:
            break
    if result.get("region") not in {"国内", "米国", "欧州", "アジア"}:
        result["region"] = "国内"
    if result.get("importance") not in {"高", "中", "低"}:
        result["importance"] = "中"
    if not isinstance(result.get("summary_lines"), list) or not result["summary_lines"]:
        result["summary_lines"] = defaults["summary_lines"]
    if not isinstance(result.get("tags"), list) or not result["tags"]:
        result["tags"] = defaults["tags"]
    if finish_reason == "MAX_TOKENS":
        result["_finish_reason"] = finish_reason
    return render_news_summary(result, source)


def _save_news_to_obsidian(summary: dict, source: str) -> str | None:
    vault = _news_vault_root()
    if not vault:
        return None
    news_dir = _lease_news_dir(vault, create=True)
    if not news_dir:
        return None

    note = render_news_obsidian_note(summary, source)
    file_path = news_dir / note["filename"]
    file_path.write_text(note["content"], encoding="utf-8")
    try:
        record_lease_news_collection(
            date_str=note["date"],
            note_path=str(file_path.relative_to(vault)),
            article_count=1,
            source_summary=source[:100],
            tag_summary=", ".join(summary.get("tags", [])),
        )
    except Exception:
        pass

    try:
        from api.knowledge.news_classifier import write_classified_news_summary

        background_executor.submit(lambda: write_classified_news_summary(vault, limit=30, days=14))
    except Exception:
        pass
    try:
        from api.knowledge.obsidian_loader import _chunk_by_h2, _parse_frontmatter
        from api.knowledge.vector_store import get_store

        raw = file_path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        chunks = _chunk_by_h2(body, str(file_path), file_path.name, meta, file_path.stat().st_mtime)
        if chunks:
            background_executor.submit(lambda: get_store().upsert_chunks(chunks))
    except Exception:
        pass
    try:
        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")

        def _run_wikilink():
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            try:
                from auto_wikilink import run_on_files

                run_on_files([file_path], vault)
            except Exception:
                pass

        background_executor.submit(_run_wikilink)
    except Exception:
        pass
    return str(file_path)


@router.post("/summarize")
def summarize_lease_news(req: LeaseNewsSummarizeRequest):
    """ニュースURL or 本文テキストをAI要約し、Obsidianに保存する。"""
    source = req.url or "手動入力"
    if req.url and req.url.strip():
        try:
            text = _fetch_url_text(req.url.strip())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"URLの取得に失敗: {exc}") from exc
    elif req.body_text and req.body_text.strip():
        text = req.body_text.strip()
    else:
        raise HTTPException(status_code=400, detail="URLまたは本文テキストを入力してください")

    summary = _summarize_news_with_gemini(text, source)
    saved_path = _save_news_to_obsidian(summary, source)
    return {
        "status": "ok",
        "title": summary.get("title", ""),
        "summary_lines": summary.get("summary_lines", []),
        "usage_memo": summary.get("usage_memo", ""),
        "summary_codes": summary.get("summary_codes", []),
        "usage_codes": summary.get("usage_codes", []),
        "key_phrases": summary.get("key_phrases", []),
        "tags": summary.get("tags", []),
        "region": summary.get("region", "国内"),
        "importance": summary.get("importance", "中"),
        "saved_path": saved_path,
    }


@router.get("/recent")
def get_recent_lease_news(limit: int = 5):
    """Obsidianの業界リスクニュースから直近N件の要約を返す。"""
    vault = _news_vault_root()
    if not vault:
        return {"items": []}
    news_dir = _lease_news_dir(vault)
    if not news_dir:
        return {"items": []}

    md_files = sorted(news_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    items: list[dict] = []
    seen_keys: set[str] = set()
    for file_path in md_files:
        try:
            raw = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        item = parse_recent_news_note(raw, file_path=str(file_path), file_stem=file_path.stem)
        dedupe_key = recent_news_dedupe_key(item)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        items.append(item)
        if len(items) >= max(1, min(int(limit), 20)):
            break
    return {"items": items}


class LeaseNewsJudgmentChangeRequest(BaseModel):
    case_id: str = ""
    company_name: str = ""
    score: float | None = None
    model_decision: str = ""
    final_decision: str = ""
    news_focus: list[str] = Field(default_factory=list)
    news_focus_summary: str = ""
    news_focus_tag_summary: str = ""
    news_focus_note_path: str = ""
    news_focus_note_date: str = ""
    reason: str = ""
    input_snapshot: dict = Field(default_factory=dict)


class LeaseNewsUsageFeedbackRequest(BaseModel):
    source_path: str = Field(min_length=1, max_length=500)
    source_title: str = Field(default="", max_length=240)
    outcome: str = Field(pattern="^(used|irrelevant|question_changed|condition_changed)$")
    surface: str = Field(default="news_dashboard", max_length=80)
    case_id: str = Field(default="", max_length=120)
    note: str = Field(default="", max_length=500)


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


@router.post("/feedback")
def record_lease_news_usage_feedback_api(req: LeaseNewsUsageFeedbackRequest):
    """Record whether a surfaced news action changed real screening work."""
    try:
        feedback = record_lease_news_usage_feedback(
            source_path=req.source_path,
            source_title=req.source_title,
            outcome=req.outcome,
            surface=req.surface,
            case_id=req.case_id,
            note=req.note,
        )
        writeback = record_lease_news_usage_feedback_event(feedback)
        return {"status": "recorded", "feedback": feedback, "writeback": writeback}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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


@router.post("/judgment-change")
def record_lease_news_judgment_change_api(
    req: LeaseNewsJudgmentChangeRequest,
    background_tasks: BackgroundTasks,
):
    """ニュース参照後の判断変更を記録する。"""
    import datetime as dt

    from judgment_feedback import record_judgment_feedback
    from lease_news_digest import record_lease_news_judgment_change

    try:
        feedback = record_judgment_feedback(
            case_id=req.case_id or f"news-{dt.datetime.now().isoformat()}",
            model_decision=req.model_decision,
            human_decision=req.final_decision,
            reason=req.reason,
            source="lease_news_debate",
            score=req.score,
            input_snapshot=req.input_snapshot,
            evidence_snapshot={
                "news_focus": req.news_focus,
                "summary": req.news_focus_summary,
                "tags": req.news_focus_tag_summary,
                "note_path": req.news_focus_note_path,
                "note_date": req.news_focus_note_date,
            },
        )
        if not feedback.get("success"):
            raise HTTPException(status_code=422, detail=feedback.get("error"))
        background_tasks.add_task(
            record_cloudrun_input_event,
            event_type="lease_news_judgment_change",
            surface="lease_news_judgment_change",
            payload=req.model_dump(),
        )
        bucket = record_lease_news_judgment_change(
            date_str=dt.date.today().isoformat(),
            note_path=req.news_focus_note_path or "",
            source_note_date=req.news_focus_note_date or "",
            company_name=req.company_name or "",
            score=req.score,
            final_decision=req.final_decision or "",
            reason=req.reason or "",
            focus_lines=tuple(req.news_focus or []),
            theme_summary=req.news_focus_summary or "",
            tag_summary=req.news_focus_tag_summary or "",
        )
        return {"status": "recorded", "metrics": bucket, "model_improvement": feedback}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
