"""Gemini 利用の1日予算ガード（REV-485）。

data/ai_usage.jsonl から当日（JST）の概算費用を出し、上限に近づいたら
検証 → 夜間ジョブ・自発系 の順に呼び出しを止める。紫苑のチャットと審査
（essential）は止めない。止めた呼び出しと上限到達は data/ai_budget_events.jsonl
に残し、朝報（aurion_core_daily）に出す。本文・引数は記録しない。

環境変数:
- AI_DAILY_BUDGET_YEN（既定 50）/ AI_MONTHLY_BUDGET_YEN（既定 2000）
- AI_USD_JPY（既定 150）/ AI_COST_CALIBRATION（既定 1.5。2026-10-01〜07 の
  実請求 778円 ÷ トークンからの概算 約530円（税込換算）。未計測経路と税を含めた補正）
- AI_LIVE_VERIFY=1: 検証クラスの実呼び出しを許可（既定は止める）
- AI_VERIFY_MAX_CALLS（既定 30/日）: AI_LIVE_VERIFY=1 でも1日の検証呼び出し上限
- AI_CALL_CLASS: 呼び出しクラスの明示（essential / proactive / nightly / verification）
- AI_BUDGET_GUARD=off で無効。pytest 中は =force の時だけ有効
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

JST = timezone(timedelta(hours=9))

ESSENTIAL = "essential"
PROACTIVE = "proactive"
NIGHTLY = "nightly"
VERIFICATION = "verification"
CLASSES = (ESSENTIAL, PROACTIVE, NIGHTLY, VERIFICATION)

# 当日費用が予算のこの割合に達したら止める（essential は止めない）
STOP_RATIO = {VERIFICATION: 0.6, PROACTIVE: 0.8, NIGHTLY: 0.8}

# 自発系: 紫苑が自分から生成するもの（好奇心・内省・ループ・日次画像など）
PROACTIVE_FEATURES = frozenset({
    "lease_intelligence_reflection",  # Private Reflection
    "mind_reflection",
    "mind_reflection_legacy_fallback",
    "shion_activity_reflection",  # 行動観察からの理解と好奇心
    "usage_loop_engineering",
    "novelist_daily_image",
    "novelist_daily_image_fallback",
    "world_view_update",
    "shion_self_analysis",
})
# 夜間・バッチ系
NIGHTLY_FEATURES = frozenset({
    "auto_research_lease_judgment",
    "lease_news_collection",
    "aurion_core_inference",
    "dispatch_log_summary",
    "crystallizer_bias_extraction",
    "crystallizer_pattern_synthesis",
    "rule_engine_apply",
    "improvement_consolidation",
    "answer_quality_improvement",
    "codex_queue",
    "shion_triage",
})
# 手動実行の検証・実験スクリプト（launchd 経由は夜間扱い）
_VERIFY_SCRIPT_RE = re.compile(r"^(eval|evaluate|experiment|replay|compare|bench|probe)[_\-.]")

# USD / 1M tokens: (入力, 出力)。キャッシュ読込は入力の1割。上から順に部分一致。
# 出典: https://ai.google.dev/gemini-api/docs/pricing（2026-10-07確認。不明モデルは高め）
PRICES: tuple[tuple[str, float, float], ...] = (
    ("flash-lite", 0.25, 1.50),
    ("image", 0.50, 60.0),
    ("2.5-flash", 0.30, 2.50),
    ("pro", 2.00, 12.00),
    ("flash", 0.50, 3.00),
)
DEFAULT_PRICE = (0.50, 3.00)

_REPO_ROOT = Path(__file__).resolve().parent
_cache: dict[str, Any] = {}


class AIBudgetBlocked(RuntimeError):
    """予算ガードで止めた呼び出し。呼び出し側の既存の例外処理で握られる前提。"""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def daily_budget_yen() -> float:
    return _env_float("AI_DAILY_BUDGET_YEN", 50.0)


def monthly_budget_yen() -> float:
    return _env_float("AI_MONTHLY_BUDGET_YEN", 2000.0)


def guard_enabled() -> bool:
    flag = os.environ.get("AI_BUDGET_GUARD", "on").strip().lower()
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return flag == "force"
    return flag not in {"0", "off", "false", "no"}


def _events_path() -> Path:
    from ai_runtime_client import usage_log_path

    configured = os.environ.get("AI_BUDGET_EVENTS_PATH", "").strip()
    return Path(configured).expanduser() if configured else usage_log_path().with_name("ai_budget_events.jsonl")


def call_class(feature: str, source: dict[str, str] | None = None) -> str:
    """呼び出しクラス。明示 > 検証（worktree・その場実行・検証スクリプト）> launchd > feature。"""
    explicit = os.environ.get("AI_CALL_CLASS", "").strip().lower()
    if explicit in CLASSES:
        return explicit
    source = source or {}
    entry = source.get("source", "")
    launchd = os.environ.get("XPC_SERVICE_NAME", "")
    scheduled = launchd.startswith("com.tunelease.") and launchd != "com.tunelease.next"
    if source.get("worktree") or entry in {"-", "-c"}:
        return VERIFICATION
    if not scheduled and _VERIFY_SCRIPT_RE.match(entry):
        return VERIFICATION
    if feature in PROACTIVE_FEATURES:
        return PROACTIVE
    if scheduled or feature in NIGHTLY_FEATURES:
        return NIGHTLY
    return ESSENTIAL


def entry_cost_usd(entry: dict[str, Any]) -> float:
    if str(entry.get("provider") or "") != "google":
        return 0.0
    model = str(entry.get("model") or "")
    price_in, price_out = next(((pi, po) for key, pi, po in PRICES if key in model), DEFAULT_PRICE)
    tokens_in = int(entry.get("input_tokens") or 0)
    cached = min(int(entry.get("cached_tokens") or 0), tokens_in)
    # thinking トークンは出力として課金されるが candidates には入らないので total から引き直す
    total = int(entry.get("total_tokens") or 0)
    tokens_out = max(int(entry.get("output_tokens") or 0), total - tokens_in)
    return ((tokens_in - cached) * price_in + cached * price_in * 0.1 + tokens_out * price_out) / 1_000_000


def to_yen(usd: float) -> float:
    return usd * _env_float("AI_USD_JPY", 150.0) * _env_float("AI_COST_CALIBRATION", 1.5)


def _iter_entries(path: Path) -> Iterable[dict[str, Any]]:
    for log_path in (path.with_suffix(f"{path.suffix}.1"), path):
        if not log_path.exists():
            continue
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue


def _jst_date(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).astimezone(JST).date().isoformat()
    except (TypeError, ValueError):
        return ""


def summarize_days(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """JST 日付ごとの概算費用（円）・クラス別費用・検証呼び出し数。"""
    from ai_runtime_client import usage_log_path

    days: dict[str, dict[str, Any]] = {}
    for entry in _iter_entries(path or usage_log_path()):
        day = _jst_date(str(entry.get("timestamp") or ""))
        if not day:
            continue
        row = days.setdefault(day, {"yen": 0.0, "by_class": {}, "verification_calls": 0})
        yen = to_yen(entry_cost_usd(entry))
        cls = str(entry.get("call_class") or ESSENTIAL)
        row["yen"] += yen
        row["by_class"][cls] = row["by_class"].get(cls, 0.0) + yen
        if cls == VERIFICATION:
            row["verification_calls"] += 1
    return days


def today_status(*, now: datetime | None = None, max_age_s: float = 60.0) -> dict[str, Any]:
    """当日分の費用（60秒キャッシュ。essential 以外の呼び出し前だけ読む）。"""
    today = (now or datetime.now(JST)).astimezone(JST).date().isoformat()
    cached = _cache.get("today")
    if cached and cached["date"] == today and time.monotonic() - cached["at"] < max_age_s:
        return cached["status"]
    row = summarize_days().get(today, {"yen": 0.0, "by_class": {}, "verification_calls": 0})
    status = {"date": today, **row, "budget_yen": daily_budget_yen()}
    _cache["today"] = {"date": today, "at": time.monotonic(), "status": status}
    return status


def note_call(cls: str, cost_usd: float) -> None:
    """記録直後に当日キャッシュへ加算（60秒の間の連続呼び出しで上限を飛び越えない）。"""
    cached = _cache.get("today")
    if cached:
        status = cached["status"]
        status["yen"] += to_yen(cost_usd)
        if cls == VERIFICATION:
            status["verification_calls"] += 1


def _record_event(event: dict[str, Any]) -> None:
    try:
        path = _events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        pass
    print(f"[ai_budget] {event.get('event')} class={event.get('call_class')} feature={event.get('feature')} "
          f"spent=¥{event.get('spent_yen')} budget=¥{event.get('budget_yen')}", file=sys.stderr)


def _note_limit_reached(status: dict[str, Any]) -> None:
    if status["yen"] < status["budget_yen"] or _cache.get("limit_logged") == status["date"]:
        return
    _cache["limit_logged"] = status["date"]
    _record_event({
        "ts": datetime.now(JST).isoformat(timespec="seconds"),
        "date": status["date"],
        "event": "limit_reached",
        "spent_yen": round(status["yen"], 1),
        "budget_yen": status["budget_yen"],
    })


def check(feature: str, cls: str) -> None:
    """止めるべき呼び出しなら AIBudgetBlocked。essential は常に通す。"""
    if cls == ESSENTIAL or not guard_enabled():
        return
    status = today_status()
    _note_limit_reached(status)
    reason = ""
    if cls == VERIFICATION and os.environ.get("AI_LIVE_VERIFY", "").strip() != "1":
        reason = "live_verify_flag_required"
    elif cls == VERIFICATION and status["verification_calls"] >= int(_env_float("AI_VERIFY_MAX_CALLS", 30)):
        reason = "verify_daily_cap"
    elif status["yen"] >= status["budget_yen"] * STOP_RATIO.get(cls, 1.0):
        reason = "daily_budget"
    if not reason:
        return
    _record_event({
        "ts": datetime.now(JST).isoformat(timespec="seconds"),
        "date": status["date"],
        "event": "blocked",
        "call_class": cls,
        "feature": feature,
        "reason": reason,
        "spent_yen": round(status["yen"], 1),
        "budget_yen": status["budget_yen"],
    })
    hint = "（検証で本物の呼び出しが必要なら AI_LIVE_VERIFY=1。件数は AI_VERIFY_MAX_CALLS まで）" if cls == VERIFICATION else ""
    raise AIBudgetBlocked(f"AI予算ガードで停止: {cls}/{feature} reason={reason}{hint}")


def deferred(cls: str) -> bool:
    """Gemini 以外（Jev 照合など）の自発処理を、同じ基準で先送りするか。"""
    if cls == ESSENTIAL or not guard_enabled():
        return False
    try:
        status = today_status()
    except Exception:
        return False
    return status["yen"] >= status["budget_yen"] * STOP_RATIO.get(cls, 1.0)


def _read_events(day: str) -> list[dict[str, Any]]:
    path = _events_path()
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("date") == day:
            events.append(item)
    return events


def morning_report_lines(*, today: datetime | None = None) -> list[str]:
    """朝報用: 前日の概算費用・クラス別・止めた件数と、今月の着地見込み。"""
    now = (today or datetime.now(JST)).astimezone(JST)
    yesterday = (now.date() - timedelta(days=1)).isoformat()
    days = summarize_days()
    row = days.get(yesterday, {"yen": 0.0, "by_class": {}})
    budget = daily_budget_yen()
    events = _read_events(yesterday)
    blocked = [e for e in events if e.get("event") == "blocked"]
    lines: list[str] = []
    if row["yen"] >= budget or any(e.get("event") == "limit_reached" for e in events):
        lines.append(f"> [!warning] Gemini 1日予算に到達（{yesterday} 推定 ¥{row['yen']:.0f} / 上限 ¥{budget:.0f}）")
    if blocked:
        by: dict[str, int] = {}
        for e in blocked:
            key = f"{e.get('call_class')}:{e.get('reason')}"
            by[key] = by.get(key, 0) + 1
        lines.append("- ⛔ 予算ガードで止めた呼び出し: " + " / ".join(f"{k} {v}件" for k, v in sorted(by.items())))
    month = now.strftime("%Y-%m")
    month_days = [d for d in days if d.startswith(month) and d < now.date().isoformat()]
    month_yen = sum(days[d]["yen"] for d in month_days)
    breakdown = " / ".join(f"{k} ¥{v:.0f}" for k, v in sorted(row["by_class"].items(), key=lambda kv: -kv[1]))
    lines.append(
        f"- 💴 Gemini 推定（ai_usage.jsonl×補正{_env_float('AI_COST_CALIBRATION', 1.5)}）: 前日 ¥{row['yen']:.0f}"
        f"（{breakdown or '記録なし'}）・上限 ¥{budget:.0f}/日・今月記録分 ¥{month_yen:.0f}"
        f"（{len(month_days)}日分）・月目標 ¥{monthly_budget_yen():.0f}"
    )
    return lines
