"""紫苑の能動的アラートチェック（受け身のチャット応答とは別系統）。

get_recent_errors（logs/api.log・app.log の集計）だけを使い、外部API課金なしで
「直近でエラーが急増していないか」を判定する。フロントはこれを定期ポーリングし、
異常があるときだけ紫苑からの割り込み発言として表示する。

判断領域の先回り（check_judgment_prediction_alert）はここに同居するが別系統で、
scripts/build_predictive_framework_report.py が出した観測結果を読むだけに徹する。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 暫定閾値。実運用のログ量を見ながら調整する。
_ALERT_LOOKBACK_HOURS = 3
_ALERT_ERROR_LINE_THRESHOLD = 10

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREDICTIVE_REPORT_PATH = _REPO_ROOT / "reports" / "predictive_framework_latest.json"
# レポートが古いまま出続けると「今の話」に見えて誤解を生むため、期限を切る。
_PREDICTIVE_REPORT_MAX_AGE_DAYS = 7
_JUDGMENT_ALERT_LIMIT = 3


def check_shion_proactive_alerts() -> dict:
    """直近のエラーログを見て、紫苑から知らせるべき異常があるか判定する。"""
    from lease_intelligence_tools import get_recent_errors

    errors = get_recent_errors(hours=_ALERT_LOOKBACK_HOURS, limit=5)
    total_lines = errors.get("total_error_lines", 0) or 0
    patterns = errors.get("patterns") or []

    has_alert = total_lines >= _ALERT_ERROR_LINE_THRESHOLD
    message = None
    if has_alert:
        top = patterns[0] if patterns else {}
        top_pattern = top.get("pattern", "不明なエラー")
        top_count = top.get("count", 0)
        message = (
            f"直近{_ALERT_LOOKBACK_HOURS}時間でエラーが{total_lines}件出ています。"
            f"一番多いのは「{top_pattern}」（{top_count}件）です。ログを見ておいたほうがよさそうです。"
        )

    return {
        "has_alert": has_alert,
        "message": message,
        "lookback_hours": _ALERT_LOOKBACK_HOURS,
        "total_error_lines": total_lines,
        "top_patterns": patterns[:3],
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _load_predictive_report(report_path: Path) -> dict | None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return report if isinstance(report, dict) else None


def _report_age_days(report: dict) -> float | None:
    raw = str(report.get("generated_at") or "").strip()
    if not raw:
        return None
    try:
        generated_at = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    return (datetime.now(tz=timezone.utc) - generated_at).total_seconds() / 86400.0


def check_judgment_prediction_alert(
    *,
    report_path: Path | None = None,
    limit: int = _JUDGMENT_ALERT_LIMIT,
) -> dict:
    """繰り返し外している前提を、案件を開いた時点で先回りして伝える。

    エラー急増検知とは別系統。scripts/build_predictive_framework_report.py が
    出した `reports/predictive_framework_latest.json` を読むだけで、集計も
    判断資産の更新もここではしない。該当が無ければ黙る（沈黙が既定）。
    """
    path = report_path or _PREDICTIVE_REPORT_PATH
    silent = {
        "has_alert": False,
        "message": None,
        "items": [],
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    report = _load_predictive_report(path)
    if report is None:
        return {**silent, "reason": "report_unavailable"}

    age_days = _report_age_days(report)
    if age_days is not None and age_days > _PREDICTIVE_REPORT_MAX_AGE_DAYS:
        return {**silent, "reason": "report_stale", "report_age_days": round(age_days, 1)}

    coverage = report.get("prediction_coverage") if isinstance(report.get("prediction_coverage"), dict) else {}
    if not coverage.get("trustworthy"):
        # 事前予測が1件も無い状態の集計を根拠に警告すると、逆算結果を
        # 実績のように見せてしまうため出さない。
        return {**silent, "reason": "prediction_coverage_not_trustworthy"}

    repeat_beliefs = report.get("repeat_beliefs") if isinstance(report.get("repeat_beliefs"), list) else []
    items = [
        row
        for row in repeat_beliefs
        if isinstance(row, dict) and row.get("alertable")
    ][: max(1, limit)]

    warnings = report.get("calibration_warnings") if isinstance(report.get("calibration_warnings"), list) else []
    warning_messages = [
        str(item.get("message") or "").strip()
        for item in warnings
        if isinstance(item, dict) and str(item.get("message") or "").strip()
    ]

    if not items and not warning_messages:
        return {**silent, "reason": "no_repeat_miss"}

    lines: list[str] = []
    for row in items:
        belief = str(row.get("target_belief") or "").strip() or "不明な前提"
        occurrences = row.get("occurrences")
        lines.append(f"「{belief}」の見立てが直近{occurrences}件で外れていて、まだ採否がついていません。")
    lines.extend(warning_messages)

    return {
        "has_alert": True,
        "message": "\n".join(lines),
        "items": items,
        "calibration_warnings": warning_messages,
        "report_generated_at": report.get("generated_at", ""),
        "policy": "read_only_human_review_required",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def check_shion_latent_need_alert(prefecture: str = "", industry: str = "") -> dict:
    """案件の業種・地域から、まだ提示していない業界動向・審査上の気づきがあるか判定する。

    エラー急増検知（check_shion_proactive_alerts）とは別系統。既存の日次業界ブリーフ
    （lease_news_digest.build_lease_news_brief、/api/lease-news/brief と同じデータ源）を
    再利用し、能動的な一言として出せる内容があるかだけを軽量に判定する。
    「once per day」の重複抑制はフロント側（既存の lease-news-brief-seen-<date> キー）が担う。
    """
    from lease_news_digest import build_lease_news_brief

    brief = build_lease_news_brief(prefecture=prefecture or "", industry=industry or "")
    available = bool(getattr(brief, "available", False))

    message = None
    if available:
        opening = (getattr(brief, "opening_line", "") or "").strip()
        question = (getattr(brief, "question_line", "") or "").strip()
        message = "\n".join(line for line in (opening, question) if line) or None

    return {
        "has_alert": bool(message),
        "message": message,
        "topic": industry or "",
        "prefecture": prefecture or "",
        "note_date": getattr(brief, "note_date", "") if available else "",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
