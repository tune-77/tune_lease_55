"""紫苑の能動的アラートチェック（受け身のチャット応答とは別系統）。

get_recent_errors（logs/api.log・app.log の集計）だけを使い、外部API課金なしで
「直近でエラーが急増していないか」を判定する。フロントはこれを定期ポーリングし、
異常があるときだけ紫苑からの割り込み発言として表示する。
"""
from __future__ import annotations

from datetime import datetime, timezone

# 暫定閾値。実運用のログ量を見ながら調整する。
_ALERT_LOOKBACK_HOURS = 3
_ALERT_ERROR_LINE_THRESHOLD = 10


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
