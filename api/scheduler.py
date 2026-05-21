"""
APScheduler による定期バッチスケジューラ。
毎日 02:00 に知識結晶化バッチを実行する。
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def run_crystallization_batch() -> dict:
    """
    知識結晶化バッチのメイン処理。
    1. 外れ値/意見割れ案件を抽出
    2. Gemini でパターンを言語化
    3. Obsidian に書き出す

    Returns:
        {"status": str, "cases_found": int, "file": str | None}
    """
    logger.info("[Crystallizer] バッチ開始")

    try:
        from api.crystallizer.anomaly_extractor import extract_anomalies
        cases = extract_anomalies()
        logger.info(f"[Crystallizer] 抽出案件数: {len(cases)}")

        if not cases:
            return {"status": "no_anomalies", "cases_found": 0, "file": None}

        from api.crystallizer.pattern_synthesizer import synthesize_pattern
        pattern_text = synthesize_pattern(cases)

        from api.crystallizer.obsidian_writer import write_pattern_to_obsidian
        fpath = write_pattern_to_obsidian(pattern_text, cases)

        logger.info(f"[Crystallizer] 書き出し完了: {fpath}")
        return {"status": "ok", "cases_found": len(cases), "file": fpath}

    except Exception as e:
        logger.error(f"[Crystallizer] バッチエラー: {e}", exc_info=True)
        return {"status": "error", "detail": str(e), "cases_found": 0, "file": None}


def start_scheduler() -> BackgroundScheduler:
    """
    APScheduler を起動して毎日 02:00 に結晶化バッチを登録する。
    FastAPI の startup イベントから呼ぶ。
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="Asia/Tokyo")
    _scheduler.add_job(
        run_crystallization_batch,
        trigger=CronTrigger(hour=2, minute=0, timezone="Asia/Tokyo"),
        id="crystallization_daily",
        name="知識結晶化バッチ（毎日02:00）",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("[Scheduler] 起動完了。毎日 02:00 JST に結晶化バッチを実行します。")
    return _scheduler


def stop_scheduler() -> None:
    """FastAPI の shutdown イベントから呼ぶ。"""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[Scheduler] 停止しました。")
