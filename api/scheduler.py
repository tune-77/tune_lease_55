"""
APScheduler による定期バッチスケジューラ。
毎日 02:00 に知識結晶化バッチを実行する。
毎日 03:00 に紫苑フィードバック傾向ループを実行し、提案を改善ログへ投入する。
毎日 03:30 に紫苑画面利用ループを実行し、提案を改善ログへ投入する。
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from pathlib import Path
from typing import Any

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


def _push_proposals_to_improvement_log(
    proposals: list[dict[str, Any]],
    source: str,
) -> int:
    """
    紫苑の自己提案を cloudrun_improvement_log.jsonl へ追記する。
    重複チェック: 同じ title が既存エントリにあればスキップ。
    戻り値: 追記した件数
    """
    import json
    import os

    data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))
    log_path = data_dir / "cloudrun_improvement_log.jsonl"

    # 既存タイトルを読み込んでおく（重複防止）
    existing_titles: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                t = str(entry.get("title") or "").strip()
                if t:
                    existing_titles.add(t)
            except json.JSONDecodeError:
                continue

    pushed = 0
    ts_now = dt.datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        for p in proposals:
            title = str(p.get("title") or "").strip()
            if not title or title in existing_titles:
                continue
            body_parts = []
            if p.get("pattern"):
                body_parts.append(f"## パターン\n{p['pattern']}")
            if p.get("suggestion"):
                body_parts.append(f"## 提案\n{p['suggestion']}")
            if p.get("reason"):
                body_parts.append(f"## 理由\n{p['reason']}")
            entry = {
                "event_id": str(uuid.uuid4()),
                "ts": ts_now,
                "title": title,
                "body": "\n\n".join(body_parts),
                "surface": "shion_self_proposal",
                "source": source,
                "proposed_by": "shion",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing_titles.add(title)
            pushed += 1

    return pushed


def run_shion_feedback_loop() -> dict:
    """
    紫苑フィードバック傾向ループ（毎日 03:00）。
    人間の応答評価 + 経験イベントの弱シグナルを統合分析し、改善提案を生成して改善ログへ投入する。
    その後、採用済み提案の before/after PDCA評価も実行する。
    """
    logger.info("[ShionFeedbackLoop] バッチ開始")
    try:
        from api.feedback_pattern_loop import evaluate_proposal_impact, generate_proposals

        # 1. 提案生成（A: ソース拡充 — feedback + experience signals）
        result = generate_proposals()
        if not result.get("generated"):
            logger.info(f"[ShionFeedbackLoop] 提案なし: {result.get('reason', '')}")
        else:
            proposals = result.get("proposals", [])
            pushed = _push_proposals_to_improvement_log(proposals, source="feedback_pattern_loop")
            logger.info(f"[ShionFeedbackLoop] 提案{len(proposals)}件 / 改善ログ投入{pushed}件")

        # 2. PDCA評価（B: ループを閉じる — 採用済み提案の効果検証）
        pdca = evaluate_proposal_impact()
        logger.info(f"[ShionFeedbackLoop] PDCA評価: {pdca.get('evaluated', 0)}件")

        return {
            "status": "ok",
            "proposals_generated": len(result.get("proposals", [])) if result.get("generated") else 0,
            "pdca_evaluated": pdca.get("evaluated", 0),
        }

    except Exception as e:
        logger.error(f"[ShionFeedbackLoop] エラー: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


def run_shion_usage_loop() -> dict:
    """
    紫苑画面利用ループ（毎日 03:30）。
    画面訪問ログを分析し、UI/UX 改善提案を生成して改善ログへ投入する。
    """
    logger.info("[ShionUsageLoop] バッチ開始")
    try:
        from api.usage_loop_engineering import generate_proposals
        result = generate_proposals()
        if not result.get("generated"):
            logger.info(f"[ShionUsageLoop] 提案なし: {result.get('reason', '')}")
            return {"status": "no_proposals", "reason": result.get("reason", "")}

        proposals = result.get("proposals", [])
        pushed = _push_proposals_to_improvement_log(proposals, source="usage_loop")
        logger.info(f"[ShionUsageLoop] 完了。提案{len(proposals)}件 / 改善ログ投入{pushed}件")
        return {"status": "ok", "proposals_generated": len(proposals), "pushed_to_log": pushed}

    except Exception as e:
        logger.error(f"[ShionUsageLoop] エラー: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


def start_scheduler() -> BackgroundScheduler:
    """
    APScheduler を起動して定期バッチを登録する。
    FastAPI の startup イベントから呼ぶ。
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="Asia/Tokyo")

    # 知識結晶化バッチ（毎日 02:00）
    _scheduler.add_job(
        run_crystallization_batch,
        trigger=CronTrigger(hour=2, minute=0, timezone="Asia/Tokyo"),
        id="crystallization_daily",
        name="知識結晶化バッチ（毎日02:00）",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # 紫苑フィードバック傾向ループ（毎日 03:00）
    _scheduler.add_job(
        run_shion_feedback_loop,
        trigger=CronTrigger(hour=3, minute=0, timezone="Asia/Tokyo"),
        id="shion_feedback_loop_daily",
        name="紫苑フィードバック傾向ループ（毎日03:00）",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # 紫苑画面利用ループ（毎日 03:30）
    _scheduler.add_job(
        run_shion_usage_loop,
        trigger=CronTrigger(hour=3, minute=30, timezone="Asia/Tokyo"),
        id="shion_usage_loop_daily",
        name="紫苑画面利用ループ（毎日03:30）",
        replace_existing=True,
        misfire_grace_time=300,
    )

    _scheduler.start()
    logger.info(
        "[Scheduler] 起動完了。"
        "毎日 02:00 JST に結晶化バッチ、"
        "03:00 に紫苑フィードバックループ、"
        "03:30 に紫苑利用ループを実行します。"
    )
    return _scheduler


def stop_scheduler() -> None:
    """FastAPI の shutdown イベントから呼ぶ。"""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[Scheduler] 停止しました。")
