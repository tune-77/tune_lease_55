"""
APScheduler による定期バッチスケジューラ。
毎日 02:00 に知識結晶化バッチを実行する。
毎日 03:00 に紫苑フィードバック傾向ループを実行し、提案を改善ログへ投入する。
毎日 03:30 に紫苑画面利用ループを実行し、提案を改善ログへ投入する。
毎日 04:00 に記憶減衰バッチを実行する（REV-219）。
毎日 04:05 に無交流ペナルティを適用する（REV-220）。
毎日 04:10 にチャット会話要約キャッシュを再構築する。
実行結果は scheduler_job_runs.jsonl に記録し、起動時に当日分の取りこぼしを補完する。
"""
from __future__ import annotations

import datetime as dt
import fcntl
import functools
import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_scheduler: BackgroundScheduler | None = None
_JOB_RUNS_LOCK = threading.Lock()


def run_crystallization_batch() -> dict:
    """
    知識結晶化バッチのメイン処理。
    1. 外れ値/意見割れ案件を抽出
    2. Gemini でパターンを言語化
    3. Obsidian に書き出す

    Returns:
        {"status": str, "cases_found": int, "file": str | None}
    """
    # REV-585: 28日で6件・題材の重複ばかりで記憶にも審査にも効いていないため既定で停止（Gemini 呼び出しも止まる）。
    # 再開は CRYSTALLIZATION_ENABLED=1。既存の Generated/ ノートは消さない。
    if os.environ.get("CRYSTALLIZATION_ENABLED", "").strip().lower() not in {"1", "true", "on", "yes"}:
        logger.info("[Crystallizer] CRYSTALLIZATION_ENABLED が未設定のため停止中（スキップ）")
        return {"status": "disabled", "cases_found": 0, "file": None}

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

        if fpath is None:
            logger.info("[Crystallizer] 過去7日以内に同一案件セット済み。重複スキップ。")
            return {"status": "skipped_duplicate", "cases_found": len(cases), "file": None}

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
            for key, label in (
                ("hypothesis", "仮説"),
                ("evidence", "根拠"),
                ("proposed_change", "変更案"),
                ("success_metric", "成功指標"),
                ("verification_plan", "検証方法"),
                ("risk", "リスク"),
                ("pattern", "パターン"),
                ("suggestion", "提案"),
                ("reason", "理由"),
            ):
                value = str(p.get(key) or "").strip()
                if value:
                    body_parts.append(f"## {label}\n{value}")
            entry = {
                "event_id": str(uuid.uuid4()),
                "ts": ts_now,
                "title": title,
                "body": "\n\n".join(body_parts),
                "surface": "shion_self_proposal",
                "source": source,
                "proposed_by": "shion",
                "target_page": str(p.get("target_page") or ""),
                "hypothesis": str(p.get("hypothesis") or ""),
                "evidence": str(p.get("evidence") or ""),
                "proposed_change": str(p.get("proposed_change") or ""),
                "success_metric": str(p.get("success_metric") or ""),
                "verification_plan": str(p.get("verification_plan") or ""),
                "risk": str(p.get("risk") or ""),
                "priority": str(p.get("priority") or ""),
                "status": str(p.get("status") or ""),
                "proposal_schema": str(p.get("proposal_schema") or ""),
                "human_decision_status": str(p.get("human_decision_status") or p.get("status") or ""),
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
        reason = ""
        generation_status = "ok"
        if not result.get("generated"):
            reason = str(result.get("reason", ""))
            generation_status = "error" if reason.startswith("Gemini生成に失敗") else "no_proposals"
            log = logger.warning if generation_status == "error" else logger.info
            log(f"[ShionFeedbackLoop] 提案なし: {reason}")
        else:
            proposals = result.get("proposals", [])
            pushed = _push_proposals_to_improvement_log(proposals, source="feedback_pattern_loop")
            logger.info(f"[ShionFeedbackLoop] 提案{len(proposals)}件 / 改善ログ投入{pushed}件")

        # 2. PDCA評価（B: ループを閉じる — 採用済み提案の効果検証）
        pdca = evaluate_proposal_impact()
        logger.info(f"[ShionFeedbackLoop] PDCA評価: {pdca.get('evaluated', 0)}件")

        return {
            "status": generation_status,
            "reason": reason,
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
            reason = str(result.get("reason", ""))
            status = "error" if reason.startswith("Gemini生成に失敗") else "no_proposals"
            log = logger.warning if status == "error" else logger.info
            log(f"[ShionUsageLoop] 提案なし: {reason}")
            return {"status": status, "reason": reason}

        proposals = result.get("proposals", [])
        pushed = _push_proposals_to_improvement_log(proposals, source="usage_loop")
        logger.info(f"[ShionUsageLoop] 完了。提案{len(proposals)}件 / 改善ログ投入{pushed}件")
        return {"status": "ok", "proposals_generated": len(proposals), "pushed_to_log": pushed}

    except Exception as e:
        logger.error(f"[ShionUsageLoop] エラー: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


def run_shion_memory_decay() -> dict:
    """記憶減衰バッチ（毎日 04:00 / REV-219）。"""
    logger.info("[MemoryDecay] バッチ開始")
    try:
        from api.shion_memory_decay import run_memory_decay_batch
        return run_memory_decay_batch()
    except Exception as e:
        logger.error(f"[MemoryDecay] エラー: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


def run_shion_inactivity_decay() -> dict:
    """無交流ペナルティ適用（毎日 04:05 / REV-220）。"""
    logger.info("[Relationship] 無交流ペナルティチェック開始")
    try:
        from api.shion_relationship import apply_inactivity_decay
        state = apply_inactivity_decay()
        return {"status": "ok", "score": state.get("score"), "trend": state.get("trend")}
    except Exception as e:
        logger.error(f"[Relationship] エラー: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


def run_chat_summary_refresh() -> dict:
    """チャット会話要約キャッシュの再構築バッチ（毎日 04:10）。

    call_gemini_chat系のcontext構築が使う「直近ウィンドウ外の古い会話」の
    要約（chat_memory.get_summary）を、メッセージ数が一定以上増えたユーザー
    だけ日次バッチでのみ再構築する。同期経路（/api/chat）はキャッシュを
    読むだけにして、構築コストをレイテンシに乗せない。
    """
    logger.info("[ChatSummaryRefresh] バッチ開始")
    try:
        from api.chat_memory import refresh_stale_chat_summaries
        return refresh_stale_chat_summaries()
    except Exception as e:
        logger.error(f"[ChatSummaryRefresh] エラー: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


_DAILY_JOBS = [
    ("crystallization_daily", run_crystallization_batch, 2, 0, "知識結晶化バッチ（毎日02:00）"),
    ("shion_feedback_loop_daily", run_shion_feedback_loop, 3, 0, "紫苑フィードバック傾向ループ（毎日03:00）"),
    ("shion_usage_loop_daily", run_shion_usage_loop, 3, 30, "紫苑画面利用ループ（毎日03:30）"),
    ("shion_memory_decay_daily", run_shion_memory_decay, 4, 0, "記憶減衰バッチ（毎日04:00）"),
    ("shion_inactivity_decay_daily", run_shion_inactivity_decay, 4, 5, "無交流ペナルティ（毎日04:05）"),
    ("chat_summary_refresh_daily", run_chat_summary_refresh, 4, 10, "チャット会話要約キャッシュ再構築（毎日04:10）"),
]


def _job_runs_path() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))
    return data_dir / "scheduler_job_runs.jsonl"


def _read_job_records(path: Path) -> list[dict[str, Any]]:
    """実行台帳を厳格に読む。1行でも壊れていれば安全のため補完・実行を止める。"""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("ts") or not record.get("job_id"):
                raise ValueError("required fields are missing")
            dt.datetime.fromisoformat(str(record["ts"]))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"実行台帳の{line_no}行目が不正です: {exc}") from exc
        records.append(record)
    return records


def _append_job_record(record: dict[str, Any]) -> None:
    path = _job_runs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@contextmanager
def _job_runs_file_lock(path: Path):
    """複数プロセスが同時起動してもclaimの確認と追記を直列化する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _claim_daily_job(job_id: str, now: dt.datetime | None = None) -> tuple[bool, str]:
    """副作用の前に当日分をclaimし、同日中の二重実行を防ぐ。"""
    current = now or dt.datetime.now()
    path = _job_runs_path()
    with _JOB_RUNS_LOCK:
        try:
            with _job_runs_file_lock(path):
                records = _read_job_records(path)
                for record in records:
                    recorded_at = dt.datetime.fromisoformat(str(record["ts"]))
                    if recorded_at.date() == current.date() and str(record["job_id"]) == job_id:
                        return False, "already_claimed"
                _append_job_record(
                    {
                        "ts": current.isoformat(timespec="seconds"),
                        "job_id": job_id,
                        "status": "running",
                        "result": {},
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning(f"[Scheduler] 実行台帳を確認できないため実行中止: {exc}")
            return False, "ledger_unavailable"
    return True, ""


def _run_daily_job_once(job_id: str, func: Any) -> dict[str, Any]:
    claimed, reason = _claim_daily_job(job_id)
    if not claimed:
        status = "skipped_duplicate" if reason == "already_claimed" else "error"
        return {"status": status, "reason": reason}
    return func()


def _scheduler_job_listener(event: Any) -> None:
    if getattr(event, "code", None) == EVENT_JOB_MISSED:
        logger.warning(f"[Scheduler] 実行結果: job_id={event.job_id} status=missed")
        return

    result = event.retval if isinstance(getattr(event, "retval", None), dict) else {}
    exception = getattr(event, "exception", None)
    if getattr(event, "code", None) == EVENT_JOB_ERROR:
        status = "exception"
        recorded_result: Any = str(exception)
    else:
        status = str(result.get("status") or "ok")
        recorded_result = result

    record = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "job_id": str(event.job_id).removesuffix("_catchup"),
        "status": status,
        "result": recorded_result,
    }
    try:
        with _JOB_RUNS_LOCK:
            with _job_runs_file_lock(_job_runs_path()):
                _append_job_record(record)
    except OSError as exc:
        logger.warning(f"[Scheduler] 実行記録の書込みに失敗: {exc}")

    detail = result.get("reason") or result.get("detail") or ""
    log = logger.info if status in ("ok", "no_anomalies", "skipped_duplicate", "no_proposals") else logger.warning
    log(f"[Scheduler] 実行結果: job_id={record['job_id']} status={status} detail={detail}")


def _schedule_catchup_jobs(scheduler: BackgroundScheduler, now: dt.datetime) -> list[str]:
    runs_path = _job_runs_path()
    if not runs_path.exists():
        logger.info("[Scheduler] 初回のため取りこぼし補完をスキップ")
        return []

    try:
        records = _read_job_records(runs_path)
    except (OSError, ValueError) as exc:
        logger.warning(f"[Scheduler] 実行記録の読込みに失敗: {exc}")
        return []
    completed_today: set[str] = set()
    for record in records:
        recorded_at = dt.datetime.fromisoformat(str(record["ts"]))
        if recorded_at.date() == now.date():
            completed_today.add(str(record.get("job_id") or ""))

    scheduled: list[str] = []
    for job_id, func, hour, minute, name in _DAILY_JOBS:
        scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if scheduled_time > now or job_id in completed_today:
            continue
        run_date = now + dt.timedelta(seconds=90 + 30 * len(scheduled))
        catchup_id = f"{job_id}_catchup"
        scheduler.add_job(
            functools.partial(_run_daily_job_once, job_id, func),
            trigger="date",
            run_date=run_date,
            id=catchup_id,
            name=f"{name}（取りこぼし補完）",
            replace_existing=True,
            misfire_grace_time=6 * 3600,
            coalesce=True,
        )
        scheduled.append(job_id)
        logger.warning(f"[Scheduler] 取りこぼし補完: {job_id} を {run_date} に実行")
    return scheduled


def start_scheduler() -> BackgroundScheduler:
    """
    APScheduler を起動して定期バッチを登録する。
    FastAPI の startup イベントから呼ぶ。
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="Asia/Tokyo")
    _scheduler.add_listener(_scheduler_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    for job_id, func, hour, minute, name in _DAILY_JOBS:
        # Mac スリープ中に予定時刻を過ぎても復帰後に1回だけ実行する
        _scheduler.add_job(
            functools.partial(_run_daily_job_once, job_id, func),
            trigger=CronTrigger(hour=hour, minute=minute, timezone="Asia/Tokyo"),
            id=job_id,
            name=name,
            replace_existing=True,
            misfire_grace_time=6 * 3600,
            coalesce=True,
        )

    _schedule_catchup_jobs(_scheduler, dt.datetime.now())

    _scheduler.start()
    logger.info(
        "[Scheduler] 起動完了。"
        "毎日 02:00 JST に結晶化バッチ、"
        "03:00 に紫苑フィードバックループ、"
        "03:30 に紫苑利用ループ、"
        "04:00 に記憶減衰バッチ、"
        "04:05 に無交流ペナルティ、"
        "04:10 にチャット会話要約キャッシュ再構築を実行します。"
    )
    return _scheduler


def stop_scheduler() -> None:
    """FastAPI の shutdown イベントから呼ぶ。"""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[Scheduler] 停止しました。")
