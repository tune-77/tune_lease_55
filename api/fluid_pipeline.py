"""fluid_pipeline.py — ドリフト検知・再学習・PDCA反省のバックグラウンド自動制御。

新規ファイル。既存ファイルは一切変更しない。
FastAPI の審査フローを止めない非同期・バックグラウンド実装。

パイプライン順序:
  1. macro_drift_monitor.check_concept_drift() でドリフト判定
  2. ドリフトあり → retraining_pipeline.run_retraining() を実行
  3. 再学習完了（または不要）→ llm_pdca_reflection.run_monthly_pdca_reflection() を実行
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 状態管理（モジュールレベルで保持）
# ---------------------------------------------------------------------------

_status: dict[str, Any] = {
    "state": "idle",               # idle | running | completed | error
    "last_triggered_at": None,
    "last_completed_at": None,
    "triggered_by": None,
    "drift_detected": None,
    "retraining_result": None,
    "pdca_result": None,
    "error": None,
}
_status_lock = threading.Lock()


def _update_status(**kwargs: Any) -> None:
    with _status_lock:
        _status.update(kwargs)


# ---------------------------------------------------------------------------
# パイプライン本体（同期、バックグラウンドスレッドで実行）
# ---------------------------------------------------------------------------

def _run_pipeline_sync(triggered_by: str) -> None:
    """ドリフト検知→再学習→PDCA反省を順番に実行する。例外を外部に伝播させない。"""
    _update_status(state="running", error=None)

    # リポジトリルートを sys.path に追加（retraining_pipeline 等のインポート用）
    _api_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_api_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    try:
        # ── Step 1: ドリフト検知 ─────────────────────────────────────────────
        try:
            from macro_drift_monitor import check_concept_drift
            drift_result = check_concept_drift()
        except Exception as exc:
            logger.error("[fluid_pipeline] drift check failed: %s", exc)
            drift_result = {"is_drift": False, "message": f"drift check error: {exc}"}

        is_drift = bool(drift_result.get("is_drift", False))
        _update_status(drift_detected=is_drift)
        logger.info(
            "[fluid_pipeline] drift_check is_drift=%s msg=%s",
            is_drift,
            drift_result.get("message", ""),
        )

        # ── Step 2: ドリフトあり → 再学習 ───────────────────────────────────
        retraining_result: Optional[dict] = None
        if is_drift:
            logger.info("[fluid_pipeline] drift detected — starting retraining")
            try:
                from retraining_pipeline import run_retraining
                retraining_result = run_retraining(
                    triggered_by=f"fluid_pipeline:{triggered_by}",
                )
                _update_status(retraining_result=retraining_result)
                logger.info(
                    "[fluid_pipeline] retraining status=%s",
                    retraining_result.get("status"),
                )
            except Exception as exc:
                logger.error("[fluid_pipeline] retraining failed: %s", exc)
                retraining_result = {"status": "error", "error": str(exc)}
                _update_status(retraining_result=retraining_result)

        # ── Step 3: PDCA反省（再学習成功／スキップ時、またはドリフトなし時も実行）──
        should_run_pdca = (
            retraining_result is None
            or retraining_result.get("status") in ("success", "skipped", "rolled_back")
        )
        pdca_result: Optional[dict] = None
        if should_run_pdca:
            logger.info("[fluid_pipeline] starting pdca reflection")
            try:
                from llm_pdca_reflection import run_monthly_pdca_reflection
                pdca_result = run_monthly_pdca_reflection(force=False)
                _update_status(pdca_result=pdca_result)
                logger.info(
                    "[fluid_pipeline] pdca status=%s",
                    (pdca_result or {}).get("status"),
                )
            except Exception as exc:
                logger.error("[fluid_pipeline] pdca reflection failed: %s", exc)
                pdca_result = {"status": "error", "error": str(exc)}
                _update_status(pdca_result=pdca_result)

        _update_status(
            state="completed",
            last_completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("[fluid_pipeline] pipeline completed")

    except Exception as exc:
        logger.error("[fluid_pipeline] unexpected error: %s", exc)
        _update_status(state="error", error=str(exc))


# ---------------------------------------------------------------------------
# パブリック API
# ---------------------------------------------------------------------------

def trigger_fluid_pipeline(triggered_by: str = "manual") -> dict:
    """バックグラウンドでパイプラインを起動する。

    実行中なら skipped を返す（二重起動防止）。
    FastAPI の request/response サイクルをブロックしない。
    """
    with _status_lock:
        if _status["state"] == "running":
            return {"status": "skipped", "reason": "already running"}
        _status["state"] = "running"
        _status["last_triggered_at"] = datetime.now(timezone.utc).isoformat()
        _status["triggered_by"] = triggered_by

    thread = threading.Thread(
        target=_run_pipeline_sync,
        args=(triggered_by,),
        daemon=True,
        name="fluid-pipeline",
    )
    thread.start()
    logger.info("[fluid_pipeline] triggered by=%s thread=%s", triggered_by, thread.name)
    return {
        "status": "triggered",
        "triggered_by": triggered_by,
        "started_at": _status["last_triggered_at"],
    }


def get_fluid_status() -> dict:
    """現在のパイプライン状態スナップショットを返す。"""
    with _status_lock:
        return dict(_status)
