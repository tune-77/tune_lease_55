"""fluid_pipeline.py — 流体化イベントバス（Phase 1-A）

既存モジュールを「接続された川」にするオーケストレーター。
新しいロジックは一切書かず、既存モジュールを呼ぶだけ。

設計原則:
  - 全ての処理は try/except で囲い、例外を外部に伝播させない
  - 再学習はサブプロセス（非同期）で起動し、Streamlit セッションをブロックしない
  - FileLock（.retraining.lock）を共有して並行実行を防ぐ
  - 全イベントは data/fluid_pipeline_log.jsonl に記録される

使い方:
    from fluid_pipeline import FluidPipeline
    FluidPipeline().on_case_registered()        # 案件登録後に呼ぶ
    FluidPipeline().on_outcome_registered(...)  # 支払状況登録後に呼ぶ
    FluidPipeline().status()                    # 現在状態を取得
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent
DB_PATH = str(_DIR / "data" / "lease_data.db")
LOG_FILE = str(_DIR / "data" / "fluid_pipeline_log.jsonl")
LOCK_FILE = str(_DIR / "models" / ".retraining.lock")


class FluidPipeline:
    """既存モジュールを繋ぐ軽量イベントバス。"""

    # ──────────────────────────────────────────────────────────────
    # パブリック API
    # ──────────────────────────────────────────────────────────────

    def on_case_registered(self) -> None:
        """案件が審査・登録された後に呼ぶ。

        screening_records への記録後に score_calculation.py から呼ばれる。
        """
        self._log("on_case_registered", {})

        # 1. コンセプトドリフト検知（既存 macro_drift_monitor.py）
        drift = self._check_drift()
        if drift.get("is_drift"):
            self._log("drift_detected", drift)
            self._notify_slack(
                f"⚠️ [FluidPipeline] コンセプトドリフト検知\n{drift.get('message', '')}"
            )

        # 2. 再学習判定 → 非同期起動
        retrain_check = self._check_retraining_needed()
        if retrain_check.get("needed"):
            self._log("retraining_triggered", retrain_check)
            self._spawn_retraining("fluid_pipeline_auto")

    def on_outcome_registered(self, case_id: str, status: str) -> None:
        """支払状況（延滞/デフォルト）が登録された後に呼ぶ。

        outcome_recorder.py から呼ばれる。
        """
        self._log("on_outcome_registered", {"case_id": case_id, "status": status})

        # 延滞/デフォルト登録時は即時再学習チェック
        if status in ("late_30", "late_90", "default"):
            retrain_check = self._check_retraining_needed()
            if retrain_check.get("needed"):
                self._log("retraining_triggered_by_outcome", retrain_check)
                self._spawn_retraining("fluid_pipeline_outcome")
            else:
                self._log("retraining_waiting", {
                    "reason": retrain_check.get("reason", ""),
                    "delinquent_count": retrain_check.get("delinquent_count", 0),
                })

    def on_model_updated(self, metrics: dict) -> None:
        """モデル更新完了後に呼ぶ（retraining_pipeline.py から）。

        PDCA 反省を実行し、軍師プロンプトを更新する。
        """
        self._log("on_model_updated", metrics)

        # PDCA 反省（既存 llm_pdca_reflection.py）— 非同期
        try:
            self._spawn_pdca_reflection()
            self._log("pdca_reflection_triggered", {})
        except Exception as e:
            logger.warning("[FluidPipeline] PDCA reflection spawn failed: %s", e)

    def status(self) -> dict:
        """現在の状態サマリーを返す。Streamlit の admin 画面から呼ぶ。"""
        retrain = self._check_retraining_needed()
        drift = self._check_drift()
        last_events = self._read_recent_log(5)
        return {
            "retraining": retrain,
            "drift": drift,
            "last_events": last_events,
        }

    # ──────────────────────────────────────────────────────────────
    # 既存モジュール呼び出し（全て try/except）
    # ──────────────────────────────────────────────────────────────

    def _check_drift(self) -> dict:
        try:
            from macro_drift_monitor import check_concept_drift
            return check_concept_drift() or {"is_drift": False}
        except Exception as e:
            logger.debug("[FluidPipeline] _check_drift error: %s", e)
            return {"is_drift": False, "message": str(e)}

    def _check_retraining_needed(self) -> dict:
        try:
            from retraining_pipeline import check_retraining_needed
            result = check_retraining_needed(db_path=DB_PATH)
            # 後方互換: bool が返ってきた場合
            if isinstance(result, bool):
                return {"needed": result, "reason": "", "delinquent_count": 0}
            return result
        except Exception as e:
            logger.debug("[FluidPipeline] _check_retraining_needed error: %s", e)
            return {"needed": False, "reason": str(e), "delinquent_count": 0}

    def _spawn_retraining(self, triggered_by: str = "fluid_pipeline") -> bool:
        """再学習をサブプロセスで非同期起動する。ロック中はスキップ。"""
        lock_path = Path(LOCK_FILE)
        if lock_path.exists():
            logger.info("[FluidPipeline] retraining lock exists, skipping spawn")
            return False

        script = str(_DIR / "_fluid_retrain_worker.py")
        if not Path(script).exists():
            self._write_retrain_worker(script)

        try:
            subprocess.Popen(
                [sys.executable, script, "--triggered-by", triggered_by],
                cwd=str(_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info("[FluidPipeline] retraining spawned: %s", triggered_by)
            return True
        except Exception as e:
            logger.error("[FluidPipeline] retraining spawn failed: %s", e)
            return False

    def _spawn_pdca_reflection(self) -> bool:
        """PDCA 反省をサブプロセスで非同期起動する。"""
        try:
            subprocess.Popen(
                [sys.executable, "-c",
                 "import sys; sys.path.insert(0, '.'); "
                 "from llm_pdca_reflection import run_monthly_pdca_reflection; "
                 "run_monthly_pdca_reflection(force=True)"],
                cwd=str(_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception as e:
            logger.error("[FluidPipeline] PDCA spawn failed: %s", e)
            return False

    def _notify_slack(self, message: str) -> None:
        """Slack 通知（既存 slack_notify.py）。失敗しても無視。"""
        try:
            from slack_notify import send_slack_message
            send_slack_message(message)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # ログ管理
    # ──────────────────────────────────────────────────────────────

    def _log(self, event: str, data: dict) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "event": event,
            **data,
        }
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _read_recent_log(self, n: int = 10) -> list[dict]:
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            result = []
            for line in reversed(lines[-n * 3:]):
                try:
                    result.append(json.loads(line))
                except Exception:
                    pass
            return result[:n]
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────
    # ワーカースクリプト自動生成
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _write_retrain_worker(path: str) -> None:
        """再学習用ワーカースクリプトを生成する（初回のみ）。"""
        code = '''\
"""_fluid_retrain_worker.py — FluidPipeline から spawn される再学習ワーカー。"""
import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--triggered-by", default="fluid_pipeline")
args = parser.parse_args()

try:
    from retraining_pipeline import run_retraining
    result = run_retraining(
        triggered_by=args.triggered_by,
        db_path="data/lease_data.db",
        model_dir="models/",
    )
    # モデル更新完了通知
    if result.get("model_updated"):
        from fluid_pipeline import FluidPipeline
        FluidPipeline().on_model_updated(result)
    print(json.dumps(result, ensure_ascii=False))
except Exception as e:
    print(f"[_fluid_retrain_worker] error: {e}", file=sys.stderr)
'''
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
