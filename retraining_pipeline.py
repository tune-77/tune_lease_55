"""retraining_pipeline.py — デフォルト予測モデルの自動・手動再学習パイプライン。

再学習対象はデフォルト判定モデルである。
  正常先（delinquent=0）: past_cases テーブル（成約/失注/検収完了）の財務特徴量
  高リスク先（delinquent=1）: excluded_grade_cases テーブル（格付9）の財務特徴量
RandomForest + LightGBM アンサンブルモデルであり、scoring_core.py / total_scorer.py 等の
既存スコアリングロジックには一切影響しない。
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from filelock import FileLock
    from filelock import Timeout as FileLockTimeout
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False
    FileLockTimeout = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "ind_medical", "ind_transport", "ind_construction", "ind_manufacturing", "ind_service",
    "sales_log", "bank_credit_log", "lease_credit_log",
    "op_profit", "ord_profit", "net_income", "machines", "other_assets", "rent",
    "gross_profit", "depreciation", "dep_expense", "rent_expense",
    # grade_4_6 / grade_watch / grade_none は意図的に除外する。
    # excluded_grade_cases(original_grade='9') を delinquent=1 のソースに使う設計上、
    # grade フラグを特徴量に含めると「格付9 → 全フラグ0」という循環参照が生まれ
    # AUC=1.00 のデータリーケージを引き起こす (circular reference)。
    "contracts",
]

RF_MODEL_FILE = "spread_predictor_v2.pkl"
LGBM_MODEL_FILE = "lgbm_model.pkl"
LOCK_FILENAME = ".retraining.lock"

_RETRAINING_LOG_DDL = """
CREATE TABLE IF NOT EXISTS retraining_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_by      TEXT    NOT NULL,
    status            TEXT    NOT NULL,
    records_used      INTEGER,
    new_auc           REAL,
    prev_auc          REAL,
    model_updated     INTEGER NOT NULL DEFAULT 0,
    rollback_reason   TEXT,
    error_message     TEXT,
    duration_seconds  REAL,
    started_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at      TEXT
);
"""


# ---------------------------------------------------------------------------
# TypedDict
# ---------------------------------------------------------------------------

class RetrainingResult(TypedDict):
    status: str                   # "success" | "skipped" | "rolled_back" | "error"
    records_used: int
    new_auc: Optional[float]
    prev_auc: Optional[float]
    model_updated: bool
    rollback_reason: Optional[str]
    error: Optional[str]
    duration_seconds: float


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _ensure_retraining_log(conn: sqlite3.Connection) -> None:
    conn.executescript(_RETRAINING_LOG_DDL)


def _log_to_db(
    db_path: str,
    triggered_by: str,
    result: RetrainingResult,
    started_at: str,
) -> None:
    """BR-425: 全実行結果を retraining_log に記録する。失敗しても外部に伝播しない。"""
    try:
        conn = sqlite3.connect(db_path)
        _ensure_retraining_log(conn)
        completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """
            INSERT INTO retraining_log
                (triggered_by, status, records_used, new_auc, prev_auc,
                 model_updated, rollback_reason, error_message, duration_seconds,
                 started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                triggered_by,
                result["status"],
                result.get("records_used", 0),
                result["new_auc"],
                result["prev_auc"],
                1 if result["model_updated"] else 0,
                result["rollback_reason"],
                result["error"],
                result["duration_seconds"],
                started_at,
                completed_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("[retraining_pipeline] log_to_db failed: %s", exc)


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _inputs_to_feature_row(inputs: dict) -> list[float]:
    """inputs dict から FEATURE_COLS 順の特徴量リストを返す。analysis_regression.py と同じ変換。
    grade フラグ (grade_4_6/grade_watch/grade_none) は FEATURE_COLS から除外済み。
    """
    major = str(inputs.get("industry_major") or "").strip()

    ind_medical       = 1.0 if ("医療" in major or "福祉" in major) else 0.0
    ind_transport     = 1.0 if "運輸" in major else 0.0
    ind_construction  = 1.0 if "建設" in major else 0.0
    ind_manufacturing = 1.0 if "製造" in major else 0.0
    ind_service       = 1.0 if any(x in major for x in ["卸売", "小売", "サービス"]) else 0.0

    nenshu       = max(_safe_float(inputs.get("nenshu")), 0.0)
    bank_credit  = max(_safe_float(inputs.get("bank_credit")), 0.0)
    lease_credit = max(_safe_float(inputs.get("lease_credit")), 0.0)

    return [
        ind_medical, ind_transport, ind_construction, ind_manufacturing, ind_service,
        math.log1p(nenshu), math.log1p(bank_credit), math.log1p(lease_credit),
        _safe_float(inputs.get("op_profit"))    / 1000.0,
        _safe_float(inputs.get("ord_profit"))   / 1000.0,
        _safe_float(inputs.get("net_income"))   / 1000.0,
        _safe_float(inputs.get("machines"))     / 1000.0,
        _safe_float(inputs.get("other_assets")) / 1000.0,
        _safe_float(inputs.get("rent"))         / 1000.0,
        _safe_float(inputs.get("gross_profit")) / 1000.0,
        _safe_float(inputs.get("depreciation")) / 1000.0,
        _safe_float(inputs.get("dep_expense"))  / 1000.0,
        _safe_float(inputs.get("rent_expense")) / 1000.0,
        _safe_float(inputs.get("contracts")),
    ]


def _fetch_training_data(db_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    デフォルト判定モデル用の学習データを構築する。

    データソース:
      正常先（delinquent=0）: past_cases（成約/失注/検収完了）の inputs 財務特徴量
      高リスク先（delinquent=1）: excluded_grade_cases（original_grade='9'）の inputs 財務特徴量

    特徴量は analysis_regression.py の COEFF_MAIN_KEYS と同一変換（対数・千円換算・フラグ）。
    NULL は列ごとの中央値で補完する。
    delinquent=1 が 5件未満の場合は (empty, empty, 0) を返してスキップを促す。
    """
    conn = sqlite3.connect(db_path)
    X_list: list[list[float]] = []
    y_list: list[int] = []

    # ── 正常先: past_cases ──
    try:
        normal_rows = conn.execute(
            "SELECT data FROM past_cases "
            "WHERE json_extract(data, '$.final_status') IN ('成約', '失注', '検収完了')"
        ).fetchall()
        for row in normal_rows:
            try:
                d = json.loads(row[0])
                inputs = d.get("inputs") or {}
                if not inputs:
                    continue
                X_list.append(_inputs_to_feature_row(inputs))
                y_list.append(0)
            except Exception:  # noqa: BLE001
                continue
        logger.info(
            "[retraining_pipeline] _fetch_training_data: normal(past_cases)=%d", len(X_list)
        )
    except sqlite3.OperationalError as exc:
        logger.error("[retraining_pipeline] _fetch_training_data: past_cases read failed: %s", exc)

    # ── 高リスク先: excluded_grade_cases（格付9） ──
    n_before_delinquent = len(X_list)
    try:
        delinquent_rows = conn.execute(
            "SELECT data FROM excluded_grade_cases WHERE original_grade = '9'"
        ).fetchall()
        for row in delinquent_rows:
            try:
                d = json.loads(row[0])
                inputs = d.get("inputs") or {}
                if not inputs:
                    continue
                X_list.append(_inputs_to_feature_row(inputs))
                y_list.append(1)
            except Exception:  # noqa: BLE001
                continue
        n_delinquent_loaded = len(X_list) - n_before_delinquent
        logger.info(
            "[retraining_pipeline] _fetch_training_data: delinquent(excluded_grade)=%d",
            n_delinquent_loaded,
        )
    except sqlite3.OperationalError as exc:
        logger.error(
            "[retraining_pipeline] _fetch_training_data: excluded_grade_cases read failed: %s", exc
        )

    conn.close()

    n = len(X_list)
    if n == 0:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0), 0

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    # 非有限値を列ごとの中央値で補完
    for col_idx in range(X.shape[1]):
        mask = ~np.isfinite(X[:, col_idx])
        if mask.any():
            finite_vals = X[~mask, col_idx]
            median_val = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0
            X[mask, col_idx] = median_val

    # FP-001: クラス均衡チェック
    n_delinquent = int(np.sum(y == 1))
    if n_delinquent < 5:
        logger.warning(
            "[retraining_pipeline] _fetch_training_data: delinquent cases=%d < 5, skip.",
            n_delinquent,
        )
        return np.empty((0, len(FEATURE_COLS))), np.empty(0), 0

    logger.info(
        "[retraining_pipeline] _fetch_training_data: total=%d, delinquent=%d, normal=%d",
        n, n_delinquent, n - n_delinquent,
    )
    return X, y, n


def _backup_models(model_dir: Path) -> bool:
    """BR-422: 再学習前にモデルファイルをバックアップする。失敗時は False を返す。"""
    pairs = [
        (RF_MODEL_FILE, RF_MODEL_FILE.replace(".pkl", ".bak.pkl")),
        (LGBM_MODEL_FILE, LGBM_MODEL_FILE.replace(".pkl", ".bak.pkl")),
    ]
    for src_name, bak_name in pairs:
        src = model_dir / src_name
        bak = model_dir / bak_name
        if src.exists():
            try:
                shutil.copy2(str(src), str(bak))
                logger.info(
                    "[retraining_pipeline] phase=backup src=%s bak=%s",
                    src_name, bak_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[retraining_pipeline] phase=backup_failed src=%s err=%s",
                    src_name, exc,
                )
                return False
    return True


def _restore_backup(model_dir: Path) -> None:
    """バックアップから元のモデルを復元する。"""
    pairs = [
        (RF_MODEL_FILE.replace(".pkl", ".bak.pkl"), RF_MODEL_FILE),
        (LGBM_MODEL_FILE.replace(".pkl", ".bak.pkl"), LGBM_MODEL_FILE),
    ]
    for bak_name, dst_name in pairs:
        bak = model_dir / bak_name
        dst = model_dir / dst_name
        if bak.exists():
            try:
                shutil.copy2(str(bak), str(dst))
                logger.info(
                    "[retraining_pipeline] phase=restore bak=%s dst=%s",
                    bak_name, dst_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[retraining_pipeline] phase=restore_failed bak=%s err=%s",
                    bak_name, exc,
                )


def _eval_prev_model(model_dir: Path, X_test: np.ndarray, y_test: np.ndarray) -> Optional[float]:
    """バックアップ RF モデルをテストセットで評価して AUC を返す。評価不可なら None。"""
    bak_path = model_dir / RF_MODEL_FILE.replace(".pkl", ".bak.pkl")
    if not bak_path.exists():
        return None
    try:
        prev_model = joblib.load(str(bak_path))
        if len(np.unique(y_test)) < 2:
            return None
        proba = prev_model.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, proba))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# パブリック API
# ---------------------------------------------------------------------------

def check_retraining_needed(
    min_records: int = 50,
    min_delinquent: int = 5,
    db_path: str = "data/lease_data.db",
) -> dict:
    """
    再学習が必要かどうかを判定する。FP-001 クラス均衡チェック込み。

    Returns:
        dict with keys:
          - needed (bool): True なら再学習を実行してよい
          - reason (str): スキップ/実行の理由
          - record_count (int): 学習対象レコード数
          - delinquent_count (int): デフォルト確認済み件数
    """
    try:
        conn = sqlite3.connect(db_path)
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM screening_records WHERE outcome IS NOT NULL"
        ).fetchone()

        # delinquent 件数（screening_outcomes 優先）
        delinquent_count = 0
        try:
            (delinquent_count,) = conn.execute(
                "SELECT COUNT(*) FROM screening_outcomes "
                "WHERE delinquent = 1 OR actual_status IN ('late_30','late_90','default')"
            ).fetchone()
        except sqlite3.OperationalError:
            # actual_status カラムが存在しない旧スキーマへのフォールバック
            (delinquent_count,) = conn.execute(
                "SELECT COUNT(*) FROM screening_outcomes WHERE delinquent = 1"
            ).fetchone()

        conn.close()

        if int(count) < min_records:
            return {
                "needed": False,
                "reason": f"レコード不足: {count}件 < 必要{min_records}件",
                "record_count": int(count),
                "delinquent_count": int(delinquent_count),
            }
        if int(delinquent_count) < min_delinquent:
            return {
                "needed": False,
                "reason": (
                    f"延滞/デフォルト実績不足: {delinquent_count}件 < 必要{min_delinquent}件。"
                    f" 支払状況を outcome_recorder で入力してください。"
                ),
                "record_count": int(count),
                "delinquent_count": int(delinquent_count),
            }
        return {
            "needed": True,
            "reason": f"再学習条件OK: {count}件 / delinquent={delinquent_count}件",
            "record_count": int(count),
            "delinquent_count": int(delinquent_count),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("[retraining_pipeline] check_retraining_needed error: %s", exc)
        return {"needed": False, "reason": str(exc), "record_count": 0, "delinquent_count": 0}


def run_retraining(
    triggered_by: str = "auto",
    min_records: int = 50,
    auc_threshold: float = 0.75,
    test_ratio: float = 0.2,
    model_dir: str = "models/",
    db_path: str = "data/lease_data.db",
    dry_run: bool = False,
) -> RetrainingResult:
    """
    screening_records テーブルのデータで RF/LGBM スコア集約メタモデルを再学習する。

    再学習対象はスコア集約メタモデル（total_score / asset_score / tenant_score /
    q_risk_score / competitor_pressure_score → delinquent 判定）であり、
    scoring_core.py / total_scorer.py / asset_scorer.py 等の既存スコアリングロジックには
    一切影響しない。

    フェイルセーフ: AUC が auc_threshold 未満なら旧モデルを維持（ロールバック）。
    例外を外部に伝播させない。
    dry_run=True の場合は評価のみ行いモデルファイルを更新しない（バックアップもスキップ）。
    """
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.monotonic()

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    def _make_result(**kwargs) -> RetrainingResult:
        base: RetrainingResult = {
            "status": "error",
            "records_used": 0,
            "new_auc": None,
            "prev_auc": None,
            "model_updated": False,
            "rollback_reason": None,
            "error": None,
            "duration_seconds": time.monotonic() - t0,
        }
        base.update(kwargs)  # type: ignore[typeddict-item]
        base["duration_seconds"] = time.monotonic() - t0
        return base

    # BR-426: 並行実行防止
    lock = None
    if _FILELOCK_AVAILABLE:
        lock_path = str(model_dir_path / LOCK_FILENAME)
        lock = FileLock(lock_path, timeout=0)
        try:
            lock.acquire()
        except FileLockTimeout:
            result = _make_result(status="skipped", error="retraining already in progress")
            _log_to_db(db_path, triggered_by, result, started_at)
            return result

    try:
        logger.info(
            "[retraining_pipeline] phase=start triggered_by=%s dry_run=%s",
            triggered_by, dry_run,
        )

        # レコード取得 + BR-421 件数チェック
        logger.info("[retraining_pipeline] phase=record_check")
        try:
            X, y, n_records = _fetch_training_data(db_path)
        except Exception as exc:  # noqa: BLE001
            result = _make_result(status="error", error=f"data fetch failed: {exc}")
            _log_to_db(db_path, triggered_by, result, started_at)
            return result

        if n_records < min_records:
            logger.info(
                "[retraining_pipeline] phase=skipped records=%d min=%d",
                n_records, min_records,
            )
            result = _make_result(status="skipped", records_used=n_records)
            _log_to_db(db_path, triggered_by, result, started_at)
            return result

        # BR-422: バックアップ（dry_run=True の場合はスキップ: W10修正済み）
        if not dry_run:
            logger.info("[retraining_pipeline] phase=backup_start")
            if not _backup_models(model_dir_path):
                result = _make_result(
                    status="error",
                    records_used=n_records,
                    error="backup failed; retraining aborted to protect existing models",
                )
                _log_to_db(db_path, triggered_by, result, started_at)
                return result
            logger.info("[retraining_pipeline] phase=backup_complete")

        # Train/Test split
        stratify = y if len(np.unique(y)) > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=42, stratify=stratify
            )
        except Exception as exc:  # noqa: BLE001
            if not dry_run:
                _restore_backup(model_dir_path)
            result = _make_result(
                status="rolled_back" if not dry_run else "error",
                records_used=n_records,
                rollback_reason=f"train_test_split failed: {exc}" if not dry_run else None,
                error=str(exc) if dry_run else None,
            )
            _log_to_db(db_path, triggered_by, result, started_at)
            return result

        # 旧モデルの AUC（バックアップから評価）
        prev_auc: Optional[float] = None
        if not dry_run:
            prev_auc = _eval_prev_model(model_dir_path, X_test, y_test)

        # 学習
        logger.info("[retraining_pipeline] phase=train_start records=%d", n_records)
        new_auc: Optional[float] = None
        rf = None
        lgbm = None

        try:
            rf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
            )
            rf.fit(X_train, y_train)

            ensemble_proba = rf.predict_proba(X_test)[:, 1]

            # LGBM はインストール済みの場合にアンサンブル（モック等で失敗時は RF のみ）
            if _LGBM_AVAILABLE:
                try:
                    lgbm = LGBMClassifier(
                        n_estimators=100, random_state=42, verbose=-1,
                        class_weight="balanced",
                    )
                    lgbm.fit(X_train, y_train)
                    lgbm_proba = np.array(
                        lgbm.predict_proba(X_test)[:, 1], dtype=float
                    )
                    ensemble_proba = (ensemble_proba + lgbm_proba) / 2.0
                except Exception:  # noqa: BLE001
                    lgbm = None

            # W9修正済み: 2値分類なので average パラメータなし
            if len(np.unique(y_test)) >= 2:
                new_auc = float(roc_auc_score(y_test, ensemble_proba))
            else:
                new_auc = 0.5  # クラスが 1 種類しかない場合のデフォルト

            logger.info(
                "[retraining_pipeline] phase=train_complete new_auc=%.4f", new_auc
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("[retraining_pipeline] phase=train_error err=%s", exc)
            if not dry_run:
                _restore_backup(model_dir_path)
            result = _make_result(
                status="rolled_back" if not dry_run else "error",
                records_used=n_records,
                prev_auc=prev_auc,
                rollback_reason=f"training exception: {exc}" if not dry_run else None,
                error=str(exc) if dry_run else None,
            )
            _log_to_db(db_path, triggered_by, result, started_at)
            return result

        # BR-423: AUC 閾値チェック
        if new_auc is not None and new_auc < auc_threshold:
            logger.info(
                "[retraining_pipeline] phase=rollback new_auc=%.4f threshold=%.4f",
                new_auc, auc_threshold,
            )
            if not dry_run:
                _restore_backup(model_dir_path)
            rollback_reason = f"new_auc={new_auc:.4f} < auc_threshold={auc_threshold}"
            result = _make_result(
                status="rolled_back",
                records_used=n_records,
                new_auc=new_auc,
                prev_auc=prev_auc,
                rollback_reason=rollback_reason,
            )
            _log_to_db(db_path, triggered_by, result, started_at)
            return result

        # モデル保存
        model_updated = False
        if not dry_run:
            logger.info("[retraining_pipeline] phase=save_start")
            try:
                joblib.dump(rf, str(model_dir_path / RF_MODEL_FILE))
                if lgbm is not None:
                    joblib.dump(lgbm, str(model_dir_path / LGBM_MODEL_FILE))
                model_updated = True
                logger.info("[retraining_pipeline] phase=save_complete")
            except Exception as exc:  # noqa: BLE001
                logger.error("[retraining_pipeline] phase=save_error err=%s", exc)
                _restore_backup(model_dir_path)
                result = _make_result(
                    status="rolled_back",
                    records_used=n_records,
                    new_auc=new_auc,
                    prev_auc=prev_auc,
                    rollback_reason=f"model save failed: {exc}",
                )
                _log_to_db(db_path, triggered_by, result, started_at)
                return result

        logger.info(
            "[retraining_pipeline] phase=complete model_updated=%s new_auc=%.4f",
            model_updated, new_auc,
        )
        result = _make_result(
            status="success",
            records_used=n_records,
            new_auc=new_auc,
            prev_auc=prev_auc,
            model_updated=model_updated,
        )
        _log_to_db(db_path, triggered_by, result, started_at)
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error("[retraining_pipeline] phase=unexpected_error err=%s", exc)
        if not dry_run:
            try:
                _restore_backup(model_dir_path)
            except Exception:  # noqa: BLE001
                pass
        result = _make_result(status="error", error=f"unexpected: {exc}")
        _log_to_db(db_path, triggered_by, result, started_at)
        return result

    finally:
        if lock is not None and _FILELOCK_AVAILABLE:
            try:
                lock.release()
            except Exception:  # noqa: BLE001
                pass
