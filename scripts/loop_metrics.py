#!/usr/bin/env python3
"""Build a read-only loop engineering health report from existing artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary as build_prompt_summary, load_jsonl
from coeff_definitions import COEFFS

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_LATEST_REPORT = REPORTS_DIR / "latest.json"
DEFAULT_RECURSIVE_REPORT = REPORTS_DIR / "recursive_self_improvement_latest.json"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "loop_engineering_latest.json"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "loop_engineering_latest.md"
DEFAULT_COEFF_OVERRIDES = REPO_ROOT / "data" / "coeff_overrides.json"
DEFAULT_COEFF_AUTO = REPO_ROOT / "data" / "coeff_auto.json"
DEFAULT_PREFLIGHT_RETRY_STATE = REPO_ROOT / ".claude" / "state" / "preflight_retries.json"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_PDCA_LOG = DEFAULT_DATA_DIR / "shion_self_pdca_log.jsonl"
# 4つの結果ループが提案を永続化する jsonl（出典: api/*_loop.py）
_OUTCOME_LOOP_FILES = {
    "outcome_drift": "outcome_drift_proposals.jsonl",
    "feedback_pattern": "feedback_pattern_proposals.jsonl",
    "judgment_divergence": "judgment_divergence_proposals.jsonl",
    "knowledge_gap": "knowledge_gap_proposals.jsonl",
}
DEFAULT_MODEL_PATHS = (
    REPO_ROOT / "data" / "ml_rf_v4.pkl",
    REPO_ROOT / "data" / "lgb_main_model.joblib",
    REPO_ROOT / "data" / "lgb_qual_model.joblib",
    REPO_ROOT / "models" / "lgbm_model.pkl",
)

_REQUIRED_COEFF_KEYS = (
    "intercept",
    "sales_log",
    "op_profit",
    "ord_profit",
    "net_income",
    "bank_credit_log",
    "lease_credit_log",
)
_EXPECTED_MODEL_KEYS = (
    "全体_既存先",
    "全体_新規先",
    "医療_既存先",
    "運送業_既存先",
    "サービス業_既存先",
    "製造業_既存先",
)
_AUTO_WEIGHT_KEYS = (
    "_auto_weight_borrower",
    "_auto_weight_asset",
    "_auto_weight_quant",
    "_auto_weight_qual",
    "_auto_blend_w_main",
    "_auto_blend_w_bench",
    "_auto_blend_w_ind",
)


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    return (data, True) if isinstance(data, dict) else ({}, False)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_json_any(path: Path) -> tuple[Any, bool, str]:
    if not path.exists():
        return None, False, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), True, ""
    except Exception as exc:
        return None, False, str(exc)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _effective_coeffs_for_health(overrides: dict[str, Any], key: str) -> dict[str, Any]:
    base_key = key if key in COEFFS else key.replace("_既存先", "").replace("_新規先", "")
    base = dict(COEFFS.get(base_key, COEFFS["全体_既存先"]))
    if isinstance(overrides.get(base_key), dict):
        base.update(overrides[base_key])
    if isinstance(overrides.get(key), dict):
        base.update(overrides[key])
    return base


def _check_model_load(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size <= 0:
        return False, "empty"
    joblib_cmd = [
        sys.executable,
        "-c",
        "import joblib, sys; joblib.load(sys.argv[1])",
        str(path),
    ]
    try:
        result = subprocess.run(
            joblib_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        joblib_error = (result.stderr or "").strip().splitlines()[-1:] or [f"exit={result.returncode}"]
    except Exception as exc:
        joblib_error = [str(exc)]

    pickle_cmd = [
        sys.executable,
        "-c",
        "import pickle, sys; pickle.load(open(sys.argv[1], 'rb'))",
        str(path),
    ]
    try:
        result = subprocess.run(
            pickle_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        pickle_error = (result.stderr or "").strip().splitlines()[-1:] or [f"exit={result.returncode}"]
    except Exception as exc:
        pickle_error = [str(exc)]
    return False, f"joblib: {joblib_error[0]} / pickle: {pickle_error[0]}"


def build_scoring_coeff_health(
    *,
    coeff_overrides_path: Path = DEFAULT_COEFF_OVERRIDES,
    coeff_auto_path: Path = DEFAULT_COEFF_AUTO,
    model_paths: tuple[Path, ...] = DEFAULT_MODEL_PATHS,
) -> dict[str, Any]:
    overrides_raw, overrides_available, overrides_error = _load_json_any(coeff_overrides_path)
    auto_raw, auto_available, auto_error = _load_json_any(coeff_auto_path)
    overrides = overrides_raw if isinstance(overrides_raw, dict) else {}
    auto = auto_raw if isinstance(auto_raw, dict) else {}

    issues: list[dict[str, str]] = []
    if not overrides_available:
        issues.append({"severity": "warn", "code": "coeff_overrides_unavailable", "message": f"{coeff_overrides_path} を読めません: {overrides_error}"})
    if not auto_available:
        issues.append({"severity": "warn", "code": "coeff_auto_unavailable", "message": f"{coeff_auto_path} を読めません: {auto_error}"})

    key_results: dict[str, dict[str, Any]] = {}
    for key in _EXPECTED_MODEL_KEYS:
        coeffs = _effective_coeffs_for_health(overrides, key)
        missing = [name for name in _REQUIRED_COEFF_KEYS if name not in coeffs]
        non_numeric = [name for name in _REQUIRED_COEFF_KEYS if name in coeffs and not _is_finite_number(coeffs.get(name))]
        numeric_values = [float(coeffs[name]) for name in _REQUIRED_COEFF_KEYS if name in coeffs and _is_finite_number(coeffs.get(name))]
        nonzero_required = sum(1 for value in numeric_values if abs(value) > 1e-12)
        effective_numeric = [
            float(value)
            for name, value in coeffs.items()
            if not str(name).startswith("_") and _is_finite_number(value)
        ]
        nonzero_total = sum(1 for value in effective_numeric if abs(value) > 1e-12)
        key_results[key] = {
            "missing_required": missing,
            "non_numeric_required": non_numeric,
            "required_nonzero_count": nonzero_required,
            "total_numeric_count": len(effective_numeric),
            "total_nonzero_count": nonzero_total,
        }
        if missing:
            issues.append({"severity": "attention", "code": "coeff_required_missing", "message": f"{key}: 必須係数が欠落しています: {', '.join(missing)}"})
        if non_numeric:
            issues.append({"severity": "attention", "code": "coeff_required_non_numeric", "message": f"{key}: 必須係数が数値ではありません: {', '.join(non_numeric)}"})
        if not missing and not non_numeric and nonzero_required == 0:
            issues.append({"severity": "attention", "code": "coeff_required_all_zero", "message": f"{key}: 必須係数がすべて0です"})
        if len(effective_numeric) and nonzero_total == 0:
            issues.append({"severity": "attention", "code": "coeff_all_zero", "message": f"{key}: 有効係数がすべて0です"})

    auto_values: dict[str, float | None] = {}
    for key in _AUTO_WEIGHT_KEYS:
        value = auto.get(key)
        auto_values[key] = float(value) if _is_finite_number(value) else None
        if key in auto and auto_values[key] is None:
            issues.append({"severity": "attention", "code": "auto_weight_non_numeric", "message": f"{key}: 自動重みが数値ではありません"})
        if auto_values[key] is not None and not (0.0 <= float(auto_values[key]) <= 1.0):
            issues.append({"severity": "attention", "code": "auto_weight_out_of_range", "message": f"{key}: 自動重みが0〜1の範囲外です ({auto_values[key]})"})

    borrower_asset_sum = None
    if auto_values["_auto_weight_borrower"] is not None and auto_values["_auto_weight_asset"] is not None:
        borrower_asset_sum = round(float(auto_values["_auto_weight_borrower"]) + float(auto_values["_auto_weight_asset"]), 6)
        if abs(borrower_asset_sum - 1.0) > 0.05:
            issues.append({"severity": "warn", "code": "borrower_asset_weight_sum", "message": f"借手/物件の自動重み合計が1から離れています: {borrower_asset_sum}"})
    quant_qual_sum = None
    if auto_values["_auto_weight_quant"] is not None and auto_values["_auto_weight_qual"] is not None:
        quant_qual_sum = round(float(auto_values["_auto_weight_quant"]) + float(auto_values["_auto_weight_qual"]), 6)
        if abs(quant_qual_sum - 1.0) > 0.05:
            issues.append({"severity": "warn", "code": "quant_qual_weight_sum", "message": f"定量/定性の自動重み合計が1から離れています: {quant_qual_sum}"})
    blend_sum = None
    if all(auto_values[key] is not None for key in ("_auto_blend_w_main", "_auto_blend_w_bench", "_auto_blend_w_ind")):
        blend_sum = round(
            float(auto_values["_auto_blend_w_main"]) + float(auto_values["_auto_blend_w_bench"]) + float(auto_values["_auto_blend_w_ind"]),
            6,
        )
        if abs(blend_sum - 1.0) > 0.05:
            issues.append({"severity": "warn", "code": "blend_weight_sum", "message": f"モデルブレンド重み合計が1から離れています: {blend_sum}"})

    model_results: dict[str, dict[str, Any]] = {}
    for path in model_paths:
        ok, error = _check_model_load(path)
        model_results[str(path)] = {
            "available": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "load_ok": ok,
            "error": error,
        }
        if not ok:
            issues.append({"severity": "attention", "code": "model_load_failed", "message": f"{path} をロードできません: {error}"})

    attention_count = sum(1 for issue in issues if issue["severity"] == "attention")
    warn_count = sum(1 for issue in issues if issue["severity"] == "warn")
    status = "attention" if attention_count else "warn" if warn_count else "ok"
    return {
        "status": status,
        "coeff_sources": {
            "overrides": {"path": str(coeff_overrides_path), "available": overrides_available},
            "auto": {"path": str(coeff_auto_path), "available": auto_available},
        },
        "expected_model_key_count": len(_EXPECTED_MODEL_KEYS),
        "checked_model_keys": key_results,
        "auto_weights": {
            "values": auto_values,
            "borrower_asset_sum": borrower_asset_sum,
            "quant_qual_sum": quant_qual_sum,
            "blend_sum": blend_sum,
        },
        "model_files": model_results,
        "issues": issues,
    }


def _latest_codex_queue_result(reports_dir: Path) -> tuple[dict[str, Any], Path | None]:
    """最新の codex_queue_result_*.json を返す（無ければ空）。"""
    candidates = sorted(reports_dir.glob("codex_queue_result_*.json"))
    if not candidates:
        return {}, None
    latest = candidates[-1]
    data, ok = _load_json(latest)
    return (data if ok else {}), latest


def build_guard_health(
    *,
    reports_dir: Path = REPORTS_DIR,
    preflight_retry_state_path: Path = DEFAULT_PREFLIGHT_RETRY_STATE,
    preflight_max_retries: int | None = None,
) -> dict[str, Any]:
    """安全ガード層（Codex自律実行ブレーカー・PR前プリフライト）の作動状況を集約する。

    器が自分の安全装置を観測できるようにするための読み取り専用ヘルス。
    - Codex キュー: reports/codex_queue_result_*.json の guards ブロック
      （日次上限の繰り越し・連続失敗停止）。出典: scripts/execute_codex_queue.py。
    - プリフライト: .claude/state/preflight_retries.json のリトライ枠超過。
      出典: scripts/preflight_pr_guard.py。
    """
    if preflight_max_retries is None:
        try:
            preflight_max_retries = int(os.environ.get("PREFLIGHT_MAX_RETRIES", "2"))
        except ValueError:
            preflight_max_retries = 2

    issues: list[dict[str, str]] = []

    # ── Codex 自律実行キューのガード ──
    codex, codex_path = _latest_codex_queue_result(reports_dir)
    guards = codex.get("guards") if isinstance(codex.get("guards"), dict) else {}
    carried_over = guards.get("carried_over") if isinstance(guards.get("carried_over"), list) else []
    aborted = bool(guards.get("aborted_by_consecutive_failures"))
    if aborted:
        issues.append({
            "severity": "attention",
            "code": "codex_consecutive_failures_abort",
            "message": f"Codex自律実行が連続失敗（上限 {guards.get('max_consecutive_failures')}）で停止しています",
        })
    if carried_over:
        issues.append({
            "severity": "warn",
            "code": "codex_daily_limit_carryover",
            "message": f"日次上限で {len(carried_over)} 件が翌日以降へ繰り越されています",
        })

    # ── PR前プリフライトのリトライ枠 ──
    retry_state, retry_available = _load_json(preflight_retry_state_path)
    over_budget: list[str] = []
    max_retry_count = 0
    if isinstance(retry_state, dict):
        for sig, entry in retry_state.items():
            count = _safe_int(entry.get("count")) if isinstance(entry, dict) else 0
            max_retry_count = max(max_retry_count, count)
            if count > preflight_max_retries:
                over_budget.append(str(sig))
    if over_budget:
        issues.append({
            "severity": "warn",
            "code": "preflight_retry_budget_exhausted",
            "message": f"プリフライトのリトライ枠超過が {len(over_budget)} 箇所（上限 {preflight_max_retries}）— 人間へのバトンタッチを検討",
        })

    attention_count = sum(1 for i in issues if i["severity"] == "attention")
    warn_count = sum(1 for i in issues if i["severity"] == "warn")
    status = "attention" if attention_count else "warn" if warn_count else "ok"
    return {
        "status": status,
        "codex_queue": {
            "available": codex_path is not None,
            "path": str(codex_path) if codex_path else "",
            "total": _safe_int(codex.get("total")),
            "failed": _safe_int(codex.get("failed")),
            "carried_over_count": len(carried_over),
            "aborted_by_consecutive_failures": aborted,
            "max_consecutive_failures": _safe_int(guards.get("max_consecutive_failures")),
        },
        "preflight_guard": {
            "available": retry_available,
            "tracked_signatures": len(retry_state) if isinstance(retry_state, dict) else 0,
            "over_budget_count": len(over_budget),
            "max_retry_count": max_retry_count,
            "max_retries": preflight_max_retries,
        },
        "issues": issues,
    }


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """jsonl を dict 行のリストで読む（欠損・壊れ行は無視・空リスト）。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except OSError:
        return []
    return rows


def build_outcome_health(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    pdca_log_path: Path = DEFAULT_PDCA_LOG,
) -> dict[str, Any]:
    """結果ループ（Outcome Loop）の作動状況と効果測定を集約する。

    「改善を入れる」だけでなく「効いたか」を器が俯瞰できるようにするための
    読み取り専用ヘルス。api/*_loop.py の重い依存は import せず、各ループが
    永続化する data/*.jsonl を直接読む（欠損は available:false）。

    - 4結果ループの提案件数・最新日時（出典: api/outcome_drift_loop.py ほか）
    - feedback_pattern の PDCA効果ログ（data/shion_self_pdca_log.jsonl）の
      delta（negative_rate の after - before、負=改善）を集計。
    """
    issues: list[dict[str, str]] = []

    loops: dict[str, dict[str, Any]] = {}
    for name, filename in _OUTCOME_LOOP_FILES.items():
        path = data_dir / filename
        rows = _read_jsonl_rows(path)
        latest_at = max((str(r.get("generated_at") or "") for r in rows), default="")
        loops[name] = {
            "available": path.exists(),
            "proposal_count": len(rows),
            "latest_generated_at": latest_at,
        }

    pdca_rows = _read_jsonl_rows(pdca_log_path)
    measured = len(pdca_rows)
    improved = sum(1 for r in pdca_rows if _safe_float(r.get("delta")) < 0)
    worsened = sum(1 for r in pdca_rows if _safe_float(r.get("delta")) > 0)
    avg_delta = round(sum(_safe_float(r.get("delta")) for r in pdca_rows) / measured, 3) if measured else 0.0

    if measured > 0 and worsened > improved:
        issues.append({
            "severity": "warn",
            "code": "adopted_improvements_net_worsening",
            "message": f"採用済み改善の効果測定で悪化({worsened})が改善({improved})を上回っています（効いていない改善の見直しを検討）",
        })

    warn_count = sum(1 for i in issues if i["severity"] == "warn")
    status = "warn" if warn_count else "ok"
    return {
        "status": status,
        "loops": loops,
        "pdca": {
            "available": pdca_log_path.exists(),
            "measured_count": measured,
            "improved_count": improved,
            "worsened_count": worsened,
            "avg_delta": avg_delta,
        },
        "issues": issues,
    }


def _percent(part: int | float, total: int | float) -> float:
    return round(float(part) / float(total) * 100, 1) if total else 0.0


def build_loop_metrics(
    *,
    latest_report_path: Path = DEFAULT_LATEST_REPORT,
    recursive_report_path: Path = DEFAULT_RECURSIVE_REPORT,
    prompt_log_path: Path = DEFAULT_LOG_PATH,
    coeff_overrides_path: Path = DEFAULT_COEFF_OVERRIDES,
    coeff_auto_path: Path = DEFAULT_COEFF_AUTO,
    model_paths: tuple[Path, ...] = DEFAULT_MODEL_PATHS,
    guard_reports_dir: Path = REPORTS_DIR,
    preflight_retry_state_path: Path = DEFAULT_PREFLIGHT_RETRY_STATE,
    outcome_data_dir: Path = DEFAULT_DATA_DIR,
    pdca_log_path: Path = DEFAULT_PDCA_LOG,
) -> dict[str, Any]:
    latest_report, latest_available = _load_json(latest_report_path)
    recursive_report, recursive_available = _load_json(recursive_report_path)
    prompt_rows = load_jsonl(prompt_log_path)
    prompt_available = prompt_log_path.exists()
    prompt_summary = build_prompt_summary(prompt_rows)
    scoring_health = build_scoring_coeff_health(
        coeff_overrides_path=coeff_overrides_path,
        coeff_auto_path=coeff_auto_path,
        model_paths=model_paths,
    )
    guard_health = build_guard_health(
        reports_dir=guard_reports_dir,
        preflight_retry_state_path=preflight_retry_state_path,
    )
    outcome_health = build_outcome_health(
        data_dir=outcome_data_dir,
        pdca_log_path=pdca_log_path,
    )

    applied_count = _safe_int(latest_report.get("applied_count"))
    needs_review_count = _safe_int(latest_report.get("needs_review_count"))
    failed_count = _safe_int(latest_report.get("failed_count"))
    improvement_total = applied_count + needs_review_count + failed_count

    recursive_measurement = recursive_report.get("measurement_summary") or {}
    canonical_count = _safe_int(recursive_report.get("canonical_candidate_count"))
    ranked_queue_count = _safe_int(recursive_report.get("ranked_queue_count"))
    suppressed_count = _safe_int(recursive_report.get("suppressed_count"))

    available_sources = sum([latest_available, recursive_available, prompt_available])
    status = "ok"
    recommendations: list[str] = []

    if not latest_available:
        status = "attention"
        recommendations.append("reports/latest.json を生成して改善候補ループの正本を確認する")
    if not recursive_available:
        if status == "ok":
            status = "warn"
        recommendations.append("日次改善パイプライン後に recursive_self_improvement_latest.json を生成する")
    if prompt_available and not prompt_rows:
        if status == "ok":
            status = "warn"
        recommendations.append("prompt feedback log は存在するが空のため、AI応答改善ループの観測が不足している")
    if prompt_available and prompt_rows and _safe_float(prompt_summary.get("pdca_rate")) == 0.0:
        status = "attention"
        recommendations.append("prompt feedback log に行があるのに PDCA反映率が0%です。ログパス・pdca_ai_rules・記録フィールドを確認する")
    if scoring_health["status"] == "attention":
        status = "attention"
        recommendations.append("スコアリング係数/モデルのヘルスチェックに重大な異常があります")
    elif scoring_health["status"] == "warn" and status == "ok":
        status = "warn"
        recommendations.append("スコアリング係数/モデルのヘルスチェックに警告があります")
    if needs_review_count >= 25:
        if status == "ok":
            status = "warn"
        recommendations.append("needs_review が多いため、低リスク候補と高リスク候補を分けて棚卸しする")
    # 滞留(churn)判定は churn_rate を使う（＝クールダウン固着のみ）。
    # noise_rate は applied 等の健全な重複排除も含むため、それ単体を異常扱いしない。
    # churn_rate 未提供の古いレポートは noise_rate にフォールバック（後方互換）。
    churn_signal = (
        _safe_float(recursive_measurement.get("churn_rate"))
        if "churn_rate" in recursive_measurement
        else _safe_float(recursive_measurement.get("noise_rate"))
    )
    if churn_signal >= 50.0:
        if status == "ok":
            status = "warn"
        recommendations.append(
            "抑制の滞留(churn)が高いため、needs_review/suppressed のクールダウン固着や"
            "台帳の suppressed 再記録を確認する（健全な重複排除は含めない）"
        )
    if guard_health["status"] == "attention":
        status = "attention"
        recommendations.append("安全ガードに重大な作動: Codex自律実行が連続失敗で停止しています。原因を確認する")
    elif guard_health["status"] == "warn":
        if status == "ok":
            status = "warn"
        recommendations.append("安全ガードに警告: 日次上限の繰り越しやプリフライトのリトライ枠超過を確認する")
    if outcome_health["status"] == "warn":
        if status == "ok":
            status = "warn"
        recommendations.append("結果ループ: 採用済み改善の効果測定で悪化が改善を上回っています。効いていない改善の見直しを検討する")
    if not recommendations:
        recommendations.append("現状は読み取り専用の定点観測を継続する")

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "sources": {
            "latest_report": {"path": str(latest_report_path), "available": latest_available},
            "recursive_report": {"path": str(recursive_report_path), "available": recursive_available},
            "prompt_feedback_log": {
                "path": str(prompt_log_path),
                "available": prompt_available,
                "rows": len(prompt_rows),
            },
        },
        "health": {
            "source_coverage_rate": _percent(available_sources, 3),
            "review_pressure_rate": _percent(needs_review_count, improvement_total),
            "auto_application_rate": _percent(applied_count, applied_count + needs_review_count),
            "recursive_queue_rate": _percent(ranked_queue_count, canonical_count),
            "recursive_suppression_rate": _percent(suppressed_count, canonical_count),
        },
        "improvement_loop": {
            "applied_count": applied_count,
            "needs_review_count": needs_review_count,
            "failed_count": failed_count,
            "total": improvement_total,
        },
        "recursive_loop": {
            "canonical_candidate_count": canonical_count,
            "ranked_queue_count": ranked_queue_count,
            "suppressed_count": suppressed_count,
            "measurement_summary": {
                "pdca_rate": _safe_float(recursive_measurement.get("pdca_rate")),
                "response_changed_rate": _safe_float(recursive_measurement.get("response_changed_rate")),
                "repeat_issue_rate": _safe_float(recursive_measurement.get("repeat_issue_rate")),
                "reuse_rate": _safe_float(recursive_measurement.get("reuse_rate")),
                "noise_rate": _safe_float(recursive_measurement.get("noise_rate")),
                "churn_rate": _safe_float(recursive_measurement.get("churn_rate")),
                "suppressed_healthy_count": _safe_int(recursive_measurement.get("suppressed_healthy_count")),
                "suppressed_churn_count": _safe_int(recursive_measurement.get("suppressed_churn_count")),
            },
        },
        "prompt_feedback_loop": {
            "total": _safe_int(prompt_summary.get("total")),
            "pdca_count": _safe_int(prompt_summary.get("pdca_count")),
            "pdca_rate": _safe_float(prompt_summary.get("pdca_rate")),
            "previous_diff_count": _safe_int(prompt_summary.get("previous_diff_count")),
            "previous_diff_rate": _safe_float(prompt_summary.get("previous_diff_rate")),
            "surface_counts": prompt_summary.get("surface_counts") or {},
        },
        "scoring_coefficients": scoring_health,
        "guard_health": guard_health,
        "outcome_health": outcome_health,
        "recommendations": recommendations,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Loop Engineering Health")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Source coverage: {report['health']['source_coverage_rate']}%")
    lines.append("")
    lines.append("## Improvement Loop")
    improvement = report["improvement_loop"]
    lines.append(f"- Applied: {improvement['applied_count']}")
    lines.append(f"- Needs review: {improvement['needs_review_count']}")
    lines.append(f"- Failed: {improvement['failed_count']}")
    lines.append(f"- Review pressure: {report['health']['review_pressure_rate']}%")
    lines.append("")
    lines.append("## Recursive Loop")
    recursive = report["recursive_loop"]
    measurement = recursive["measurement_summary"]
    lines.append(f"- Canonical candidates: {recursive['canonical_candidate_count']}")
    lines.append(f"- Ranked queue: {recursive['ranked_queue_count']}")
    lines.append(f"- Suppressed: {recursive['suppressed_count']}")
    lines.append(f"- Repeat issue rate: {measurement['repeat_issue_rate']}%")
    lines.append(f"- Reuse rate: {measurement['reuse_rate']}%")
    lines.append(f"- Noise rate: {measurement['noise_rate']}%")
    lines.append(
        f"- Churn rate: {measurement.get('churn_rate', 0.0)}% "
        f"(healthy dedup: {measurement.get('suppressed_healthy_count', 0)}, "
        f"churn: {measurement.get('suppressed_churn_count', 0)})"
    )
    lines.append("")
    lines.append("## Prompt Feedback Loop")
    prompt = report["prompt_feedback_loop"]
    lines.append(f"- Total: {prompt['total']}")
    lines.append(f"- PDCA applied: {prompt['pdca_count']} ({prompt['pdca_rate']}%)")
    lines.append(f"- Previous response diffs: {prompt['previous_diff_count']} ({prompt['previous_diff_rate']}%)")
    lines.append("")
    lines.append("## Scoring Coefficients")
    scoring = report["scoring_coefficients"]
    lines.append(f"- Status: `{scoring['status']}`")
    lines.append(f"- Checked model keys: {scoring['expected_model_key_count']}")
    lines.append(f"- Borrower/asset weight sum: {scoring['auto_weights']['borrower_asset_sum']}")
    lines.append(f"- Quant/qual weight sum: {scoring['auto_weights']['quant_qual_sum']}")
    lines.append(f"- Blend weight sum: {scoring['auto_weights']['blend_sum']}")
    if scoring["issues"]:
        for issue in scoring["issues"][:8]:
            lines.append(f"- [{issue['severity']}] {issue['message']}")
    else:
        lines.append("- No coefficient/model issues detected")
    lines.append("")
    lines.append("## Guard / Safety")
    guard = report.get("guard_health") or {}
    codex = guard.get("codex_queue", {})
    preflight = guard.get("preflight_guard", {})
    lines.append(f"- Status: `{guard.get('status', 'ok')}`")
    lines.append(
        f"- Codex queue: failed {codex.get('failed', 0)}/{codex.get('total', 0)}, "
        f"carried over {codex.get('carried_over_count', 0)}, "
        f"aborted={codex.get('aborted_by_consecutive_failures', False)}"
    )
    lines.append(
        f"- Preflight retries: over-budget {preflight.get('over_budget_count', 0)} "
        f"(max count {preflight.get('max_retry_count', 0)} / limit {preflight.get('max_retries', 0)})"
    )
    if guard.get("issues"):
        for issue in guard["issues"][:8]:
            lines.append(f"- [{issue['severity']}] {issue['message']}")
    else:
        lines.append("- No guard activations detected")
    lines.append("")
    lines.append("## Outcome Loops")
    outcome = report.get("outcome_health") or {}
    lines.append(f"- Status: `{outcome.get('status', 'ok')}`")
    for name, info in (outcome.get("loops") or {}).items():
        lines.append(
            f"- {name}: {info.get('proposal_count', 0)} proposals "
            f"(latest {info.get('latest_generated_at') or 'n/a'})"
        )
    pdca = outcome.get("pdca", {})
    lines.append(
        f"- PDCA effect: measured {pdca.get('measured_count', 0)}, "
        f"improved {pdca.get('improved_count', 0)}, worsened {pdca.get('worsened_count', 0)}, "
        f"avg delta {pdca.get('avg_delta', 0.0)}"
    )
    if outcome.get("issues"):
        for issue in outcome["issues"][:8]:
            lines.append(f"- [{issue['severity']}] {issue['message']}")
    lines.append("")
    lines.append("## Recommendations")
    for item in report["recommendations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-report", type=Path, default=DEFAULT_LATEST_REPORT)
    parser.add_argument("--recursive-report", type=Path, default=DEFAULT_RECURSIVE_REPORT)
    parser.add_argument("--prompt-log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_loop_metrics(
        latest_report_path=args.latest_report.expanduser(),
        recursive_report_path=args.recursive_report.expanduser(),
        prompt_log_path=args.prompt_log.expanduser(),
    )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    write_outputs(
        report,
        output_json=args.output_json.expanduser(),
        output_md=args.output_md.expanduser(),
    )
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
