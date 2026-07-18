#!/usr/bin/env python3
"""Build a read-only loop engineering health report from existing artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
    if _safe_float(recursive_measurement.get("noise_rate")) >= 50.0:
        if status == "ok":
            status = "warn"
        recommendations.append("noise_rate が高いため、重複候補と抑制ルールを確認する")
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
