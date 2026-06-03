#!/usr/bin/env python3
"""Small scoring smoke harness for lease screening changes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "scoring_harness_latest.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SAMPLE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "existing_service_standard",
        "name": "Existing service company with normal profits",
        "inputs": {
            "customer_type": "既存先",
            "industry_major": "R サービス業",
            "industry_sub": "92 その他の事業サービス業",
            "sales_dept": "本部",
            "grade": "1-3",
            "nenshu": 250_000,
            "gross_profit": 55_000,
            "op_profit": 18_000,
            "ord_profit": 16_000,
            "net_income": 9_000,
            "net_assets": 80_000,
            "total_assets": 220_000,
            "bank_credit": 60_000,
            "lease_credit": 15_000,
            "contracts": 3,
            "acquisition_cost": 12_000,
            "lease_term": 60,
            "asset_score": 70,
            "asset_name": "一般車両",
            "main_bank": "メイン先",
            "competitor": "競合なし",
            "intuition_score": 3,
        },
    },
    {
        "case_id": "new_manufacturing_watch",
        "name": "New manufacturing customer with thin equity",
        "inputs": {
            "customer_type": "新規先",
            "industry_major": "E 製造業",
            "industry_sub": "24 金属製品製造業",
            "sales_dept": "本部",
            "grade": "4-6",
            "nenshu": 180_000,
            "gross_profit": 28_000,
            "op_profit": 3_000,
            "ord_profit": 2_000,
            "net_income": 800,
            "net_assets": 5_000,
            "total_assets": 160_000,
            "machines": 45_000,
            "bank_credit": 95_000,
            "lease_credit": 8_000,
            "contracts": 0,
            "acquisition_cost": 30_000,
            "lease_term": 72,
            "asset_score": 55,
            "asset_name": "工作機械",
            "main_bank": "非メイン先",
            "competitor": "競合あり",
            "competitor_rate": 1.9,
            "deal_source": "銀行紹介",
            "intuition_score": 2,
        },
    },
    {
        "case_id": "negative_equity_alert",
        "name": "Negative equity case should still return a bounded result",
        "inputs": {
            "customer_type": "既存先",
            "industry_major": "D 建設業",
            "industry_sub": "06 総合工事業",
            "sales_dept": "本部",
            "grade": "7-8",
            "nenshu": 90_000,
            "gross_profit": 12_000,
            "op_profit": -2_000,
            "ord_profit": -3_000,
            "net_income": -4_000,
            "net_assets": -12_000,
            "total_assets": 70_000,
            "bank_credit": 55_000,
            "lease_credit": 20_000,
            "contracts": 1,
            "acquisition_cost": 8_000,
            "lease_term": 48,
            "asset_score": 45,
            "asset_name": "建設機械",
            "main_bank": "非メイン先",
            "competitor": "競合あり",
            "intuition_score": 4,
        },
    },
]


REQUIRED_NUMERIC_KEYS = [
    "score",
    "score_base",
    "score_borrower",
    "approval_line",
    "user_op_margin",
    "user_equity_ratio",
    "bench_op_margin",
    "bench_equity_ratio",
    "asset_score",
]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_scoring_result(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    for key in REQUIRED_NUMERIC_KEYS:
        if not _is_finite_number(result.get(key)):
            failures.append(f"{key} is missing or not finite")

    for key in ("score", "score_base", "score_borrower", "asset_score"):
        value = result.get(key)
        if _is_finite_number(value) and not 0.0 <= float(value) <= 100.0:
            failures.append(f"{key} is out of range: {value}")

    hantei = result.get("hantei")
    if hantei not in {"承認圏内", "要審議"}:
        failures.append(f"hantei is invalid: {hantei!r}")

    if _is_finite_number(result.get("score")) and _is_finite_number(result.get("approval_line")):
        expected = "承認圏内" if float(result["score"]) >= float(result["approval_line"]) else "要審議"
        if hantei != expected:
            failures.append(f"hantei mismatch: expected {expected}, got {hantei}")

    if not isinstance(result.get("comparison"), str) or not result["comparison"].strip():
        failures.append("comparison is missing")

    if not isinstance(result.get("score_contributions"), list):
        failures.append("score_contributions is missing or not a list")

    for list_key in ("asset_score_warnings", "credit_risk_warnings", "asset_warnings", "default_warnings"):
        if not isinstance(result.get(list_key), list):
            failures.append(f"{list_key} is missing or not a list")

    return failures


def run_harness(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from scoring_core import run_quick_scoring

    case_results: list[dict[str, Any]] = []
    for case in cases or SAMPLE_CASES:
        try:
            result = run_quick_scoring(case["inputs"])
            failures = validate_scoring_result(result)
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "name": case["name"],
                    "ok": not failures,
                    "failures": failures,
                    "summary": {
                        "score": result.get("score"),
                        "score_borrower": result.get("score_borrower"),
                        "hantei": result.get("hantei"),
                        "user_op_margin": result.get("user_op_margin"),
                        "user_equity_ratio": result.get("user_equity_ratio"),
                        "credit_risk_group_level": result.get("credit_risk_group_level"),
                    },
                }
            )
        except Exception as exc:
            case_results.append(
                {
                    "case_id": case.get("case_id", "unknown"),
                    "name": case.get("name", ""),
                    "ok": False,
                    "failures": [f"exception: {type(exc).__name__}: {exc}"],
                    "summary": {},
                }
            )

    failed = sum(1 for item in case_results if not item["ok"])
    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "total": len(case_results),
        "passed": len(case_results) - failed,
        "failed": failed,
        "cases": case_results,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the small lease scoring smoke harness.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH, help="JSON report path")
    parser.add_argument("--no-write", action="store_true", help="Do not write a JSON report")
    args = parser.parse_args(argv)

    report = run_harness()
    if not args.no_write:
        write_report(report, args.output)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
