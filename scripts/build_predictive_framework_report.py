#!/usr/bin/env python3
"""Build the read-only predictive framework report.

予測誤差ループが「実際に当たっているか」を1枚で見るための集計。
新しい予測器を足すためではなく、既に記録されている

- data/prediction_snapshots.jsonl（審査時点の予測）
- data/prediction_numeric_forecasts.jsonl（将来財務予測）
- data/prediction_numeric_actuals.jsonl（後日判明した財務実績）
- data/prediction_error_events.jsonl（観測と誤差）
- data/prediction_error_update_candidates.jsonl（信念更新候補）
- data/prediction_error_candidate_reviews.jsonl（人間の採否）

を読むだけに徹する。スコアリング・プロンプト・判断資産・モデル重みには
一切触れない。出力は人間レビューの入口であって、自動反映はしない。
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "prediction_snapshots.jsonl"
DEFAULT_NUMERIC_FORECASTS = REPO_ROOT / "data" / "prediction_numeric_forecasts.jsonl"
DEFAULT_NUMERIC_ACTUALS = REPO_ROOT / "data" / "prediction_numeric_actuals.jsonl"
DEFAULT_EVENTS = REPO_ROOT / "data" / "prediction_error_events.jsonl"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "prediction_error_update_candidates.jsonl"
DEFAULT_REVIEWS = REPO_ROOT / "data" / "prediction_error_candidate_reviews.jsonl"

DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "predictive_framework_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "predictive_framework_latest.md"

SCHEMA_VERSION = 1

# 予測が事前に固定されていたことを示す出所。これ以外は結果登録時の逆算なので、
# 「予測が当たったか」の証拠としては弱い。
SNAPSHOT_SOURCE = "snapshot"

RISK_ORDER = ["high", "medium", "low", "unknown"]
RISK_LABELS = {"high": "高リスク予測", "medium": "中リスク予測", "low": "低リスク予測", "unknown": "リスク未判定"}

REPEAT_BELIEF_LIMIT = 10
# 同じ前提が何回外れたら「繰り返し外している」として先回り提示の対象にするか。
# 緩めると通知が増えて無視されるため、既定の review_threshold と同じ3から始める。
REPEAT_ALERT_THRESHOLD = 3
# リスク帯の逆転を判定するのに必要な最低件数。少数件での逆転は偶然でしかない。
MIN_CALIBRATION_EVENTS_PER_BAND = 5
MIN_CONFIDENCE_CALIBRATION_EVENTS = 30
MIN_CONFIDENCE_BIN_EVENTS = 5
MIN_NUMERIC_ACTUAL_MATCHES = 5
TOKYO_TIMEZONE = ZoneInfo("Asia/Tokyo")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _text(value: Any) -> str:
    return str(value or "").strip()


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _timestamp_or_none(value: Any) -> datetime | None:
    """ISO日付/日時を比較可能なUTC datetimeへ変換する。"""
    raw = _text(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _observation_date_or_none(value: Any) -> date | None:
    """実績日を東京時間の暦日として読む。旧timestamp形式もJSTへ変換する。"""
    raw = _text(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        parsed = _timestamp_or_none(raw)
        return parsed.astimezone(TOKYO_TIMEZONE).date() if parsed is not None else None


def build_snapshot_summary(snapshot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_risk: dict[str, int] = {}
    cases: set[str] = set()
    for row in snapshot_rows:
        case_id = _text(row.get("case_id"))
        if case_id:
            cases.add(case_id)
        prediction = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        risk = _text(prediction.get("risk_level")) or "unknown"
        by_risk[risk] = by_risk.get(risk, 0) + 1
    return {
        "total": len(snapshot_rows),
        "unique_cases": len(cases),
        "by_risk_level": by_risk,
    }


def _forecast_value(forecast: dict[str, Any], key: str, percentile: str, year: int) -> float | None:
    bands = forecast.get(key) if isinstance(forecast.get(key), dict) else {}
    values = bands.get(percentile)
    if not isinstance(values, list) or year < 0 or year >= len(values):
        return None
    return _float_or_none(values[year])


def _numeric_metric(rows: list[dict[str, float | None]], actual_key: str, forecast_key: str) -> dict[str, Any]:
    errors: list[float] = []
    percentage_errors: list[float] = []
    interval_hits: list[float] = []
    for row in rows:
        actual = _float_or_none(row.get(actual_key))
        predicted = _float_or_none(row.get(forecast_key))
        if actual is None or predicted is None:
            continue
        errors.append(abs(predicted - actual))
        if actual != 0:
            percentage_errors.append(abs(predicted - actual) / abs(actual))
        low = _float_or_none(row.get(f"{forecast_key}_p10"))
        high = _float_or_none(row.get(f"{forecast_key}_p90"))
        if low is not None and high is not None:
            interval_hits.append(1.0 if low <= actual <= high else 0.0)
    return {
        "observations": len(errors),
        "mae": _mean(errors),
        "mape": _mean(percentage_errors),
        "interval_80_coverage": _mean(interval_hits),
    }


def build_numeric_forecast_summary(
    forecast_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """将来財務予測と後日実績を案件・予測年で突合し、shadow誤差を測る。"""
    cases: set[str] = set()
    by_method: dict[str, int] = {}
    for row in forecast_rows:
        case_id = _text(row.get("case_id"))
        if case_id:
            cases.add(case_id)
        forecast = row.get("forecast") if isinstance(row.get("forecast"), dict) else {}
        method = _text(forecast.get("method")) or "unknown"
        by_method[method] = by_method.get(method, 0) + 1

    latest_actuals: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sorted(actual_rows or [], key=lambda item: _text(item.get("recorded_at"))):
        case_id = _text(row.get("case_id"))
        try:
            year = int(row.get("observed_year"))
        except (TypeError, ValueError):
            continue
        if case_id and year >= 1:
            latest_actuals[(case_id, year)] = row

    forecasts_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in forecast_rows:
        case_id = _text(row.get("case_id"))
        if case_id:
            forecasts_by_case.setdefault(case_id, []).append(row)
    for rows in forecasts_by_case.values():
        rows.sort(key=lambda item: _timestamp_or_none(item.get("captured_at")) or datetime.min.replace(tzinfo=timezone.utc))

    matched_rows: list[dict[str, float | None]] = []
    actuals_without_observation_date = 0
    for (case_id, year), actual_row in latest_actuals.items():
        observed_date = _observation_date_or_none(actual_row.get("observed_at"))
        if observed_date is None:
            actuals_without_observation_date += 1
            continue
        candidates = forecasts_by_case.get(case_id) or []
        if not candidates:
            continue
        eligible = [
            row for row in candidates
            if (
                (captured_at := _timestamp_or_none(row.get("captured_at"))) is not None
                and captured_at.astimezone(TOKYO_TIMEZONE).date() <= observed_date
            )
        ]
        if not eligible:
            continue
        forecast_row = eligible[-1]
        forecast = forecast_row.get("forecast") if isinstance(forecast_row.get("forecast"), dict) else {}
        actual = actual_row.get("actual") if isinstance(actual_row.get("actual"), dict) else {}
        matched_rows.append(
            {
                "sales": _float_or_none(actual.get("sales")),
                "sales_p50": _forecast_value(forecast, "sales_percentiles", "50", year),
                "sales_p50_p10": _forecast_value(forecast, "sales_percentiles", "10", year),
                "sales_p50_p90": _forecast_value(forecast, "sales_percentiles", "90", year),
                "op_profit": _float_or_none(actual.get("op_profit")),
                "op_profit_p50": _forecast_value(forecast, "op_percentiles", "50", year),
                "op_profit_p50_p10": _forecast_value(forecast, "op_percentiles", "10", year),
                "op_profit_p50_p90": _forecast_value(forecast, "op_percentiles", "90", year),
            }
        )

    matched = len(matched_rows)
    return {
        "total": len(forecast_rows),
        "unique_cases": len(cases),
        "by_method": by_method,
        "actuals_total": len(actual_rows or []),
        "actuals_latest": len(latest_actuals),
        "actuals_without_observation_date": actuals_without_observation_date,
        "matched_actuals": matched,
        "match_rate": _rate(matched, len(latest_actuals)),
        "calibratable": matched > 0,
        "trustworthy": matched >= MIN_NUMERIC_ACTUAL_MATCHES,
        "minimum_matches_required": MIN_NUMERIC_ACTUAL_MATCHES,
        "calibration_blocker": None if matched > 0 else "future_actuals_not_collected_or_unmatched",
        "matching_policy": "latest_forecast_captured_on_or_before_observed_date_asia_tokyo",
        "sales_error": _numeric_metric(matched_rows, "sales", "sales_p50"),
        "op_profit_error": _numeric_metric(matched_rows, "op_profit", "op_profit_p50"),
    }


def build_prediction_coverage(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """観測済み案件のうち、予測が事前に固定されていた割合。

    この数字が上がらない限り、以下のキャリブレーションは
    「結果を見てから作った予測」を採点しているだけなので信用できない。
    """
    by_source: dict[str, int] = {}
    for row in event_rows:
        prediction = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        source = _text(prediction.get("prediction_source")) or "unspecified"
        by_source[source] = by_source.get(source, 0) + 1
    total = len(event_rows)
    from_snapshot = by_source.get(SNAPSHOT_SOURCE, 0)
    return {
        "observed_events": total,
        "from_snapshot": from_snapshot,
        "by_source": by_source,
        "coverage_rate": _rate(from_snapshot, total),
        "trustworthy": from_snapshot > 0,
    }


def build_calibration(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """予測したリスク帯ごとに、実際どうなったかを並べる。"""
    buckets: dict[str, dict[str, int]] = {}
    for row in event_rows:
        prediction = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        risk = _text(prediction.get("risk_level")) or "unknown"
        bucket = buckets.setdefault(risk, {"events": 0, "contracted": 0, "delinquency": 0, "from_snapshot": 0})
        bucket["events"] += 1
        if outcome.get("contracted"):
            bucket["contracted"] += 1
        if outcome.get("delinquency"):
            bucket["delinquency"] += 1
        if _text(prediction.get("prediction_source")) == SNAPSHOT_SOURCE:
            bucket["from_snapshot"] += 1

    rows: list[dict[str, Any]] = []
    for risk in RISK_ORDER:
        if risk not in buckets:
            continue
        bucket = buckets[risk]
        rows.append(
            {
                "risk_level": risk,
                "label": RISK_LABELS.get(risk, risk),
                "events": bucket["events"],
                "from_snapshot": bucket["from_snapshot"],
                "contracted": bucket["contracted"],
                "contract_rate": _rate(bucket["contracted"], bucket["events"]),
                "delinquency": bucket["delinquency"],
                "delinquency_rate": _rate(bucket["delinquency"], bucket["events"]),
            }
        )
    return rows


def build_confidence_calibration(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """代理confidenceを、事前予測が実際に当たった割合へshadow校正する。

    high は「非成約または延滞」、low は「成約かつ延滞なし」を当たりと定義する。
    medium は方向を一意に定められないため校正対象外。生のconfidenceや判定は
    変更せず、十分な件数が溜まった時だけ経験的な信頼度をレポートする。
    """
    eligible: list[tuple[float, float]] = []
    for row in event_rows:
        prediction = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        if _text(prediction.get("prediction_source")) != SNAPSHOT_SOURCE:
            continue
        risk = _text(prediction.get("risk_level")).lower()
        confidence = _float_or_none(prediction.get("confidence"))
        if risk not in {"high", "low"} or confidence is None:
            continue
        confidence = min(1.0, max(0.0, confidence))
        if risk == "high":
            correct = bool(outcome.get("delinquency")) or not bool(outcome.get("contracted"))
        else:
            correct = bool(outcome.get("contracted")) and not bool(outcome.get("delinquency"))
        eligible.append((confidence, 1.0 if correct else 0.0))

    bin_specs = [
        (0.0, 0.5, "0.00–0.49"),
        (0.5, 0.65, "0.50–0.64"),
        (0.65, 0.8, "0.65–0.79"),
        (0.8, 1.001, "0.80–1.00"),
    ]
    bins: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for low, high, label in bin_specs:
        values = [(confidence, correct) for confidence, correct in eligible if low <= confidence < high]
        if not values:
            continue
        mean_confidence = _mean([value[0] for value in values])
        accuracy = _mean([value[1] for value in values])
        gap = None if mean_confidence is None or accuracy is None else round(abs(mean_confidence - accuracy), 3)
        if gap is not None:
            weighted_gap += gap * len(values)
        bins.append(
            {
                "range": label,
                "events": len(values),
                "mean_raw_confidence": mean_confidence,
                "observed_accuracy": accuracy,
                "calibration_gap": gap,
                "calibrated_confidence": accuracy if len(values) >= MIN_CONFIDENCE_BIN_EVENTS else None,
            }
        )

    total = len(eligible)
    return {
        "eligible_events": total,
        "excluded_events": max(0, len(event_rows) - total),
        "accuracy": _mean([correct for _, correct in eligible]),
        "mean_raw_confidence": _mean([confidence for confidence, _ in eligible]),
        "brier_score": _mean([(confidence - correct) ** 2 for confidence, correct in eligible]),
        "expected_calibration_error": round(weighted_gap / total, 3) if total else None,
        "trustworthy": total >= MIN_CONFIDENCE_CALIBRATION_EVENTS,
        "minimum_events_required": MIN_CONFIDENCE_CALIBRATION_EVENTS,
        "policy": "shadow_only_raw_confidence_unchanged",
        "bins": bins,
    }


def build_calibration_warnings(calibration: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """リスク帯の並びが実績と逆転していないかだけを見る。

    延滞は稀にしか観測されない（結果登録の大半は成約・失注）。両帯とも延滞0件の
    状態で率だけを比較すると 0.0 <= 0.0 が常に成立し、異常が無くても警告が出続ける。
    そのため「延滞が1件以上観測されていること」と「各帯に最低限の件数があること」を
    前提条件にし、それを満たさない間は判定不能として黙る。
    """
    by_risk = {row["risk_level"]: row for row in calibration}
    warnings: list[dict[str, Any]] = []

    high = by_risk.get("high")
    low = by_risk.get("low")
    if not high or not low:
        return warnings
    if high["events"] < MIN_CALIBRATION_EVENTS_PER_BAND or low["events"] < MIN_CALIBRATION_EVENTS_PER_BAND:
        return warnings
    if (high["delinquency"] + low["delinquency"]) < 1:
        # 延滞が一度も出ていない＝並びを判定する材料が無い。異常ではない。
        return warnings

    high_delinquency = high["delinquency_rate"]
    low_delinquency = low["delinquency_rate"]
    if high_delinquency is not None and low_delinquency is not None and high_delinquency <= low_delinquency:
        warnings.append(
            {
                "kind": "risk_order_inverted",
                "message": (
                    f"高リスク予測の延滞率({high_delinquency})が低リスク予測({low_delinquency})を上回っていない。"
                    "リスクの並びが実績と逆転している可能性がある。"
                ),
                "requires_human_review": True,
            }
        )
    return warnings


def build_surprise_rate(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """severity が high/medium の誤差を「驚き」として数える。"""
    by_severity: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in event_rows:
        error = row.get("prediction_error") if isinstance(row.get("prediction_error"), dict) else {}
        severity = _text(error.get("severity")) or "unknown"
        error_type = _text(error.get("type")) or "unknown"
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_type[error_type] = by_type.get(error_type, 0) + 1
    surprising = by_severity.get("high", 0) + by_severity.get("medium", 0)
    return {
        "total": len(event_rows),
        "surprising": surprising,
        "rate": _rate(surprising, len(event_rows)),
        "by_severity": by_severity,
        "by_error_type": by_type,
    }


def build_repeat_beliefs(
    candidate_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """同じ前提が繰り返し外れているものを、多い順に返す。"""
    latest_reviews: dict[str, dict[str, Any]] = {}
    for review in review_rows:
        candidate_id = _text(review.get("candidate_id"))
        if candidate_id:
            latest_reviews[candidate_id] = review

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidate_rows:
        belief = _text(candidate.get("target_belief")) or "unknown"
        error_type = _text(candidate.get("error_type")) or "unknown"
        key = (belief, error_type)
        entry = grouped.setdefault(
            key,
            {
                "target_belief": belief,
                "error_type": error_type,
                "occurrences": 0,
                "reviewed": 0,
                "severity": _text(candidate.get("severity")),
                "latest_suggested_update": "",
                "case_ids": [],
            },
        )
        entry["occurrences"] += 1
        entry["latest_suggested_update"] = _text(candidate.get("suggested_update")) or entry["latest_suggested_update"]
        case_id = _text(candidate.get("case_id"))
        if case_id and case_id not in entry["case_ids"]:
            entry["case_ids"].append(case_id)
        if latest_reviews.get(_text(candidate.get("candidate_id"))):
            entry["reviewed"] += 1

    rows = sorted(grouped.values(), key=lambda item: item["occurrences"], reverse=True)
    for row in rows:
        row["unreviewed"] = row["occurrences"] - row["reviewed"]
        # 繰り返し外していて、かつ誰も採否を付けていないものだけを先回り提示の対象にする
        row["alertable"] = row["occurrences"] >= REPEAT_ALERT_THRESHOLD and row["unreviewed"] > 0
    return rows[:REPEAT_BELIEF_LIMIT]


def build_candidate_review_rate(
    candidate_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reviewed_ids = {_text(review.get("candidate_id")) for review in review_rows if _text(review.get("candidate_id"))}
    by_decision: dict[str, int] = {}
    for review in review_rows:
        decision = _text(review.get("decision")) or "unknown"
        by_decision[decision] = by_decision.get(decision, 0) + 1
    total = len(candidate_rows)
    reviewed = sum(1 for candidate in candidate_rows if _text(candidate.get("candidate_id")) in reviewed_ids)
    ready = sum(1 for candidate in candidate_rows if _text(candidate.get("status")) == "ready_for_review")
    return {
        "total_candidates": total,
        "reviewed": reviewed,
        "ready_for_review": ready,
        "rate": _rate(reviewed, total),
        "by_decision": by_decision,
    }


def build_daily_review_focus(report: dict[str, Any]) -> list[str]:
    """朝に見るべきものを3つまでに絞る（増やさない）。"""
    focus: list[str] = []
    coverage = report["prediction_coverage"]
    if not coverage["trustworthy"]:
        focus.append(
            "事前予測が1件も記録されていない。まず審査を1件通して、"
            "prediction_snapshots.jsonl に記録が出るかを確認する。"
        )
    elif (coverage["coverage_rate"] or 0) < 0.5:
        focus.append(
            f"事前予測の割合が {coverage['coverage_rate']} と低い。"
            "結果登録時の逆算が多いため、キャリブレーションはまだ参考値として扱う。"
        )

    alertable = [row for row in report["repeat_beliefs"] if row.get("alertable")]
    if alertable:
        top = alertable[0]
        focus.append(
            f"「{top['target_belief']}」が {top['error_type']} で {top['occurrences']}回外れていて未レビュー。"
            "採否を1件つける。"
        )

    for warning in report["calibration_warnings"]:
        focus.append(warning["message"])

    confidence = report["confidence_calibration"]
    if confidence["trustworthy"] and (confidence["expected_calibration_error"] or 0) >= 0.15:
        focus.append(
            f"確信度の校正誤差が {confidence['expected_calibration_error']}。"
            "生のconfidenceは変更せず、経験的信頼度との乖離を人間レビューする。"
        )

    numeric = report["numeric_forecast"]
    numeric_mapes = [
        metric.get("mape")
        for metric in (numeric["sales_error"], numeric["op_profit_error"])
        if metric.get("mape") is not None
    ]
    if numeric["trustworthy"] and numeric_mapes and max(numeric_mapes) >= 0.3:
        focus.append(
            f"将来財務予測の最大MAPEが {max(numeric_mapes):.3f}。"
            "前提条件または予測方式を人間レビューする。"
        )

    if not focus:
        focus.append("予測の並びと候補レビューに未処理の異常なし。今日は何もしない。")
    return focus[:3]


def build_predictive_framework_report(
    *,
    snapshot_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    numeric_forecast_rows: list[dict[str, Any]] | None = None,
    numeric_actual_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    calibration = build_calibration(event_rows)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": "read_only_no_automatic_scoring_or_rag_change",
        "snapshots": build_snapshot_summary(snapshot_rows),
        "numeric_forecast": build_numeric_forecast_summary(
            numeric_forecast_rows or [],
            numeric_actual_rows or [],
        ),
        "prediction_coverage": build_prediction_coverage(event_rows),
        "calibration": calibration,
        "confidence_calibration": build_confidence_calibration(event_rows),
        "calibration_warnings": build_calibration_warnings(calibration),
        "surprise": build_surprise_rate(event_rows),
        "repeat_beliefs": build_repeat_beliefs(candidate_rows, review_rows),
        "candidate_review": build_candidate_review_rate(candidate_rows, review_rows),
    }
    report["daily_review_focus"] = build_daily_review_focus(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 予測的フレームワーク観測レポート")
    lines.append("")
    lines.append(f"生成: {report['generated_at']}")
    lines.append("")
    lines.append("読み取り専用の観測。スコアリング・プロンプト・判断資産は自動変更しない。")
    lines.append("")

    lines.append("## 今日見るところ")
    lines.append("")
    for item in report["daily_review_focus"]:
        lines.append(f"- {item}")
    lines.append("")

    coverage = report["prediction_coverage"]
    lines.append("## 予測カバー率")
    lines.append("")
    lines.append(f"- 観測済みイベント: {coverage['observed_events']} 件")
    lines.append(f"- うち審査時点の事前予測: {coverage['from_snapshot']} 件（{coverage['coverage_rate']}）")
    if not coverage["trustworthy"]:
        lines.append("- **事前予測が0件のため、以下のキャリブレーションはまだ根拠として使えない**")
    lines.append("")

    snapshots = report["snapshots"]
    lines.append("## 記録済みスナップショット")
    lines.append("")
    lines.append(f"- 合計 {snapshots['total']} 件 / 案件 {snapshots['unique_cases']} 件")
    if snapshots["by_risk_level"]:
        breakdown = " / ".join(f"{key}={value}" for key, value in sorted(snapshots["by_risk_level"].items()))
        lines.append(f"- リスク帯: {breakdown}")
    lines.append("")

    numeric = report["numeric_forecast"]
    lines.append("## 数値予測（将来売上・営業利益）")
    lines.append("")
    lines.append(f"- 記録 {numeric['total']} 件 / 案件 {numeric['unique_cases']} 件")
    if numeric["by_method"]:
        breakdown = " / ".join(f"{key}={value}" for key, value in sorted(numeric["by_method"].items()))
        lines.append(f"- 生成方式: {breakdown}")
    lines.append(
        f"- 実績 {numeric['actuals_latest']} 件 / 予測と突合 {numeric['matched_actuals']} 件"
        f"（突合率 {numeric['match_rate']}）"
    )
    lines.append("- 突合条件: 東京時間の実績日以前に固定された予測のうち最新のもの")
    if numeric["actuals_without_observation_date"]:
        lines.append(
            f"- 実績日なしで採点対象外: {numeric['actuals_without_observation_date']} 件"
        )
    if numeric["calibratable"]:
        sales = numeric["sales_error"]
        op_profit = numeric["op_profit_error"]
        lines.append(
            f"- 売上予測: n={sales['observations']} / MAE={sales['mae']} / "
            f"MAPE={sales['mape']} / 80%区間包含率={sales['interval_80_coverage']}"
        )
        lines.append(
            f"- 営業利益予測: n={op_profit['observations']} / MAE={op_profit['mae']} / "
            f"MAPE={op_profit['mape']} / 80%区間包含率={op_profit['interval_80_coverage']}"
        )
        if not numeric["trustworthy"]:
            lines.append(f"- 参考値: 信頼判定には突合 {numeric['minimum_matches_required']} 件以上が必要")
    else:
        lines.append("- 採点可能な実績なし。実績登録後もスコア・判定へは自動反映しない。")
    lines.append("")

    lines.append("## キャリブレーション（予測リスク帯 × 実績）")
    lines.append("")
    if report["calibration"]:
        lines.append("| 予測 | 件数 | 事前予測 | 成約 | 成約率 | 延滞 | 延滞率 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in report["calibration"]:
            lines.append(
                f"| {row['label']} | {row['events']} | {row['from_snapshot']} | "
                f"{row['contracted']} | {row['contract_rate']} | {row['delinquency']} | {row['delinquency_rate']} |"
            )
    else:
        lines.append("観測イベントなし。")
    lines.append("")

    if report["calibration_warnings"]:
        lines.append("### 要確認")
        lines.append("")
        for warning in report["calibration_warnings"]:
            lines.append(f"- {warning['message']}")
        lines.append("")

    confidence = report["confidence_calibration"]
    lines.append("## 確信度キャリブレーション（shadow）")
    lines.append("")
    lines.append(
        f"- 対象 {confidence['eligible_events']} 件 / 正解率 {confidence['accuracy']} / "
        f"平均生confidence {confidence['mean_raw_confidence']}"
    )
    lines.append(
        f"- Brier score {confidence['brier_score']} / "
        f"Expected calibration error {confidence['expected_calibration_error']}"
    )
    if confidence["bins"]:
        lines.append("| 生confidence帯 | 件数 | 平均confidence | 実績正解率 | 校正差 | 経験的confidence |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in confidence["bins"]:
            lines.append(
                f"| {row['range']} | {row['events']} | {row['mean_raw_confidence']} | "
                f"{row['observed_accuracy']} | {row['calibration_gap']} | {row['calibrated_confidence']} |"
            )
    if not confidence["trustworthy"]:
        lines.append(f"- 参考値: 信頼判定には対象 {confidence['minimum_events_required']} 件以上が必要")
    lines.append("- 生のconfidence・審査スコア・承認判定は変更しない。")
    lines.append("")

    surprise = report["surprise"]
    lines.append("## 驚き率（予測と実績のずれ）")
    lines.append("")
    lines.append(f"- {surprise['surprising']} / {surprise['total']} 件（{surprise['rate']}）")
    if surprise["by_error_type"]:
        for error_type, count in sorted(surprise["by_error_type"].items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {error_type}: {count} 件")
    lines.append("")

    lines.append("## 繰り返し外している前提")
    lines.append("")
    if report["repeat_beliefs"]:
        lines.append("| 前提 | 誤差種別 | 回数 | 未レビュー | 先回り対象 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in report["repeat_beliefs"]:
            mark = "○" if row.get("alertable") else ""
            lines.append(
                f"| {row['target_belief']} | {row['error_type']} | {row['occurrences']} | {row['unreviewed']} | {mark} |"
            )
    else:
        lines.append("候補なし。")
    lines.append("")

    review = report["candidate_review"]
    lines.append("## 候補レビュー状況")
    lines.append("")
    lines.append(f"- 候補 {review['total_candidates']} 件 / レビュー済み {review['reviewed']} 件（{review['rate']}）")
    lines.append(f"- レビュー待ち（ready_for_review）: {review['ready_for_review']} 件")
    if review["by_decision"]:
        breakdown = " / ".join(f"{key}={value}" for key, value in sorted(review["by_decision"].items()))
        lines.append(f"- 採否内訳: {breakdown}")
    lines.append("")

    return "\n".join(lines)


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--numeric-forecasts", type=Path, default=DEFAULT_NUMERIC_FORECASTS)
    parser.add_argument("--numeric-actuals", type=Path, default=DEFAULT_NUMERIC_ACTUALS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    report = build_predictive_framework_report(
        snapshot_rows=read_jsonl(args.snapshots.expanduser()),
        event_rows=read_jsonl(args.events.expanduser()),
        candidate_rows=read_jsonl(args.candidates.expanduser()),
        review_rows=read_jsonl(args.reviews.expanduser()),
        numeric_forecast_rows=read_jsonl(args.numeric_forecasts.expanduser()),
        numeric_actual_rows=read_jsonl(args.numeric_actuals.expanduser()),
    )
    write_outputs(report, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
