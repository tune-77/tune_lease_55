"""審査時点の予測を、結果を知る前に固定して保存する（予測誤差ループの前半）。

`api/prediction_error_loop.py` は結果登録時に `case_data` からスコアを逆算して
「予測だったこと」にしていた。この層は審査した瞬間の予測をそのまま残し、
後から本当の予測誤差を測れるようにする。

方針:
- スコアリング・判定・UI は一切変えない（記録するだけ）
- 記録失敗が審査応答を壊さないこと（例外は必ず握る）
- 記録形式は `api/chat_language_feedback.py` の response_impact_prediction に揃える
  （`schema_version` / `status: shadow_only` / `fingerprint` / `use_policy`）
- confidence の式は変えない。何を根拠にした値かを `confidence_basis` に残すだけ
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, load_jsonl

SNAPSHOTS_PATH = DATA_DIR / "prediction_snapshots.jsonl"
# 数値予測は判断予測と混ぜない（load_prediction_snapshot が誤って拾わないようにするため）
NUMERIC_FORECASTS_PATH = DATA_DIR / "prediction_numeric_forecasts.jsonl"

SCHEMA_VERSION = 1

_USE_POLICY = (
    "審査時点の予測を結果を知る前に固定するshadow mode。"
    "スコアや判定を変えるためではなく、後で予測誤差を測り、"
    "外れ続けている前提を人間レビューへ回すために使う。"
)

# 結果が未登録の案件だけを予測対象にする。バッチ取り込みの過去案件は
# 既に結果が判明しているため、予測として記録しない。
_UNREGISTERED_STATUSES = {"", "未登録", "未確定", "未定"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_confidence_basis(result: dict[str, Any]) -> dict[str, Any]:
    """confidence の値が何に支えられているかを残す（式そのものは変えない）。

    `CLAUDE.md` の要注意領域にある completeness_ratio / used_default_asset_score は、
    スコアがどれだけ実データに基づいているかの指標なので、後でキャリブレーションを
    見る時に「自信の根拠が薄い予測」を切り分けられるようにしておく。
    """
    result = result if isinstance(result, dict) else {}
    return {
        "formula": "min(0.95, max(0.35, abs(score - CONDITIONAL_LINE) / 60 + 0.35))",
        "completeness_ratio": _float_or_none(result.get("completeness_ratio")),
        "used_default_asset_score": bool(result.get("used_default_asset_score")),
        "quantum_risk": _float_or_none(result.get("quantum_risk")),
        "financial_consistency_score": _float_or_none(result.get("financial_consistency_score")),
        "credit_quantum_strong_warning": bool(result.get("credit_quantum_strong_warning")),
        "umap_anomaly_score": _float_or_none(result.get("umap_anomaly_score")),
    }


def _main_concern_at_screening(inputs: dict[str, Any], result: dict[str, Any]) -> str:
    """審査時点で「一番の懸念」として画面に出ていたものを1つ選ぶ。"""
    from api.prediction_error_loop import _first_text

    financial_risk = result.get("financial_consistency_risk")
    financial_reason = ""
    if isinstance(financial_risk, dict):
        financial_reason = _first_text(
            financial_risk.get("summary"),
            financial_risk.get("reasons"),
            financial_risk.get("items"),
        )
    return _first_text(
        result.get("main_concern"),
        financial_reason,
        result.get("conditional_approval_actions"),
        result.get("risk_summary"),
        result.get("diagnostic_recommendations"),
        inputs.get("industry_sub"),
    )


def build_prediction_snapshot(
    *,
    case_id: str,
    inputs: dict[str, Any] | None,
    result: dict[str, Any] | None,
    source: str = "score_full",
) -> dict[str, Any]:
    """審査結果から、後で採点できる形の予測を組み立てる。"""
    from api.prediction_error_loop import (
        _action_from_decision,
        _prediction_risk_from_score,
        prediction_confidence_from_score,
    )

    inputs = inputs if isinstance(inputs, dict) else {}
    result = result if isinstance(result, dict) else {}

    score = _float_or_none(result.get("score"))
    if score is None:
        score = _float_or_none(result.get("score_base"))
    hantei = _text(result.get("hantei"))

    used_assets = result.get("used_judgment_assets") or result.get("judgment_asset_ids") or []
    if isinstance(used_assets, str):
        used_assets = [used_assets]
    if not isinstance(used_assets, list):
        used_assets = []

    captured_at = _now_iso()
    fingerprint = hashlib.sha256(
        "\n".join(
            [
                _text(case_id),
                _text(inputs.get("company_no")),
                _text(result.get("industry_sub") or inputs.get("industry_sub")),
                "" if score is None else f"{score:.3f}",
                hantei,
            ]
        ).encode("utf-8")
    ).hexdigest()

    return {
        "id": fingerprint[:16],
        "schema_version": SCHEMA_VERSION,
        "event_type": "prediction_snapshot",
        "status": "shadow_only",
        "source": source,
        "case_id": _text(case_id),
        "captured_at": captured_at,
        "industry_sub": _text(result.get("industry_sub") or inputs.get("industry_sub")),
        "industry_major": _text(result.get("industry_major") or inputs.get("industry_major")),
        "sales_dept": _text(inputs.get("sales_dept")),
        "score": score,
        "score_base": _float_or_none(result.get("score_base")),
        "hantei": hantei,
        "prediction": {
            "risk_level": _prediction_risk_from_score(score),
            "main_concern": _main_concern_at_screening(inputs, result),
            "recommended_action": _action_from_decision(hantei, score),
            "confidence": prediction_confidence_from_score(score),
            "used_judgment_assets": used_assets,
        },
        "confidence_basis": build_confidence_basis(result),
        "fingerprint": fingerprint,
        "use_policy": _USE_POLICY,
    }


def record_prediction_snapshot(
    *,
    case_id: str,
    inputs: dict[str, Any] | None,
    result: dict[str, Any] | None,
    final_status: str = "",
    source: str = "score_full",
    snapshots_path: Path | None = None,
) -> dict[str, Any]:
    """審査時点の予測を追記する。失敗しても呼び出し元を壊さない。"""
    try:
        snapshots_path = snapshots_path or SNAPSHOTS_PATH
        if not _text(case_id):
            return {"status": "skipped", "reason": "case_id_unavailable"}
        if _text(final_status) not in _UNREGISTERED_STATUSES:
            return {"status": "skipped", "reason": "outcome_already_known"}

        snapshot = build_prediction_snapshot(
            case_id=case_id,
            inputs=inputs,
            result=result,
            source=source,
        )
        prediction = snapshot["prediction"]
        if not any(prediction.get(key) for key in ("risk_level", "main_concern", "recommended_action")):
            return {"status": "skipped", "reason": "prediction_fields_unavailable"}

        append_jsonl(snapshots_path, snapshot)
        return {"status": "recorded", "snapshot_id": snapshot["id"], "case_id": snapshot["case_id"]}
    except Exception as err:  # 記録失敗で審査応答を壊さない
        return {"status": "error", "reason": str(err)}


def record_saved_case_prediction(
    *,
    case_id: str,
    case_data: dict[str, Any] | None,
    source: str,
    snapshots_path: Path | None = None,
) -> dict[str, Any]:
    """`save_case_log()` 形式の案件から予測を記録する共通アダプター。

    通常審査以外の保存経路（討論・バッチ・Cloud Run帰還）も同じ予測誤差
    ループへ接続する。結果判明済みのバッチ行は `final_status` により既存の
    `record_prediction_snapshot()` が安全に除外する。
    """
    try:
        case_data = case_data if isinstance(case_data, dict) else {}
        inputs = case_data.get("inputs")
        result = case_data.get("result")
        inputs = inputs if isinstance(inputs, dict) else {}
        result = result if isinstance(result, dict) else {}
        final_status = _text(case_data.get("final_status") or result.get("final_status"))
        return record_prediction_snapshot(
            case_id=case_id,
            inputs=inputs,
            result=result,
            final_status=final_status,
            source=source,
            snapshots_path=snapshots_path,
        )
    except Exception as err:  # 補助記録で案件保存を壊さない
        return {"status": "error", "reason": str(err)}


def record_numeric_forecast(
    *,
    case_id: str,
    forecast: dict[str, Any],
    assumptions: dict[str, Any] | None = None,
    source: str = "future_simulation",
    forecasts_path: Path | None = None,
) -> dict[str, Any]:
    """審査時点の数値予測（将来売上・営業利益）を別台帳へ記録する。

    判断予測（`prediction_snapshots.jsonl`）とはファイルを分ける。混ぜると
    `load_prediction_snapshot()` が数値予測を最新の判断予測として拾ってしまうため。

    注意: この予測を採点するには3〜5年後の実績値が必要だが、現在のDBは
    成約/失注と延滞しか持っていない。よって当面は記録とカバー率の観測までで、
    キャリブレーションはできない。
    """
    try:
        path = forecasts_path or NUMERIC_FORECASTS_PATH
        if not _text(case_id):
            return {"status": "skipped", "reason": "case_id_unavailable"}
        if not isinstance(forecast, dict) or not forecast:
            return {"status": "skipped", "reason": "forecast_unavailable"}

        entry = {
            "schema_version": SCHEMA_VERSION,
            "event_type": "numeric_forecast_snapshot",
            "status": "shadow_only",
            "source": source,
            "case_id": _text(case_id),
            "captured_at": _now_iso(),
            "assumptions": assumptions or {},
            "forecast": {
                "method": forecast.get("method"),
                "unit": forecast.get("unit"),
                "years": forecast.get("years"),
                "deficit_prob": forecast.get("deficit_prob"),
                "final_op_median": forecast.get("final_op_median"),
                "final_op_worst10": forecast.get("final_op_worst10"),
            },
            "calibratable": False,
            "calibration_blocker": "future_actuals_not_collected",
            "use_policy": _USE_POLICY,
        }
        append_jsonl(path, entry)
        return {"status": "recorded", "case_id": entry["case_id"]}
    except Exception as err:
        return {"status": "error", "reason": str(err)}


def load_prediction_snapshot(
    case_id: str,
    *,
    snapshots_path: Path | None = None,
) -> dict[str, Any] | None:
    """指定案件の最新スナップショットを返す。無ければ None。"""
    target = _text(case_id)
    if not target:
        return None
    try:
        # load_jsonl は newest_first のため、最初に一致したものが最新になる
        for entry in load_jsonl(snapshots_path or SNAPSHOTS_PATH):
            if isinstance(entry, dict) and _text(entry.get("case_id")) == target:
                return entry
    except Exception:
        return None
    return None


def summarize_prediction_snapshots(
    *,
    snapshots_path: Path | None = None,
) -> dict[str, Any]:
    """記録済みスナップショットの件数内訳（レポート用の読み取り専用集計）。"""
    rows = load_jsonl(snapshots_path or SNAPSHOTS_PATH)
    by_risk: dict[str, int] = {}
    cases: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        cases.add(_text(row.get("case_id")))
        prediction = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        risk = _text(prediction.get("risk_level")) or "unknown"
        by_risk[risk] = by_risk.get(risk, 0) + 1
    cases.discard("")
    return {
        "total": len(rows),
        "unique_cases": len(cases),
        "by_risk_level": by_risk,
    }
