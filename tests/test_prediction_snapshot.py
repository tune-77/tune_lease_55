"""審査時点の予測スナップショット（予測誤差ループの前半）のテスト。"""
from __future__ import annotations

import pytest

from api import prediction_snapshot
from api.prediction_error_loop import build_case_result_prediction_error_payload
from api.prediction_snapshot import (
    build_prediction_snapshot,
    load_prediction_snapshot,
    record_saved_case_prediction,
    record_prediction_snapshot,
    summarize_prediction_snapshots,
)

_INPUTS = {
    "company_no": "0001",
    "company_name": "テスト工業",
    "industry_sub": "金属加工",
    "sales_dept": "第一営業部",
}
_RESULT = {
    "score": 52.0,
    "score_base": 52.0,
    "hantei": "要審議",
    "industry_sub": "金属加工",
    "main_concern": "自己資本比率が低い",
    "completeness_ratio": 0.8,
    "used_default_asset_score": False,
    "quantum_risk": 41.0,
}


def test_records_screening_time_prediction_and_reads_it_back(tmp_path, monkeypatch):
    path = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(prediction_snapshot, "SNAPSHOTS_PATH", path)

    recorded = record_prediction_snapshot(
        case_id="case-1",
        inputs=_INPUTS,
        result=_RESULT,
        final_status="未登録",
    )

    assert recorded["status"] == "recorded"
    snapshot = load_prediction_snapshot("case-1")
    assert snapshot is not None
    assert snapshot["status"] == "shadow_only"
    assert snapshot["prediction"]["risk_level"] == "high"
    assert snapshot["prediction"]["main_concern"] == "自己資本比率が低い"
    assert snapshot["confidence_basis"]["completeness_ratio"] == 0.8


def test_skips_when_outcome_is_already_known(tmp_path, monkeypatch):
    path = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(prediction_snapshot, "SNAPSHOTS_PATH", path)

    recorded = record_prediction_snapshot(
        case_id="case-batch",
        inputs=_INPUTS,
        result=_RESULT,
        final_status="成約",
    )

    assert recorded["status"] == "skipped"
    assert recorded["reason"] == "outcome_already_known"
    assert load_prediction_snapshot("case-batch") is None


def test_saved_case_adapter_records_unregistered_case(tmp_path):
    path = tmp_path / "snapshots.jsonl"

    recorded = record_saved_case_prediction(
        case_id="case-debate",
        case_data={"inputs": _INPUTS, "result": _RESULT, "final_status": "未登録"},
        source="debate_register_case",
        snapshots_path=path,
    )

    assert recorded["status"] == "recorded"
    snapshot = load_prediction_snapshot("case-debate", snapshots_path=path)
    assert snapshot is not None
    assert snapshot["source"] == "debate_register_case"


def test_saved_case_adapter_skips_known_batch_outcome(tmp_path):
    path = tmp_path / "snapshots.jsonl"

    recorded = record_saved_case_prediction(
        case_id="case-known",
        case_data={"inputs": _INPUTS, "result": _RESULT, "final_status": "失注"},
        source="batch_save",
        snapshots_path=path,
    )

    assert recorded == {"status": "skipped", "reason": "outcome_already_known"}
    assert load_prediction_snapshot("case-known", snapshots_path=path) is None


def test_never_raises_on_broken_input(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_snapshot, "SNAPSHOTS_PATH", tmp_path / "snapshots.jsonl")

    assert record_prediction_snapshot(case_id="", inputs=None, result=None)["status"] == "skipped"
    # 予測に使える項目が1つも無いケースも、例外ではなく skipped で返す
    assert record_prediction_snapshot(
        case_id="case-empty",
        inputs={},
        result={},
    )["status"] == "skipped"


def test_result_registration_prefers_snapshot_over_reconstruction(tmp_path, monkeypatch):
    path = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(prediction_snapshot, "SNAPSHOTS_PATH", path)
    record_prediction_snapshot(
        case_id="case-2",
        inputs=_INPUTS,
        result=_RESULT,
        final_status="未登録",
    )

    # 結果登録の時点では、スコアが更新されて別物になっていることがある。
    # それでも予測は審査時点の値でなければならない。
    payload = build_case_result_prediction_error_payload(
        case_data={
            "id": "case-2",
            "score": 88.0,
            "hantei": "承認圏内",
            "result": {"score": 88.0, "hantei": "承認圏内"},
        },
        status="成約",
        case_id="case-2",
    )

    prediction = payload["prediction"]
    assert prediction["prediction_source"] == "snapshot"
    assert prediction["risk_level"] == "high"
    assert prediction["main_concern"] == "自己資本比率が低い"
    assert prediction["predicted_at"]


def test_falls_back_to_reconstruction_when_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_snapshot, "SNAPSHOTS_PATH", tmp_path / "empty.jsonl")

    payload = build_case_result_prediction_error_payload(
        case_data={
            "id": "case-legacy",
            "score": 88.0,
            "hantei": "承認圏内",
            "result": {"score": 88.0, "hantei": "承認圏内"},
        },
        status="成約",
        case_id="case-legacy",
    )

    prediction = payload["prediction"]
    assert prediction["prediction_source"] == "reconstructed_at_result"
    assert prediction["risk_level"] == "low"


def test_summarize_counts_by_risk_level(tmp_path, monkeypatch):
    path = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(prediction_snapshot, "SNAPSHOTS_PATH", path)
    record_prediction_snapshot(case_id="case-a", inputs=_INPUTS, result=_RESULT, final_status="未登録")
    record_prediction_snapshot(
        case_id="case-b",
        inputs=_INPUTS,
        result={**_RESULT, "score": 95.0, "hantei": "承認圏内"},
        final_status="未登録",
    )

    summary = summarize_prediction_snapshots()

    assert summary["total"] == 2
    assert summary["unique_cases"] == 2
    assert summary["by_risk_level"] == {"high": 1, "low": 1}


def test_snapshot_keeps_confidence_formula_unchanged():
    snapshot = build_prediction_snapshot(case_id="case-x", inputs=_INPUTS, result=_RESULT)
    # 式は変えない方針なので、スコア52（条件付き承認ライン60から8点下）の値を固定する
    assert snapshot["prediction"]["confidence"] == pytest.approx(min(0.95, max(0.35, 8.0 / 60.0 + 0.35)))
