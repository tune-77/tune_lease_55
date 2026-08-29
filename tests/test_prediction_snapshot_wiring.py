"""通常審査以外の案件保存経路も予測誤差ループへ接続されることを検証する。"""

from api import prediction_snapshot
from api.routers import debate, pipeline_misc


def test_debate_registration_records_screening_prediction(monkeypatch):
    calls = []
    monkeypatch.setattr("data_cases.save_case_log", lambda _case_data: "debate-1")
    monkeypatch.setattr(
        prediction_snapshot,
        "record_saved_case_prediction",
        lambda **kwargs: calls.append(kwargs) or {"status": "recorded"},
    )

    response = debate.register_debate_case(
        debate.RegisterDebateCaseRequest(
            company_name="討論テスト社",
            industry_major="製造業",
            score=72,
            final_decision="条件付承認",
            arbiter_summary="受注根拠を追加確認",
            inputs={"sales_dept": "本店"},
        )
    )

    assert response["case_id"] == "debate-1"
    assert calls[0]["source"] == "debate_register_case"
    assert calls[0]["case_data"]["result"]["hantei"] == "条件付承認"


def test_batch_save_records_only_after_case_save_succeeds(monkeypatch):
    calls = []
    case_ids = iter(["batch-1", None])
    monkeypatch.setattr("data_cases.save_case_log", lambda _case_data: next(case_ids))
    monkeypatch.setattr("data_cases.save_excluded_grade_case", lambda _case_data: None)
    monkeypatch.setattr(
        prediction_snapshot,
        "record_saved_case_prediction",
        lambda **kwargs: calls.append(kwargs) or {"status": "recorded"},
    )

    saved, with_result, excluded = pipeline_misc._save_batch_payloads(
        [
            {"inputs": {"company_no": "1"}, "result": {"score": 65}, "final_status": "未登録"},
            {"inputs": {"company_no": "2"}, "result": {"score": 55}, "final_status": "未登録"},
        ],
        [],
    )

    assert (saved, with_result, excluded) == (1, 0, 0)
    assert len(calls) == 1
    assert calls[0]["case_id"] == "batch-1"
    assert calls[0]["source"] == "batch_save"


def test_cloudrun_promotion_records_prediction(monkeypatch):
    import api.cloudrun_pending_cases as cloudrun_pending_cases
    import api.main as main

    calls = []
    case_payload = {
        "inputs": {"company_no": "cloud-1"},
        "result": {"score": 81, "hantei": "承認圏内"},
        "final_status": "未登録",
    }
    monkeypatch.setattr(main, "_load_cloudrun_score_input", lambda _score_id: {"id": 7})
    monkeypatch.setattr(
        cloudrun_pending_cases,
        "build_score_input_case_payload",
        lambda _row, _score_id: case_payload,
    )
    monkeypatch.setattr("data_cases.save_case_log", lambda _case_data: "cloud-case-1")
    monkeypatch.setattr(
        prediction_snapshot,
        "record_saved_case_prediction",
        lambda **kwargs: calls.append(kwargs) or {"status": "recorded"},
    )

    class _BrokenConnection:
        def __enter__(self):
            raise RuntimeError("return DB unavailable")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(main, "_connect_cloudrun_return_db", lambda: _BrokenConnection())

    assert main._promote_cloudrun_score_input_to_pending_case(7) == "cloud-case-1"
    assert calls == [
        {
            "case_id": "cloud-case-1",
            "case_data": case_payload,
            "source": "cloudrun_score_promotion",
        }
    ]
