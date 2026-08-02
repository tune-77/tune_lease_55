from api.screening_experience_promotion import promote_case_result_to_screening_experience


class Request:
    def __init__(self, **kwargs):
        self.payload = kwargs


def test_promote_case_result_to_screening_experience_builds_closed_win_case():
    saved = []

    result = promote_case_result_to_screening_experience(
        case_id="case-1",
        case_data={
            "inputs": {
                "company_name": "A社",
                "industry_sub": "工作機械",
                "asset_name": "マシニングセンタ",
                "customer_type": "既存",
            },
            "result": {"score": 62.5, "hantei": "条件付き承認"},
            "grey_judgment": {"but_still_reason": "受注根拠あり", "approval_condition_memo": "検収確認"},
        },
        status="成約",
        patches={"final_rate": 2.1},
        request_factory=Request,
        save_screening_experience_case=lambda req: saved.append(req.payload) or {"id": 1, **req.payload},
        experience_exists=lambda _case_id, _source: False,
    )

    assert result["status"] == "promoted"
    assert saved[0]["source_case_id"] == "case-1"
    assert saved[0]["company_name"] == "A社"
    assert saved[0]["outcome"] == "成約・最終金利 2.1%"
    assert "受注根拠あり" in saved[0]["action_taken"]
    assert saved[0]["form_snapshot"]["asset_name"] == "マシニングセンタ"


def test_promote_case_result_to_screening_experience_skips_open_or_duplicate_case():
    common = {
        "case_id": "case-1",
        "case_data": {"company_name": "A社"},
        "request_factory": Request,
        "save_screening_experience_case": lambda req: {"id": 1},
    }

    open_result = promote_case_result_to_screening_experience(
        **common,
        status="未登録",
        experience_exists=lambda _case_id, _source: False,
    )
    duplicate_result = promote_case_result_to_screening_experience(
        **common,
        status="失注",
        experience_exists=lambda _case_id, _source: True,
    )

    assert open_result == {"status": "skipped", "reason": "not_closed_result"}
    assert duplicate_result == {"status": "skipped", "reason": "already_promoted"}
