from api.chat_grey_judgment import (
    build_grey_judgment_prompt_block,
    case_qualitative_summary,
    load_gunshi_judgment_memory,
)


def test_case_qualitative_summary_collects_inputs():
    summary = case_qualitative_summary({
        "inputs": {
            "passion_text": "現場では設備更新が急務",
            "qual_corr_company_history": "長い",
            "intuition": 7,
        }
    })

    assert "現場メモ=現場では設備更新が急務" in summary
    assert "業歴=長い" in summary
    assert "直感スコア=7" in summary


def test_load_gunshi_judgment_memory_filters_relevant_rows():
    rows = [
        {"source": "other", "reason": "関係ない"},
        {"source": "gunshi_chat", "reason": "違和感を条件に変えた", "case_id": "c1", "score": 55},
    ]

    picked = load_gunshi_judgment_memory(training_candidates_loader=lambda **_: rows)

    assert len(picked) == 1
    assert picked[0]["case_id"] == "c1"
    assert picked[0]["reason"] == "違和感を条件に変えた"


def test_build_grey_judgment_prompt_block_uses_cases_and_gunshi_memory():
    cases = [
        {
            "id": "case-1",
            "company_name": "A社",
            "final_status": "成約",
            "score": 61,
            "hantei": "条件付き承認",
            "grey_judgment": {
                "human_discomfort": "数字は足りるが受注が薄い",
                "but_still_reason": "既存取引で回収可能",
                "approval_condition_memo": "受注書確認",
            },
        }
    ]
    gunshi = [{"source": "gunshi_chat", "case_id": "g1", "score": 50, "model_decision": "否決", "human_decision": "承認", "reason": "条件付きで通した", "review_status": "approved"}]

    block, payload = build_grey_judgment_prompt_block(
        "この条件付き承認の違和感をどう見る？",
        cases_loader=lambda: cases,
        gunshi_memory_loader=lambda limit=5: gunshi,
    )

    assert payload["used"] is True
    assert payload["refs"][0]["case_id"] == "case-1"
    assert "グレー判断の過去記憶" in block
    assert "受注書確認" in block
    assert "source=gunshi_chat" in block
