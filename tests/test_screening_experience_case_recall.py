from pathlib import Path


def test_screening_experience_scoring_accepts_partial_industry_labels():
    from api.main import _SCREENING_EXPERIENCE_DEMO_SEEDS, _score_screening_experience_case

    seed = next(item for item in _SCREENING_EXPERIENCE_DEMO_SEEDS if item["company_name"] == "柴犬精密工業")
    scored = _score_screening_experience_case(
        dict(seed),
        {
            "industry_major": "製造業",
            "industry_sub": "",
            "asset_name": "工作機械",
            "score": 80,
        },
    )

    assert scored["similarity_score"] > 0
    assert "同じ大分類業種" in scored["similarity_reasons"]


def test_shion_review_prompt_requires_past_company_citation():
    text = Path("frontend/src/app/screening/page.tsx").read_text(encoding="utf-8")

    assert "過去会社引用ルール" in text
    assert "必ず過去会社名を1社以上明示" in text
    assert "今回案件との差分" in text
