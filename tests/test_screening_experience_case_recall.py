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


def test_shion_review_prompt_excludes_past_company_citation():
    """REV-358: 紫苑レビューには過去事歴（類似経験ケース・過去レビュー本文・過去会社名）を渡さない。

    類似度計算が実質キーワード一致で無関係な会社を拾い、さらに本文へ機械的に社名を差し込んで
    いたため、レビューは今回案件の数値・定性情報・判断資産だけで書く方針に変更した。
    過去案件は入力アシスト（過去案件から作成）と経験ケースパネル側にのみ残す。
    """
    text = Path("frontend/src/lib/shionReview.ts").read_text(encoding="utf-8")

    # プロンプトへの過去事歴注入ブロックと、本文への強制挿入が復活していないこと
    assert "過去会社引用ルール" not in text
    assert "必ず過去会社名を1社以上明示" not in text
    assert "buildDemoSimilarPastCaseBlock" not in text
    assert "buildPastReviewBlock" not in text
    assert "ensurePastCompanyMentionInReview" not in text
    # 手元にない過去事例を紫苑が推測で捏造しないよう明示していること
    assert "過去の類似案件や他社事例は渡していません" in text


def test_shion_review_prompt_keeps_feedback_self_correction():
    """人間評価（薄い・推測が強すぎた等）だけは次回プロンプトへ反映する。

    渡すのはレビュー文の書き方への評価ラベルのみで、過去案件の社名・スコア・本文は渡さない。
    """
    text = Path("frontend/src/lib/shionReview.ts").read_text(encoding="utf-8")

    assert "buildReviewQualityFeedbackBlock" in text
    assert "【直近レビューへの人間評価】" in text


def test_shion_review_prompt_output_format_is_case_dependent():
    """固定4項目テンプレートをやめ、必須2項目＋案件ごとに選ぶ項目の形にしていること。"""
    text = Path("frontend/src/lib/shionReview.ts").read_text(encoding="utf-8")

    assert "必ず書くのは次の2項目だけです" in text
    assert "用意された項目を全部書かないでください" in text
