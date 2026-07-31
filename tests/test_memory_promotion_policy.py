from memory_promotion_policy import classify_memory_destination, classify_memory_promotion


def test_persistent_principle_promotes_to_persistent_layer():
    decision = classify_memory_promotion(
        "永続記憶として、紫苑の運用原則は記憶の寿命と役割を分けて扱うこと。"
    )

    assert decision.memory_layer == "persistent"
    assert decision.requires_review is False


def test_judgment_asset_candidate_is_review_gated():
    decision = classify_memory_promotion(
        "判断資産として、条件付き承認では未確認リスクを追加資料と撤退条件に分ける。"
    )

    assert decision.destination == "judgment_asset_candidate"
    assert decision.affects_judgment_assets is True
    assert decision.requires_review is True
    assert decision.memory_layer in {"mid_term", "long_term"}


def test_plain_question_does_not_become_judgment_asset_candidate():
    assert classify_memory_destination("判断資産は記憶システムと関係あるのか？") != "judgment_asset_candidate"


def test_improvement_request_stays_improvement_log():
    assert classify_memory_destination("判断資産グラフの表示を改善してほしい") == "improvement_log"
