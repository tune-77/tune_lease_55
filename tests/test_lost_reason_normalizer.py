from lost_reason_normalizer import normalize_competitor_fields, normalize_lost_reason


def test_normalize_lost_reason_rate_competition_ocr_variants():
    assert normalize_lost_reason("他社合（レート）", "失注") == "他社競合（レート）"
    assert normalize_lost_reason("金利で他社に負けた", "失注") == "他社競合（レート）"


def test_normalize_lost_reason_procurement_and_postpone_variants():
    assert normalize_lost_reason("調理方法次更く製・自己責金）", "失注") == "調達方法変更"
    assert normalize_lost_reason("設備投資見合せ（普通・延期）", "失注") == "設備見合わせ"


def test_blank_reason_only_becomes_missing_for_lost_case():
    assert normalize_lost_reason("0", "失注") == "理由未入力"
    assert normalize_lost_reason("0", "成約") == ""


def test_normalize_competitor_from_reason_and_name():
    assert normalize_competitor_fields("0", "", "他社競合（レート）") == ("競合あり", "")
    assert normalize_competitor_fields("0", "他リース会社", "") == ("競合あり", "他リース会社")
    assert normalize_competitor_fields("0", "現金", "調達方法変更") == ("競合なし", "")
