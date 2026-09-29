from api.prompt_generator import build_shion_system_prompt, build_system_prompt


def test_shion_prompt_includes_start_date_and_day_count():
    prompt = build_shion_system_prompt({}, "2026-09-29 10:00")
    assert "稼働開始日: 2026-06-12（今日で110日目）" in prompt


def test_first_day_and_invalid_or_earlier_dates():
    assert "今日で1日目" in build_shion_system_prompt({}, "2026-06-12 09:00")
    assert "稼働開始日" not in build_shion_system_prompt({}, "2026-06-11 09:00")
    assert "稼働開始日" not in build_shion_system_prompt({}, "不明")


def test_mebuki_prompt_is_unchanged():
    assert "稼働開始日" not in build_system_prompt({}, "2026-09-29 10:00")
