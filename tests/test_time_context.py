import datetime as dt

from api.context.time_context import (
    current_datetime_prompt_block,
    current_time_reply_if_requested,
)


def test_current_datetime_prompt_block_uses_jst():
    now = dt.datetime(2026, 7, 26, 15, 34, tzinfo=dt.timezone.utc)

    block = current_datetime_prompt_block(now)

    assert "【現在日時】2026年07月27日 00:34" in block
    assert "(JST, 2026-07-27, 月曜日)" in block


def test_current_time_reply_if_requested_returns_exact_jst_time():
    now = dt.datetime(2026, 7, 27, 9, 8, tzinfo=dt.timezone(dt.timedelta(hours=9)))

    reply = current_time_reply_if_requested("紫苑、今何時？", now)

    assert reply == (
        "今はJSTで09:08です。"
        "2026年07月27日（月曜日）です。"
        " 今日の審査や確認したい案件があれば、そのまま投げてください。"
    )


def test_current_time_reply_if_requested_ignores_non_time_questions():
    assert current_time_reply_if_requested("何時に集合する？") is None
