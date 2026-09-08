"""日次DB件数レポートの前日比・据置検知テスト。

screening_records が API 移行で書き込まれなくなった件（REV-369）は、件数自体は
毎朝 Slack に出ていたのに数ヶ月気づけなかった。出していたのが「水位」だけで
「変化量」ではなかったため。ここではその再発検知を固定する。
"""
import json

import pytest

from scripts import aurion_core_daily as acd


def _write_state(state_dir, date: str, counts: dict[str, int]) -> None:
    payload = {
        "db": {"counts": [{"table_name": t, "n": n} for t, n in counts.items()]}
    }
    (state_dir / f"state_{date}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(acd, "STATE_DIR", tmp_path)
    monkeypatch.setattr(acd, "date_str", lambda: "2026-09-10")
    return tmp_path


# ── 履歴の読み取り ────────────────────────────────────────────────────────
def test_count_history_is_newest_first_and_excludes_today(state_dir):
    _write_state(state_dir, "2026-09-08", {"past_cases": 100})
    _write_state(state_dir, "2026-09-09", {"past_cases": 110})
    # 当日分は midnight が数時間前に書いたもの。前日比にならないので除く
    _write_state(state_dir, "2026-09-10", {"past_cases": 111})

    history = acd._count_history()

    assert [snap["past_cases"] for snap in history] == [110, 100]


def test_count_history_survives_broken_state_file(state_dir):
    _write_state(state_dir, "2026-09-09", {"past_cases": 110})
    (state_dir / "state_2026-09-08.json").write_text("{ broken", encoding="utf-8")

    assert [snap["past_cases"] for snap in acd._count_history()] == [110]


def test_count_history_empty_when_no_state(state_dir):
    assert acd._count_history() == []


# ── 前日比の付与 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "today, previous, expected_delta, expected_label",
    [
        (110, 100, 10, "+10"),
        (100, 100, 0, "±0"),
        (95, 100, -5, "-5"),
    ],
)
def test_annotate_count_deltas_labels(today, previous, expected_delta, expected_label):
    rows = acd.annotate_count_deltas(
        [{"table_name": "past_cases", "n": today}], history=[{"past_cases": previous}]
    )
    assert rows[0]["delta"] == expected_delta
    assert rows[0]["delta_label"] == expected_label


def test_annotate_count_deltas_without_history_is_unknown():
    rows = acd.annotate_count_deltas([{"table_name": "past_cases", "n": 100}], history=[])
    assert rows[0]["delta"] is None
    assert rows[0]["delta_label"] == "—"
    assert rows[0]["flat_days"] == 0


def test_annotate_count_deltas_keeps_original_keys():
    rows = acd.annotate_count_deltas(
        [{"table_name": "past_cases", "n": 100}], history=[{"past_cases": 100}]
    )
    # Slack行・Markdown表が参照する既存キーを壊さない
    assert rows[0]["table_name"] == "past_cases"
    assert rows[0]["n"] == 100


# ── 据置日数 ──────────────────────────────────────────────────────────────
def test_flat_days_counts_consecutive_unchanged_days():
    history = [{"x": 50}, {"x": 50}, {"x": 50}, {"x": 42}]
    rows = acd.annotate_count_deltas([{"table_name": "x", "n": 50}], history=history)
    assert rows[0]["flat_days"] == 3


def test_flat_days_resets_when_count_moved_yesterday():
    history = [{"x": 42}, {"x": 42}, {"x": 42}]
    rows = acd.annotate_count_deltas([{"table_name": "x", "n": 50}], history=history)
    assert rows[0]["flat_days"] == 0


# ── 据置アラート ──────────────────────────────────────────────────────────
def test_stagnant_tables_flags_only_over_threshold():
    counts = [
        {"table_name": "moving", "n": 10, "flat_days": 0},
        {"table_name": "slow", "n": 20, "flat_days": acd.STAGNATION_ALERT_DAYS - 1},
        {"table_name": "stuck", "n": 30, "flat_days": acd.STAGNATION_ALERT_DAYS},
    ]
    assert [row["table_name"] for row in acd.stagnant_tables(counts)] == ["stuck"]


def test_stagnation_note_warns_with_table_and_days():
    counts = [{"table_name": "screening_records", "n": 2109, "flat_days": 9}]
    note = acd._stagnation_note(counts)
    assert "screening_records" in note
    assert "9日" in note


def test_stagnation_note_is_quiet_when_healthy():
    counts = [{"table_name": "past_cases", "n": 100, "flat_days": 0}]
    assert "⚠️" not in acd._stagnation_note(counts)


# ── 回帰: REV-369 の壊れ方を検知できること ────────────────────────────────
def test_detects_the_rev369_breakage_shape(state_dir):
    """past_cases は増え続け screening_records だけ止まる、を検知する。"""
    for day, past in [("2026-09-06", 100), ("2026-09-07", 104), ("2026-09-08", 109), ("2026-09-09", 115)]:
        _write_state(state_dir, day, {"past_cases": past, "screening_records": 2109})

    rows = acd.annotate_count_deltas(
        [
            {"table_name": "past_cases", "n": 121},
            {"table_name": "screening_records", "n": 2109},
        ]
    )
    by_table = {row["table_name"]: row for row in rows}

    assert by_table["past_cases"]["delta_label"] == "+6"
    assert by_table["past_cases"]["flat_days"] == 0
    assert by_table["screening_records"]["delta_label"] == "±0"
    assert by_table["screening_records"]["flat_days"] == 4
    assert [r["table_name"] for r in acd.stagnant_tables(rows)] == ["screening_records"]
