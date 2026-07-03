import json
from datetime import date

from scripts.update_shion_memory_freshness import (
    apply_freshness,
    load_usage_dates,
    load_usage_dates_from_cloudrun_events,
    merge_usage_dates,
)


def test_load_usage_dates_returns_latest_per_ref(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-06-01T10:00:00", "refs": ["mem_a", "mem_b"]}),
                json.dumps({"ts": "2026-06-20T09:00:00", "refs": ["mem_a"]}),
                "broken line {",
            ]
        ),
        encoding="utf-8",
    )

    usage = load_usage_dates(log)

    assert usage["mem_a"] == "2026-06-20"
    assert usage["mem_b"] == "2026-06-01"


def test_load_usage_dates_missing_file(tmp_path):
    assert load_usage_dates(tmp_path / "none.jsonl") == {}


def test_load_usage_dates_from_cloudrun_events(tmp_path):
    events_dir = tmp_path / "cloudrun_inputs"
    events_dir.mkdir()
    (events_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "shion_memory_usage",
                        "ts": "2026-07-02T01:00:00+00:00",
                        "payload": {"ts": "2026-07-02T10:00:00", "route": "case_screening", "refs": ["mem_cloud"]},
                    }
                ),
                # 別種イベントは無視される
                json.dumps({"event_type": "wizard_input", "ts": "2026-07-02T01:00:00+00:00", "payload": {}}),
            ]
        ),
        encoding="utf-8",
    )

    usage = load_usage_dates_from_cloudrun_events(events_dir)

    assert usage == {"mem_cloud": "2026-07-02"}


def test_load_usage_dates_from_cloudrun_events_missing_dir(tmp_path):
    assert load_usage_dates_from_cloudrun_events(tmp_path / "none") == {}


def test_merge_usage_dates_keeps_latest():
    merged = merge_usage_dates(
        {"mem_a": "2026-06-01", "mem_b": "2026-06-10"},
        {"mem_a": "2026-06-20"},
    )
    assert merged == {"mem_a": "2026-06-20", "mem_b": "2026-06-10"}


def _record(rid, *, memory_type="judgment_memory", status="active", created_at="2026-01-01", last_used_at=""):
    return {
        "id": rid,
        "content": "x",
        "memory_type": memory_type,
        "status": status,
        "created_at": created_at,
        "last_used_at": last_used_at,
    }


def test_apply_freshness_demotes_old_unused_and_updates_last_used():
    index = {
        "records": [
            _record("old_unused"),
            _record("old_used"),
            _record("value_old", memory_type="value_memory"),
            _record("recent", created_at="2026-06-25"),
            _record("stale_revived", status="stale"),
            _record("deprecated_kept", status="deprecated"),
        ]
    }
    usage = {"old_used": "2026-06-28", "stale_revived": "2026-06-30"}

    summary = apply_freshness(index, usage, stale_days=45, today=date(2026, 7, 3))

    by_id = {r["id"]: r for r in index["records"]}
    # 45日超未使用の active は stale へ
    assert by_id["old_unused"]["status"] == "stale"
    # 直近使用があれば active のまま、last_used_at が更新される
    assert by_id["old_used"]["status"] == "active"
    assert by_id["old_used"]["last_used_at"] == "2026-06-28"
    # 上位規範（value_memory）は経年で stale に落とさない
    assert by_id["value_old"]["status"] == "active"
    # 作成から stale-days 未満の記憶は未使用でも落とさない
    assert by_id["recent"]["status"] == "active"
    # stale でも直近使用があれば active に戻す
    assert by_id["stale_revived"]["status"] == "active"
    # revised / deprecated / private は鮮度で動かさない
    assert by_id["deprecated_kept"]["status"] == "deprecated"

    assert summary["demoted_to_stale"] == 1
    assert summary["revived_to_active"] == 1
    assert summary["last_used_updated"] == 2
