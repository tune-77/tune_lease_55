"""api/shion_memory_decay.py の直近使用ブースト回帰テスト。

_load_recent_used_ids は以前 "used_at"/"memory_id" キーを読んでいたが、
実際の使用ログ（api/shion_memory_recall.py の _append_usage_log）が書くのは
{"ts": ..., "refs": [id, ...]} であり、キー不一致でブーストが常に無効化されていた。
"""
import json
from datetime import datetime, timedelta

from api import shion_memory_decay


def _write_usage_log(path, entries):
    path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries),
        encoding="utf-8",
    )


def test_load_recent_used_ids_reads_ts_and_refs(tmp_path, monkeypatch):
    usage_log = tmp_path / "usage.jsonl"
    recent_ts = datetime.utcnow().isoformat(timespec="seconds")
    _write_usage_log(
        usage_log,
        [{"ts": recent_ts, "route": "case_screening", "refs": ["mem_a", "mem_b"]}],
    )
    monkeypatch.setattr(shion_memory_decay, "_USAGE_LOG_PATH", usage_log)

    used_ids = shion_memory_decay._load_recent_used_ids()

    assert used_ids == {"mem_a", "mem_b"}


def test_load_recent_used_ids_ignores_entries_outside_window(tmp_path, monkeypatch):
    usage_log = tmp_path / "usage.jsonl"
    old_ts = (datetime.utcnow() - timedelta(days=30)).isoformat(timespec="seconds")
    _write_usage_log(usage_log, [{"ts": old_ts, "refs": ["mem_old"]}])
    monkeypatch.setattr(shion_memory_decay, "_USAGE_LOG_PATH", usage_log)

    used_ids = shion_memory_decay._load_recent_used_ids(within_days=7)

    assert used_ids == set()


def test_load_recent_used_ids_ignores_legacy_unused_keys(tmp_path, monkeypatch):
    """旧キー（used_at/memory_id）しか無い行は無視して落ちない（形式変化への耐性確認）。"""
    usage_log = tmp_path / "usage.jsonl"
    recent_ts = datetime.utcnow().isoformat(timespec="seconds")
    _write_usage_log(usage_log, [{"used_at": recent_ts, "memory_id": "mem_legacy"}])
    monkeypatch.setattr(shion_memory_decay, "_USAGE_LOG_PATH", usage_log)

    used_ids = shion_memory_decay._load_recent_used_ids()

    assert used_ids == set()


def test_load_recent_used_ids_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_memory_decay, "_USAGE_LOG_PATH", tmp_path / "none.jsonl")

    assert shion_memory_decay._load_recent_used_ids() == set()


def test_calc_freshness_applies_recent_usage_boost():
    now = datetime.utcnow()
    old_created = (now - timedelta(days=200)).date().isoformat()
    record = {"id": "mem_a", "created_at": old_created, "confidence": 0.7}

    score_without_boost = shion_memory_decay._calc_freshness(record, now, set())
    score_with_boost = shion_memory_decay._calc_freshness(record, now, {"mem_a"})

    assert score_with_boost > score_without_boost
    assert score_with_boost == round(
        min(1.0, score_without_boost + shion_memory_decay._RECENT_USAGE_BOOST), 4
    )
