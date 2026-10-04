import threading
import time
from pathlib import Path

import lease_news_digest


def _write_news_note(directory: Path, date: str, frontmatter_date: str) -> Path:
    path = directory / f"{date}_業界リスクニュース_test.md"
    path.write_text(f"---\ndate: {frontmatter_date}\n---\n", encoding="utf-8")
    return path


def test_latest_news_note_prefers_newest_frontmatter_date(tmp_path):
    news_dir = tmp_path / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    merged = _write_news_note(news_dir, "2026-10-01", "2026-10-05")
    _write_news_note(news_dir, "2026-10-04", "2026-10-04")

    assert lease_news_digest._latest_news_note(tmp_path) == merged


def test_latest_news_note_times_out_to_filename_date(tmp_path, monkeypatch):
    news_dir = tmp_path / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    newest = _write_news_note(news_dir, "2026-10-05", "2026-10-05")
    _write_news_note(news_dir, "2026-10-04", "2026-10-06")

    def slow_frontmatter_date(path):
        time.sleep(0.5)
        return "2026-10-06"

    monkeypatch.setattr(lease_news_digest, "_note_frontmatter_date", slow_frontmatter_date)
    monkeypatch.setattr(lease_news_digest, "_NEWS_NOTE_READ_TIMEOUT_S", 0.2)

    started = time.monotonic()
    selected = lease_news_digest._latest_news_note(tmp_path)

    assert time.monotonic() - started < 1.0
    assert selected == newest
    deadline = time.monotonic() + 1.0
    while lease_news_digest._NEWS_NOTE_READ_LOCK.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not lease_news_digest._NEWS_NOTE_READ_LOCK.locked()


def test_latest_news_note_does_not_spawn_workers_while_previous_read_is_blocked(tmp_path, monkeypatch):
    news_dir = tmp_path / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    newest = _write_news_note(news_dir, "2026-10-05", "2026-10-05")
    gate = threading.Event()
    entered = threading.Event()
    workers: list[threading.Thread] = []
    real_thread = threading.Thread

    def blocked_frontmatter_date(path):
        entered.set()
        gate.wait(timeout=2)
        return "2026-10-05"

    def recording_thread(*args, **kwargs):
        worker = real_thread(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(lease_news_digest, "_note_frontmatter_date", blocked_frontmatter_date)
    monkeypatch.setattr(lease_news_digest, "_NEWS_NOTE_READ_TIMEOUT_S", 0.05)
    monkeypatch.setattr(lease_news_digest.threading, "Thread", recording_thread)

    try:
        assert lease_news_digest._latest_news_note(tmp_path) == newest
        assert entered.wait(timeout=1)
        assert lease_news_digest._latest_news_note(tmp_path) == newest
        assert len(workers) == 1
    finally:
        gate.set()
        for worker in workers:
            worker.join(timeout=1)
