"""emotion_history の 30日トレンド補完（Cloud Run demo 対策）のテスト。

Cloud Run（揮発性・demoモード）では当日分しか記録されず、フロントの
EmotionTrendChart が「データが少なすぎます（1件）」で 30日トレンドを描画できない。
backfill_emotion_history が疎な履歴を過去日で補完し、かつ実データを上書きしない
ことを保証する。
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """一時 SQLite を指す DB モジュールを返す。"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_lease.db"))
    import api.database as database

    importlib.reload(database)
    database.init_emotion_history_table()
    return database


_BASE = {
    "hopeful_anxiety": 60.0,
    "careful_attachment": 55.0,
    "intellectual_excitement": 70.0,
    "unrewarded_effort": 30.0,
    "quiet_loneliness": 25.0,
    "earned_confidence": 50.0,
    "protective_frustration": 20.0,
}


def test_backfill_populates_trend_when_empty(db):
    inserted = db.backfill_emotion_history(_BASE, "hopeful_anxiety", days=30)
    # 当日分を除く 30 日分が入る。
    assert inserted == 30
    history = db.get_emotion_history(days=30)
    assert len(history) >= 2  # フロントがトレンドを描画できる件数
    # 値は 0..100 にクランプされている。
    for row in history:
        assert 0.0 <= row["hopeful_anxiety"] <= 100.0
        assert row["notes"] == "seed:demo-backfill"


def test_backfill_is_deterministic(db, tmp_path, monkeypatch):
    first = db.backfill_emotion_history(_BASE, "hopeful_anxiety", days=30)
    hist1 = db.get_emotion_history(days=30)
    assert first == 30

    # 別DBで同じ入力なら同じ値になる（決定論）。
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_lease2.db"))
    importlib.reload(db)
    db.init_emotion_history_table()
    db.backfill_emotion_history(_BASE, "hopeful_anxiety", days=30)
    hist2 = db.get_emotion_history(days=30)

    by_date1 = {r["recorded_at"][:10]: r["hopeful_anxiety"] for r in hist1}
    by_date2 = {r["recorded_at"][:10]: r["hopeful_anxiety"] for r in hist2}
    assert by_date1 == by_date2


def test_backfill_skips_when_history_dense(db):
    db.backfill_emotion_history(_BASE, "hopeful_anxiety", days=30)
    # 2 度目は既に十分な履歴があるので何もしない。
    again = db.backfill_emotion_history(_BASE, "hopeful_anxiety", days=30)
    assert again == 0


def test_backfill_does_not_overwrite_real_rows(db):
    # 実データを 1 件だけ入れておく（疎なので補完は走る）。
    real_scores = {k: 99.0 for k in _BASE}
    db.record_emotion_snapshot(real_scores, "earned_confidence", notes="real")

    db.backfill_emotion_history(_BASE, "hopeful_anxiety", days=30)
    history = db.get_emotion_history(days=30)

    real_rows = [r for r in history if r["notes"] == "real"]
    assert len(real_rows) == 1
    # 実データの値はそのまま（上書きされない）。
    assert real_rows[0]["hopeful_anxiety"] == 99.0
    # 実データの日付が補完行で重複していない。
    real_date = real_rows[0]["recorded_at"][:10]
    dup = [r for r in history if r["recorded_at"][:10] == real_date]
    assert len(dup) == 1


def test_backfill_no_base_scores_is_noop(db):
    assert db.backfill_emotion_history({}, "", days=30) == 0
    assert db.get_emotion_history(days=30) == []
