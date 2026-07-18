"""chatgpt-codex-connector のレビュー指摘（PR #567 P1 badge / PR #566 P2 badge）への修正。

P1: frontend/src/app/financial/page.tsx が入力（百万円）を千円へ変換せずに
    /api/forecast へ送信すると予測が1000倍小さくなる問題。フロント側の修正は
    TypeScript のため直接テストできないが、components/financial_analysis.py の
    _m2k と同じ変換係数（×1000）で実装したことをここでは変換関数のロジックとして
    ドキュメント化する（実際の回帰確認は tsc + 目視）。

P2: Cloud Run改善メモが canonical_key/タイトル類似度で過去適用済みと一致すると、
    イベントの新旧を問わず常に抑制され、再発報告が改善ログ・トリアージに
    二度と現れなかった問題。適用時刻より後のイベントは「再発疑い」として
    抑制せず再入場させる。
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def main_module(tmp_path, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    return main


# ── P2: 過去適用済みとの一致 → 再発判定 ──────────────────────────────────────


def test_old_event_before_applied_is_still_suppressed(main_module, monkeypatch):
    """適用より前のイベント（旧イベント）は従来どおり抑制される（回帰させない）。"""
    main = main_module
    fake_home = _tmp_home(main, monkeypatch)
    ledger_dir = fake_home / "Library" / "Logs" / "tunelease"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "ledger.jsonl").write_text(
        json.dumps({
            "canonical_key": "misc_regression_key",
            "title": "画面のボタンが反応しない",
            "status": "applied",
            "recorded_at": "2026-07-10T00:00:00",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-old",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-05T00:00:00Z",  # 適用(07-10)より前 → 旧イベント
                "payload": {
                    "canonical_key": "misc_regression_key",
                    "title": "画面のボタンが反応しない",
                    "body": "課題: 画面のボタンが反応しない",
                },
            }
        ],
    )

    assert main._cloudrun_improvement_items_from_gcs() == []


def test_regression_after_applied_reenters_triage(main_module, monkeypatch):
    """適用より後のイベント（再発報告）は抑制されず、再発タグ付きで再入場する。"""
    main = main_module
    fake_home = _tmp_home(main, monkeypatch)
    ledger_dir = fake_home / "Library" / "Logs" / "tunelease"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "ledger.jsonl").write_text(
        json.dumps({
            "canonical_key": "misc_regression_key",
            "title": "画面のボタンが反応しない",
            "status": "applied",
            "recorded_at": "2026-07-10T00:00:00",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-new-regression",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-17T00:00:00Z",  # 適用(07-10)より後 → 再発疑い
                "payload": {
                    "canonical_key": "misc_regression_key",
                    "title": "画面のボタンが反応しない",
                    "body": "課題: 画面のボタンが反応しない",
                },
            }
        ],
    )

    items = main._cloudrun_improvement_items_from_gcs()

    assert len(items) == 1
    assert items[0]["possible_regression"] is True
    assert "再発報告の疑い" in items[0]["title"]
    assert items[0]["status"] == "NEEDS_REVIEW"  # 改善ログ・トリアージに再入場する


def test_regression_detected_via_fuzzy_title_match(main_module, monkeypatch):
    """canonical_key が一致しなくても、タイトル類似度(閾値0.70)経由の再発も検出する。"""
    main = main_module
    fake_home = _tmp_home(main, monkeypatch)
    ledger_dir = fake_home / "Library" / "Logs" / "tunelease"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "ledger.jsonl").write_text(
        json.dumps({
            "canonical_key": "misc_other_key",
            "title": "審査画面のスコア表示が崩れる不具合",
            "status": "applied",
            "recorded_at": "2026-07-01T00:00:00",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-fuzzy-regression",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-17T00:00:00Z",
                "payload": {
                    "canonical_key": "misc_totally_different_key",
                    "title": "審査画面のスコア表示が崩れる不具合",
                    "body": "課題: 審査画面のスコア表示が崩れる不具合",
                },
            }
        ],
    )

    items = main._cloudrun_improvement_items_from_gcs()

    assert len(items) == 1
    assert items[0]["possible_regression"] is True


def test_missing_timestamp_defaults_to_suppress(main_module, monkeypatch):
    """タイムスタンプ情報が無い場合は従来どおり安全側（抑制）に倒す。"""
    main = main_module
    fake_home = _tmp_home(main, monkeypatch)
    ledger_dir = fake_home / "Library" / "Logs" / "tunelease"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "ledger.jsonl").write_text(
        json.dumps({
            "canonical_key": "misc_no_ts_key",
            "title": "何かの不具合",
            "status": "applied",
            # recorded_at を意図的に欠落させる
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        main,
        "_read_recent_cloudrun_input_events_from_gcs",
        lambda days=45: [
            {
                "event_id": "event-no-ts",
                "event_type": "improvement_note",
                "surface": "chat_improvement",
                "ts": "2026-07-17T00:00:00Z",
                "payload": {"canonical_key": "misc_no_ts_key", "title": "何かの不具合"},
            }
        ],
    )

    assert main._cloudrun_improvement_items_from_gcs() == []


def _tmp_home(main, monkeypatch):
    """Path.home() を一時ディレクトリへ差し替える（読み取り専用テスト用）。"""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp))
    return tmp


# ── _historical_applied_timestamps / _matched_applied_timestamp 単体 ────────


def test_historical_applied_timestamps_last_entry_wins(main_module, monkeypatch):
    main = main_module
    fake_home = _tmp_home(main, monkeypatch)
    ledger_dir = fake_home / "Library" / "Logs" / "tunelease"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"canonical_key": "k1", "title": "件名A", "status": "applied", "recorded_at": "2026-07-01T00:00:00"},
        {"canonical_key": "k1", "title": "件名A", "status": "applied", "recorded_at": "2026-07-15T00:00:00"},
    ]
    (ledger_dir / "ledger.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8",
    )

    key_ts, title_ts = main._historical_applied_timestamps()

    assert key_ts["k1"] == "2026-07-15T00:00:00"
    assert title_ts["件名A"] == "2026-07-15T00:00:00"


def test_matched_applied_timestamp_uses_same_matching_as_is_implemented(main_module):
    main = main_module
    impl_ts = {"審査画面のスコア表示が崩れる不具合": "2026-07-01T00:00:00"}

    matched = main._matched_applied_timestamp("審査画面のスコア表示が崩れる不具合です", impl_ts, threshold=0.70)

    assert matched == "2026-07-01T00:00:00"
    assert main._matched_applied_timestamp("まったく無関係な文言", impl_ts, threshold=0.70) is None


def test_is_historically_applied_improvement_returns_pair(main_module):
    main = main_module
    result = main._is_historically_applied_improvement(
        title="表示ラベルの整理",
        body="",
        canonical_key="",
        applied_keys=set(),
        applied_titles=set(),
    )
    assert result == (False, False)
