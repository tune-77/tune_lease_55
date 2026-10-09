from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPTS = ROOT / ".agents/skills/auto-improvement-pipeline/scripts"
if str(PIPELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SCRIPTS))

from improvement_identity import canonical_key  # noqa: E402

from scripts.check_aurion_state import append_to_export, detect_anomalies  # noqa: E402


def _exported_block(tmp_path, state_path_name: str, alerts: list[str]) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    export_file = tmp_path / "export.txt"
    import scripts.check_aurion_state as check_aurion_state

    original = check_aurion_state.EXPORT_FILE
    check_aurion_state.EXPORT_FILE = export_file
    try:
        append_to_export(alerts, Path(state_path_name))
    finally:
        check_aurion_state.EXPORT_FILE = original
    return export_file.read_text(encoding="utf-8")


def test_same_recurring_alert_collapses_to_one_canonical_key_across_days(tmp_path):
    alerts = ["DB 同期未完了 (status=None)"]
    text_day1 = _exported_block(tmp_path / "d1", "state_2026-09-10.json", alerts)
    text_day2 = _exported_block(tmp_path / "d2", "state_2026-09-14.json", alerts)

    title1, description1 = text_day1.split("\n", 1)
    title2, description2 = text_day2.split("\n", 1)

    # 出典ファイル名の日付だけが違っても、改善パイプラインの重複防止キー
    # (step1_extract_and_structure.py が使う canonical_key) は同一でなければ
    # ならない。ここが割れると同じ異常が日毎に新規REVとして発行され続ける。
    assert canonical_key(title1, description1) == canonical_key(title2, description2)


def test_different_alert_content_still_gets_distinct_canonical_key(tmp_path):
    text_db = _exported_block(
        tmp_path / "db", "state_2026-09-10.json", ["DB 同期未完了 (status=None)"]
    )
    text_qrisk = _exported_block(
        tmp_path / "qrisk",
        "state_2026-09-10.json",
        ["Q_risk が全件 0.0（計算停止の可能性, n=42）"],
    )

    title_db, description_db = text_db.split("\n", 1)
    title_qrisk, description_qrisk = text_qrisk.split("\n", 1)

    assert canonical_key(title_db, description_db) != canonical_key(
        title_qrisk, description_qrisk
    )


def test_detect_anomalies_flags_qrisk_stall():
    state = {
        "errors": [],
        "sync": {"status": "completed"},
        "vault_b_rag": {"status": "completed", "returncode": 0},
        "db": {
            "status": "completed",
            "q_risk": {"n": 42, "max_q": 0.0},
            "score_bands": [],
        },
    }

    alerts = detect_anomalies(state)

    assert any("Q_risk" in a for a in alerts)


def _alerts_env(tmp_path, monkeypatch):
    import scripts.check_aurion_state as cas

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))  # 本番 data/ に書かない
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(cas, "_ICLOUD_MAIN_VAULT_PATH", vault)
    return cas, vault / "Projects" / "tune_lease_55" / "Alerts"


def test_alerts_written_only_when_changed(tmp_path, monkeypatch):
    """REV-585: 同じ警告が続く日は Alerts を書かない。新規・悪化（文面の変化）・解消は必ず書く。"""
    cas, alert_dir = _alerts_env(tmp_path, monkeypatch)
    state = {"started_at": "x"}
    p = Path("state_x.json")

    assert cas.save_alert_file(["DB 同期未完了 (status=failed)"], p, state, today="2026-10-01") == "new"
    assert cas.save_alert_file(["DB 同期未完了 (status=failed)"], p, state, today="2026-10-02") is None
    assert not (alert_dir / "aurion_alert_2026-10-02.md").exists()

    # 悪化（文面が変わる）は書く。新しい側と消えた側の両方が載る
    worse = ["DB 同期未完了 (status=failed)", "Q_risk が全件 0.0（計算停止の可能性, n=12）"]
    assert cas.save_alert_file(worse, p, state, today="2026-10-03") == "new"
    note = (alert_dir / "aurion_alert_2026-10-03.md").read_text(encoding="utf-8")
    assert "新しく出た" in note and "Q_risk" in note and "継続中" in note

    # 解消は異常なしの日でも書く
    assert cas.save_alert_file([], p, state, today="2026-10-04") == "resolved"
    resolved = (alert_dir / "aurion_alert_2026-10-04.md").read_text(encoding="utf-8")
    assert "解消" in resolved and "検出されている異常はありません" in resolved
    assert cas.save_alert_file([], p, state, today="2026-10-05") is None


def test_unchanged_alert_is_rewritten_as_reminder_after_a_week(tmp_path, monkeypatch):
    cas, alert_dir = _alerts_env(tmp_path, monkeypatch)
    p, state = Path("state_x.json"), {}
    cas.save_alert_file(["RAG 異常"], p, state, today="2026-10-01")
    assert cas.save_alert_file(["RAG 異常"], p, state, today="2026-10-07") is None
    assert cas.save_alert_file(["RAG 異常"], p, state, today="2026-10-08") == "reminder"
    assert "2026-10-01 から" in (alert_dir / "aurion_alert_2026-10-08.md").read_text(encoding="utf-8")
