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
