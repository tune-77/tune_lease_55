"""screening_records への記録経路の回帰テスト。

Next.js + FastAPI へ移行したあと、`/api/score/full` が past_cases にしか
書かず screening_records が伸びなくなっていた（Streamlit 側は `_api_mode` の
とき記録をスキップし、API 側に委ねるコメントだけが残っていた）。
ここでは「API経由の審査が記録される」「ステータス確定が outcome に反映される」
「統計集計バッチの DB パスが実在する場所を指す」の3点を固定する。
"""
import sqlite3

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import api.main as main_module  # noqa: E402
import data_cases  # noqa: E402


def _rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT case_id, source, total_score, outcome FROM screening_records"
        ).fetchall()


# ── API経由の審査が screening_records に記録される ────────────────────────
def test_score_full_records_screening_result(tmp_path, monkeypatch):
    db = str(tmp_path / "lease_data.db")
    monkeypatch.setattr(main_module, "_LEASE_DB_PATH", db)
    # save_case_log が本物の lease_data.db を汚さないようにする。
    # api/main.py は sys.modules["data_cases"] を自前ロードで差し替えるため、
    # ここで import した data_cases と同一オブジェクトになる。
    monkeypatch.setattr(data_cases, "DB_PATH", db)

    client = TestClient(main_module.app)
    res = client.post(
        "/api/score/full",
        json={"company_name": "テスト商事", "nenshu": 500, "rieki": 20},
    )
    assert res.status_code == 200, res.text

    rows = _rows(db)
    assert len(rows) == 1, rows
    case_id, source, total_score, outcome = rows[0]
    assert source == "api"
    assert case_id == res.json()["case_id"]
    assert total_score == pytest.approx(res.json()["score"])
    assert outcome is None  # 結果はまだ出ていない


# ── スコアが範囲外でも記録が落ちない（BR-403 に弾かれない） ────────────────
def test_record_task_clamps_out_of_range_scores(tmp_path, monkeypatch):
    db = str(tmp_path / "lease_data.db")
    monkeypatch.setattr(main_module, "_LEASE_DB_PATH", db)

    main_module._record_screening_result_task(
        "case-clamp",
        {"score": 130.0, "asset_score": -5.0, "score_borrower": None, "quantum_risk": None},
    )

    assert _rows(db) == [("case-clamp", "api", 100.0, None)]


# ── 記録失敗は審査応答を壊さない ──────────────────────────────────────────
def test_record_task_never_raises(monkeypatch):
    monkeypatch.setattr(main_module, "_LEASE_DB_PATH", "/nonexistent\x00/bad.db")
    # 例外が漏れたらここで落ちる
    main_module._record_screening_result_task("case-x", {"score": 50.0})


# ── ステータス確定が outcome に反映される ────────────────────────────────
@pytest.mark.parametrize(
    "final_status, expected",
    [("成約", "contracted"), ("検収", "contracted"), ("検収完了", "completed"), ("失注", "lost")],
)
def test_sync_screening_outcome_maps_status(tmp_path, monkeypatch, final_status, expected):
    db = str(tmp_path / "lease_data.db")
    monkeypatch.setattr(data_cases, "DB_PATH", db)
    from screening_recorder import record_screening_result

    record_screening_result(
        case_id="case-1",
        screened_at="2026-09-03T00:00:00Z",
        total_score=70.0,
        asset_score=60.0,
        source="api",
        db_path=db,
    )

    data_cases._sync_screening_outcome("case-1", final_status)

    assert _rows(db)[0][3] == expected


@pytest.mark.parametrize("final_status", ["未登録", "スコアリングのみ", "稟議中", "", None])
def test_sync_screening_outcome_ignores_undecided_status(tmp_path, monkeypatch, final_status):
    db = str(tmp_path / "lease_data.db")
    monkeypatch.setattr(data_cases, "DB_PATH", db)
    from screening_recorder import record_screening_result

    record_screening_result(
        case_id="case-1",
        screened_at="2026-09-03T00:00:00Z",
        total_score=70.0,
        asset_score=60.0,
        source="api",
        db_path=db,
    )

    data_cases._sync_screening_outcome("case-1", final_status)

    assert _rows(db)[0][3] is None


# ── 案件更新の入口から outcome 同期が呼ばれる ────────────────────────────
def _seed_past_case(db_path, case_id="case-1"):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE past_cases (id TEXT PRIMARY KEY, timestamp TEXT, "
            "industry_sub TEXT, score REAL, user_eq REAL, final_status TEXT, "
            "data TEXT, sales_dept TEXT)"
        )
        conn.execute(
            "INSERT INTO past_cases (id, timestamp, final_status, data) VALUES (?,?,?,?)",
            (case_id, "2026-09-03T00:00:00", "未登録", '{"id": "case-1"}'),
        )


def test_update_case_field_syncs_outcome(tmp_path, monkeypatch):
    db = str(tmp_path / "lease_data.db")
    _seed_past_case(db)
    monkeypatch.setattr(data_cases, "DB_PATH", db)
    monkeypatch.setattr(data_cases, "refresh_stats_caches", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        data_cases, "_sync_screening_outcome", lambda cid, status: calls.append((cid, status))
    )

    assert data_cases.update_case_field("case-1", "final_status", "成約") is True
    assert calls == [("case-1", "成約")]

    calls.clear()
    assert data_cases.update_case_field("case-1", "industry_sub", "06 総合工事業") is True
    assert calls == []  # ステータス以外の更新では同期しない


def test_update_case_syncs_outcome(tmp_path, monkeypatch):
    db = str(tmp_path / "lease_data.db")
    _seed_past_case(db)
    monkeypatch.setattr(data_cases, "DB_PATH", db)
    monkeypatch.setattr(data_cases, "refresh_stats_caches", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        data_cases, "_sync_screening_outcome", lambda cid, status: calls.append((cid, status))
    )

    assert data_cases.update_case("case-1", {"final_status": "失注"}) is True
    assert calls == [("case-1", "失注")]


# ── 統計集計バッチが実在する DB パスを見ている ───────────────────────────
def test_aggregate_batch_points_at_real_db_paths():
    from runtime_paths import get_data_path, get_db_path
    from scripts import aggregate_stats_from_past_cases as agg

    assert agg._LEASE_DB == get_db_path()
    assert agg._SCREENING_DB == get_data_path("screening_db.sqlite")

    # customer_db が読む DB と同じ場所を書いていること
    import customer_db

    assert agg._SCREENING_DB == customer_db.get_db_path()
