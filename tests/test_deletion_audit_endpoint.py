"""`/api/admin/deletion-audit` の読み取り専用エンドポイントの回帰テスト。

削除監査ログは「消えた案件の唯一の記録」なので、
取得経路が壊れても気づけるように次の3点を固定する。

- 記録済みの削除イベントがそのままJSONで読めること
- status / 日付でのフィルタがクエリパラメータから効くこと
- 不正な status や日付は 500 ではなく 422 で返ること（DB障害と入力ミスを区別する）
"""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import runtime_paths
    from api.db_connection import ensure_schema

    monkeypatch.setattr(runtime_paths, "get_db_path", lambda: str(tmp_path / "audit.db"))
    monkeypatch.setattr(runtime_paths, "ensure_cloudrun_demo_db_seeded", lambda: None)
    ensure_schema()

    import api.main as main_module

    return TestClient(main_module.app)


def _record_event(*, route, reason, case_ids, complete_with, create_parents=True):
    from api.db_connection import get_connection
    from case_deletion_audit import begin_case_deletion_event, complete_case_deletion_event

    with get_connection() as conn:
        if create_parents:
            for case_id in case_ids:
                conn.execute("INSERT INTO past_cases (id) VALUES (?)", (case_id,))
        event = begin_case_deletion_event(conn, case_ids, route=route, reason=reason)
        complete_case_deletion_event(conn, event, complete_with)
        conn.commit()
    return event


# ── 記録した削除イベントがそのまま読める ──────────────────────────────────
def test_returns_recorded_deletion_event(client):
    event = _record_event(
        route="api.case_delete",
        reason="api_request",
        case_ids=["20260926000000000001_aaaaaaaa"],
        complete_with=["20260926000000000001_aaaaaaaa"],
    )

    res = client.get("/api/admin/deletion-audit")

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["total"] == 1
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert body["filters"] == {"status": "", "date_from": "", "date_to": ""}
    recorded = body["events"][0]
    assert recorded["event_id"] == event.event_id
    assert recorded["route"] == "api.case_delete"
    assert recorded["reason"] == "api_request"
    assert recorded["status"] == "completed"
    assert recorded["deleted_count"] == 1
    assert recorded["items"] == [
        {
            "case_id": "20260926000000000001_aaaaaaaa",
            "parent_table": "past_cases",
            "status": "deleted",
        }
    ]


# ── イベントが無くても空配列を返す（404にしない） ────────────────────────
def test_returns_empty_list_without_events(client):
    res = client.get("/api/admin/deletion-audit")

    assert res.status_code == 200, res.text
    assert res.json()["total"] == 0
    assert res.json()["events"] == []


# ── status フィルタがクエリパラメータから効く ────────────────────────────
# 3つのstatusは別の事象を表す:
#   completed = 一致した全件が消えた
#   partial   = 印は付けたのに消えなかった行がある（孤児リスクが現実化した状態）
#   no_match  = 指定IDが最初から親テーブルに無い
def test_status_filter_narrows_events(client):
    _record_event(
        route="api.case_delete",
        reason="api_request",
        case_ids=["20260926000000000002_bbbbbbbb"],
        complete_with=["20260926000000000002_bbbbbbbb"],
    )
    _record_event(
        route="streamlit.form_status",
        reason="manual_full_delete",
        case_ids=["20260926000000000003_cccccccc"],
        complete_with=[],  # 一致したのに消えなかった → partial
    )
    _record_event(
        route="api.clear_all_pending_cases",
        reason="clear_pending_statuses",
        case_ids=["20260926000000000009_99999999"],
        complete_with=[],
        create_parents=False,  # 親が最初から無い → no_match
    )

    completed = client.get("/api/admin/deletion-audit", params={"status": "completed"})
    partial = client.get("/api/admin/deletion-audit", params={"status": "partial"})
    no_match = client.get("/api/admin/deletion-audit", params={"status": "no_match"})

    assert completed.json()["total"] == 1
    assert completed.json()["events"][0]["route"] == "api.case_delete"
    assert partial.json()["total"] == 1
    assert partial.json()["events"][0]["route"] == "streamlit.form_status"
    assert partial.json()["events"][0]["matched_count"] == 1
    assert partial.json()["events"][0]["deleted_count"] == 0
    assert no_match.json()["total"] == 1
    assert no_match.json()["events"][0]["route"] == "api.clear_all_pending_cases"
    assert no_match.json()["filters"]["status"] == "no_match"


# ── 未来日で絞れば0件になる（日付フィルタが素通りしていない） ────────────
def test_date_filter_is_applied(client):
    _record_event(
        route="api.case_delete",
        reason="api_request",
        case_ids=["20260926000000000004_dddddddd"],
        complete_with=["20260926000000000004_dddddddd"],
    )

    res = client.get("/api/admin/deletion-audit", params={"date_from": "2099-01-01"})

    assert res.status_code == 200, res.text
    assert res.json()["total"] == 0
    assert res.json()["filters"]["date_from"] == "2099-01-01"


# ── 入力ミスは 422（DB障害の500と混ぜない） ──────────────────────────────
@pytest.mark.parametrize(
    "params",
    [
        {"status": "deleted_everything"},  # 許可外のstatus
        {"date_from": "2026-13-99"},  # 日付として不正
        {"date_from": "2026-09-30", "date_to": "2026-09-01"},  # 期間が逆
    ],
)
def test_invalid_query_returns_422(client, params):
    res = client.get("/api/admin/deletion-audit", params=params)

    assert res.status_code == 422, res.text
