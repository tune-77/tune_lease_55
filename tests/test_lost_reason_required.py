"""失注理由の必須化と一言（lost_reason_detail）の保存（REV-538）。"""
import pytest
from fastapi import BackgroundTasks, HTTPException

from api.main import CaseRegistration, CaseResultPatch, _lost_reason_detail, patch_case_result, register_case_result


def _capture_update(monkeypatch):
    """update_case の patches を記録し、失敗を返して後続の副作用（経験昇格・Vault 書き込み）に進ませない。"""
    captured = {}

    def fake_update(case_id, patches):
        captured[case_id] = patches
        return False

    # api.main は読み込み時に data_cases を入れ替えるので、名前で今のモジュールを差し替える
    monkeypatch.setattr("data_cases.update_case", fake_update)
    monkeypatch.setattr("data_cases.load_all_cases", lambda: [{"id": "c1"}])
    return captured


def test_register_lost_without_reason_is_rejected(monkeypatch):
    captured = _capture_update(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        register_case_result(CaseRegistration(case_id="c1", status="失注", lost_reason="  "), BackgroundTasks())
    assert exc.value.status_code == 400
    assert captured == {}


def test_register_lost_saves_reason_and_detail(monkeypatch):
    captured = _capture_update(monkeypatch)
    req = CaseRegistration(case_id="c1", status="失注", lost_reason="不明", lost_reason_detail=" 銀行の  融資に切替 " + "あ" * 60)
    with pytest.raises(HTTPException):  # fake_update が False を返すので 500
        register_case_result(req, BackgroundTasks())
    patches = captured["c1"]
    assert patches["lost_reason"] == "不明"
    assert patches["lost_reason_detail"].startswith("銀行の 融資に切替")
    assert len(patches["lost_reason_detail"]) == 40


def test_register_won_needs_no_reason_and_ignores_detail(monkeypatch):
    captured = _capture_update(monkeypatch)
    req = CaseRegistration(case_id="c1", status="成約", final_rate=2.5, lost_reason_detail="入れても保存しない")
    with pytest.raises(HTTPException):
        register_case_result(req, BackgroundTasks())
    assert "lost_reason" not in captured["c1"]
    assert "lost_reason_detail" not in captured["c1"]


def test_patch_saves_detail_and_keeps_old_clients(monkeypatch):
    captured = _capture_update(monkeypatch)
    with pytest.raises(HTTPException):
        patch_case_result("c1", CaseResultPatch(final_status="失注", loss_reason="調達方法変更", loss_reason_detail="自己資金"), BackgroundTasks())
    assert captured["c1"]["lost_reason_detail"] == "自己資金"
    # 一言を送らない古い呼び出しでは、項目を作らない
    with pytest.raises(HTTPException):
        patch_case_result("c1", CaseResultPatch(final_status="失注", loss_reason="不明"), BackgroundTasks())
    assert "lost_reason_detail" not in captured["c1"]


def test_detail_normalizes_whitespace():
    assert _lost_reason_detail(None) == ""
    assert _lost_reason_detail(" a\n b ") == "a b"
