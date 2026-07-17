"""Phase 1（紫苑中心・改善ループ移行計画）: トリアージの構造化記録。

- POST /api/improvement/triage は追記形式で記録し、最後のエントリが有効
- 同定は canonical_key + source_event_id + タイトルスナップショットで冗長に持つ
- 対話文脈（P1-3）は記録が無ければ空。未確定は持ち越しのみ（P1-4）
"""

import json

import pytest
from fastapi import HTTPException


@pytest.fixture()
def main_module(tmp_path, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_REPO_ROOT", str(tmp_path))
    return main


def test_record_and_get_last_entry_wins(tmp_path, main_module):
    main = main_module
    first = main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_abc123",
            decision="later",
            title="対話室の表示改善",
            source_event_id="event-1",
            rule_decision="today",
            classified_by="user",
        )
    )
    assert first["ok"] is True
    second = main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_abc123",
            decision="today",
            title="対話室の表示改善",
            source_event_id="event-1",
            rule_decision="today",
            classified_by="user",
        )
    )
    assert second["record"]["decision"] == "today"

    result = main.get_improvement_triage()
    assert len(result["records"]) == 1
    assert result["records"][0]["decision"] == "today"
    assert result["records"][0]["source_event_id"] == "event-1"
    assert result["records"][0]["title"] == "対話室の表示改善"
    assert result["counts"] == {"today": 1}

    # 追記形式でファイルに2行残っている（監査可能）
    raw = (tmp_path / "data" / "shion_improvement_triage.jsonl").read_text(encoding="utf-8")
    assert len([line for line in raw.splitlines() if line.strip()]) == 2


def test_invalid_decision_and_empty_key_rejected(main_module):
    main = main_module
    with pytest.raises(HTTPException) as exc:
        main.record_improvement_triage(
            main.ImprovementTriageRequest(canonical_key="misc_x", decision="someday")
        )
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        main.record_improvement_triage(
            main.ImprovementTriageRequest(canonical_key="  ", decision="today")
        )
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        main.record_improvement_triage(
            main.ImprovementTriageRequest(canonical_key="misc_x", decision="today", classified_by="oracle")
        )
    assert exc.value.status_code == 422


def test_triage_context_reflects_records(main_module):
    main = main_module
    main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_ctx1",
            decision="today",
            title="表示ラベルの見直し",
            rule_decision="today",
            classified_by="user",
        )
    )
    main.record_improvement_triage(
        main.ImprovementTriageRequest(
            canonical_key="misc_ctx2",
            decision="discard",
            title="古い重複候補",
            rule_decision="discard",
            classified_by="rule",
        )
    )

    context = main._build_dialogue_triage_context()

    assert "改善トリアージ状況" in context
    assert "今日やる1件" in context
    assert "捨てる1件" in context
    assert "表示ラベルの見直し" in context
    assert "判断主体: rule" in context
    # P1-4: 未確定は持ち越しのみ・自動昇格なし
    assert "自動昇格・自動破棄はしない" in context


def test_triage_context_empty_without_records(main_module):
    assert main_module._build_dialogue_triage_context() == ""


def test_triage_file_corruption_tolerated(tmp_path, main_module):
    main = main_module
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "shion_improvement_triage.jsonl").write_text(
        "壊れた行\n" + json.dumps({"canonical_key": "misc_ok", "decision": "later", "decided_at": "2026-07-17T10:00:00"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = main.get_improvement_triage()

    assert len(result["records"]) == 1
    assert result["records"][0]["canonical_key"] == "misc_ok"
