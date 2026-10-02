from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts import improvement_request_dedup_shadow as shadow


def _row(event_id, title, *, surface="shion_self_proposal", change="", ts="2026-10-01T00:00:00+00:00", body=""):
    return {"event_id": event_id, "ts": ts, "title": title, "surface": surface, "source": "usage_loop", "proposed_change": change, "body": body}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_request_items_skip_status_records_and_read_chat_issue():
    rows = [
        _row("s1", "改善ログ: approved"),
        _row("c1", "チャット改善メモ", surface="chat_improvement", body="## AI整理\n- 課題: 案件単位が千円のまま\n- 改善案: 百万円にする\n"),
    ]
    items = shadow.request_items(rows)
    assert [(i["id"], i["title"], i["detail"]) for i in items] == [("c1", "案件単位が千円のまま", "百万円にする")]


def test_run_keeps_rule_duplicates_and_asks_jev_only_for_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", str(tmp_path / "jev.jsonl"))
    log = tmp_path / "log.jsonl"
    _write(
        log,
        [
            _row("old1", "FAQの導線と内容改善", change="フッターにFAQリンク", ts="2026-09-01T00:00:00+00:00"),
            _row("old2", "未利用画面のメニュー整理", change="ナビから削除", ts="2026-09-01T00:00:00+00:00"),
            _row("new1", "FAQの発見性と内容改善", change="ナビにFAQリンク"),
            _row("new2", "未利用画面のメニュー整理", change="ナビから削除"),  # 同題 = 既存ルールで重複
        ],
    )
    asked = []

    def jev(pairs):
        asked.extend(pairs)
        return [0.8 if "FAQ" in a and "FAQ" in b else 0.1 for a, b in pairs], "jev-test"

    report = shadow.run(
        log_path=log,
        state_path=tmp_path / "state.json",
        latest_path=tmp_path / "latest.json",
        jev_fn=jev,
        embed_fn=lambda texts: None,
        now=datetime(2026, 10, 2, tzinfo=timezone.utc),
    )

    assert report["status"] == "applied" and report["new_requests"] == 2
    assert report["rule_duplicates"] >= 1
    assert all("未利用画面のメニュー整理" not in a or "未利用画面のメニュー整理" not in b for a, b in asked)
    assert [(c["new_id"], c["existing_id"]) for c in report["candidates"]] == [("new1", "old1")]
    assert json.loads((tmp_path / "state.json").read_text())["seen"] == ["new1", "new2", "old1", "old2"]

    # 2回目は新着なし → Jev を呼ばない
    asked.clear()
    again = shadow.run(log_path=log, state_path=tmp_path / "state.json", latest_path=tmp_path / "latest.json", jev_fn=jev, embed_fn=lambda t: None)
    assert again["new_requests"] == 0 and asked == []


def test_run_falls_back_to_rules_when_jev_fails(tmp_path):
    log = tmp_path / "log.jsonl"
    _write(log, [_row("a", "FAQの導線と内容改善"), _row("b", "FAQの発見性と内容改善")])

    def broken(pairs):
        raise TimeoutError

    report = shadow.run(log_path=log, state_path=tmp_path / "state.json", latest_path=tmp_path / "latest.json", jev_fn=broken, embed_fn=lambda t: None, now=datetime(2026, 10, 2, tzinfo=timezone.utc))

    assert report["status"] == "fallback" and report["candidates"] == []
    assert not (tmp_path / "state.json").exists()  # 次回もう一度聞く
