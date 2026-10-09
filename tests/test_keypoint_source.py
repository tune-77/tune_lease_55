"""REV-544: 会話キーポイントの出所（user / shion / unknown）。"""
from __future__ import annotations

import json
import sqlite3

from api.keypoint_source import NOT_TAUGHT_NOTE, keypoint_content_source, source_suffix

USER_MSG = "トラックのメーカーは主に4つあって、日野トラック、いすゞ、三菱ふそう、UDトラック"
REPLY = "運送業や建設業の倒産増加は収益構造の限界を示唆している。人手に依存しない体制構築を審査基準に加える。"


def test_keypoint_content_source():
    assert keypoint_content_source("トラックのメーカーは日野・いすゞ・三菱ふそう・UDトラックの4つ", USER_MSG, REPLY) == "user"
    assert keypoint_content_source("人手に依存しない体制構築を審査基準に加える", "審査で気をつけることは？", REPLY) == "shion"
    assert keypoint_content_source("料率競争での失注先は再アプローチの対象", USER_MSG, REPLY) == "unknown"


def test_source_suffix():
    assert source_suffix("user") == ""
    assert source_suffix("shion") == "（紫苑の発言）"
    assert source_suffix(None) == "（出所不明）"


def test_save_keypoints_records_source_and_labels_recall(tmp_path):
    from lease_intelligence_mind import (
        build_memory_recall_block,
        load_lease_intelligence_mind,
        mind_directory,
        save_conversation_keypoints,
    )

    save_conversation_keypoints(
        tmp_path,
        "lease-intelligence-dialogue",
        ["人手に依存しない体制構築を審査基準に加える", "トラックのメーカーは日野・いすゞ・三菱ふそう・UDトラックの4つ"],
        "2026-10-09",
        user_message=USER_MSG,
        reply=REPLY,
    )
    kps = load_lease_intelligence_mind(tmp_path)["conversation_keypoints"]
    assert [kp["content_source"] for kp in kps[-2:]] == ["shion", "user"]
    note = (mind_directory(tmp_path) / "Memory" / "2026-10-09.md").read_text(encoding="utf-8")
    assert "- 人手に依存しない体制構築を審査基準に加える（紫苑の発言）" in note
    block = build_memory_recall_block(tmp_path)
    assert "[会話・紫苑の発言] 人手に依存しない体制構築を審査基準に加える" in block
    assert NOT_TAUGHT_NOTE in block


def test_reflection_keypoints_are_shion(tmp_path):
    from lease_intelligence_mind import load_lease_intelligence_mind, save_conversation_keypoints

    save_conversation_keypoints(
        tmp_path, "private_reflection_feedback_loop", ["Private Reflectionからの学び: 材料の鮮度を見る"], "2026-10-09",
        content_source="shion",
    )
    assert load_lease_intelligence_mind(tmp_path)["conversation_keypoints"][-1]["content_source"] == "shion"


def _db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, user_id TEXT, role TEXT, content TEXT, created_at TEXT)")
    conn.executemany("INSERT INTO chat_messages (user_id, role, content, created_at) VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_backfill_script_marks_without_changing_content(tmp_path, monkeypatch):
    from lease_intelligence_mind import _write_state, load_lease_intelligence_mind
    from scripts import mark_keypoint_content_source as script

    vault = tmp_path / "vault"
    vault.mkdir()
    state = load_lease_intelligence_mind(vault)
    state["conversation_keypoints"] = [
        {"date": "2026-10-09", "type": "conversation_keypoint", "content": "人手に依存しない体制構築を審査基準に加える",
         "session_id": "lease-intelligence-dialogue"},
        {"date": "2026-10-09", "type": "conversation_keypoint", "content": "Private Reflectionからの学び: 鮮度",
         "session_id": "private_reflection_feedback_loop"},
        {"date": "2026-01-01", "type": "conversation_keypoint", "content": "対応するやり取りが無い要点",
         "session_id": "lease-intelligence-dialogue"},
        {"date": "2026-10-09", "type": "conversation_keypoint", "content": "既に出所がある", "content_source": "user",
         "session_id": "lease-intelligence-dialogue"},
    ]
    _write_state(vault, state)
    db = tmp_path / "lease_data.db"
    # 2026-10-09 00:30 JST = 2026-10-08 15:30 UTC
    _db(db, [("lease-intelligence-dialogue", "user", "審査で気をつけることは？", "2026-10-08 15:30:00"),
             ("lease-intelligence-dialogue", "assistant", REPLY, "2026-10-08 15:30:00")])
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(script, "get_data_path", lambda *parts: str(tmp_path.joinpath("data", *parts)))

    assert script.main(["--vault", str(vault), "--db", str(db)]) == 0
    assert "content_source" not in load_lease_intelligence_mind(vault)["conversation_keypoints"][0]  # 確認のみ

    assert script.main(["--vault", str(vault), "--db", str(db), "--apply"]) == 0
    kps = load_lease_intelligence_mind(vault)["conversation_keypoints"]
    assert [kp["content_source"] for kp in kps] == ["shion", "shion", "unknown", "user"]
    assert [kp["content"] for kp in kps] == [kp["content"] for kp in state["conversation_keypoints"]]
    backups = list((tmp_path / "data" / "backups" / "keypoint_content_source").glob("mind_*.json"))
    assert len(backups) == 1 and "content_source" not in json.loads(backups[0].read_text())["conversation_keypoints"][0]
