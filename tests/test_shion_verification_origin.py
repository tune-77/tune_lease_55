"""REV-591: 検証・テストの会話に印を付け、紫苑の記憶・内省の材料から外す。"""

from __future__ import annotations

import contextvars
import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import shion_verification_origin as origin
from shion_verification_origin import (
    VERIFICATION_HEADER,
    VERIFICATION_NOTE_MARK,
    VerificationOriginMiddleware,
    history_user_id,
    is_verification_row,
    is_verification_turn,
    is_verification_user_id,
    mark_verification_turn,
    origin_fields,
    verification_client_headers,
)


@pytest.fixture(autouse=True)
def _no_verify_env(monkeypatch):
    monkeypatch.delenv("AI_LIVE_VERIFY", raising=False)
    monkeypatch.delenv("SHION_VERIFICATION", raising=False)


def _in_verification_turn(fn, *args, **kwargs):
    """印の付いた会話の文脈で fn を呼ぶ（テストの外に印を漏らさない）。"""

    def run():
        mark_verification_turn(header_value="1")
        return fn(*args, **kwargs)

    return contextvars.copy_context().run(run)


def test_verification_user_ids_and_rows():
    for uid in ("rev544_verify", "verify_rev532", "rev540_verify", "x:verification", "default:verification"):
        assert is_verification_user_id(uid), uid
    for uid in ("default", "lease-intelligence-dialogue", "verifyx", "screening-shion-review:1"):
        assert not is_verification_user_id(uid), uid
    assert is_verification_row({"user_id": "default", "origin": "verification"})
    assert is_verification_row({"user_id": "default", "metadata": {"origin": "verification"}})
    assert is_verification_row({"user_id": "rev540_verify"})
    assert not is_verification_row({"user_id": "default", "metadata": {}})


def test_mark_and_fields_stay_inside_the_turn():
    assert not is_verification_turn() and origin_fields() == {}
    assert _in_verification_turn(origin_fields) == {"origin": "verification"}
    assert _in_verification_turn(history_user_id, "lease-intelligence-dialogue") == "lease-intelligence-dialogue:verification"
    assert history_user_id("lease-intelligence-dialogue") == "lease-intelligence-dialogue"
    assert not is_verification_turn()
    assert contextvars.copy_context().run(mark_verification_turn, user_id="rev591_verify")
    assert not contextvars.copy_context().run(mark_verification_turn, user_id="default")


def test_client_headers_follow_live_verify_env():
    assert verification_client_headers({}) == {}
    assert verification_client_headers({"AI_LIVE_VERIFY": "1"}) == {VERIFICATION_HEADER: "1"}
    assert verification_client_headers({"SHION_VERIFICATION": "true"}) == {VERIFICATION_HEADER: "1"}


def _probe_app() -> TestClient:
    from api.background_executor import background_executor

    app = FastAPI()
    app.add_middleware(VerificationOriginMiddleware)

    @app.get("/probe")
    def probe():  # 同期エンドポイント（スレッドプール）と背景処理の両方で印が見えるか
        return {"turn": is_verification_turn(), "background": background_executor.submit(is_verification_turn).result()}

    return TestClient(app)


def test_middleware_marks_request_and_background_work(monkeypatch):
    client = _probe_app()
    assert client.get("/probe", headers={VERIFICATION_HEADER: "1"}).json() == {"turn": True, "background": True}
    assert client.get("/probe").json() == {"turn": False, "background": False}
    monkeypatch.setenv("AI_LIVE_VERIFY", "1")  # 検証用に起動したサーバーは全部の会話が検証扱い
    assert client.get("/probe").json() == {"turn": True, "background": True}


def test_local_chat_log_row_carries_origin(tmp_path, monkeypatch):
    import api.chat_side_effects as side_effects

    monkeypatch.setattr(side_effects, "get_data_path", lambda name: str(tmp_path / name))
    kwargs = dict(
        surface="lease_intelligence_dialogue",
        user_id="lease-intelligence-dialogue",
        category="dialogue",
        response_mode="shion",
        user_message="今日ちょっと疲れた",
        assistant_reply="お疲れさま",
        metadata={},
    )
    side_effects.append_local_cloudrun_chat_log(**kwargs)
    _in_verification_turn(side_effects.append_local_cloudrun_chat_log, **kwargs)
    rows = [json.loads(line) for line in (tmp_path / "cloudrun_chat_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "origin" not in rows[0]
    assert rows[1]["origin"] == "verification"
    assert not _in_verification_turn(side_effects.should_auto_save_chat, improvement_mode=False)
    assert side_effects.should_auto_save_chat(improvement_mode=False)


def test_memory_writers_skip_verification_turns(tmp_path, monkeypatch):
    from api import shion_mutual_prediction, user_personal_memory
    from api.chat_judgment_asset_capture import capture_chat_judgment_asset_if_needed
    from lease_intelligence_pending import extract_and_save_promises, save_countermeasures_to_dispatch

    assert shion_mutual_prediction.mutual_prediction_enabled()
    assert not _in_verification_turn(shion_mutual_prediction.mutual_prediction_enabled)
    assert _in_verification_turn(user_personal_memory.capture_user_personal_memory, "私は猫が好きです")["reason"] == "verification"
    result = _in_verification_turn(
        capture_chat_judgment_asset_if_needed,
        "判断資産に入れて: 同じ質問が3回続いたら心理状態を確認する",
        user_id="lease-intelligence-dialogue",
        surface="lease_intelligence_dialogue",
        candidates_loader=lambda **_: pytest.fail("検証の会話で候補を読まない"),
        candidate_creator=lambda _req: pytest.fail("検証の会話で候補を作らない"),
        request_factory=dict,
        cloudrun_event_recorder=lambda **_: {},
    )
    assert result == {"captured": False, "reason": "verification"}
    assert _in_verification_turn(extract_and_save_promises, "運送業は？", "後で調べておきます。") == []
    assert _in_verification_turn(save_countermeasures_to_dispatch, "q", "③対応策\n- 何かする") == 0


def test_affect_memory_records_nothing_and_recall_skips_marked(tmp_path):
    from api.user_affect_memory import recall_user_affect, remember_and_build_block

    path = tmp_path / "user_affect_state.json"
    _in_verification_turn(remember_and_build_block, "u", "疲れ", 0.6, surface="lease_intelligence_dialogue", path=path)
    assert not path.exists() or "疲れ" not in path.read_text(encoding="utf-8")
    now = datetime(2026, 10, 10, 9, 0, 0)
    path.write_text(
        json.dumps(
            {
                "users": {
                    "u": {
                        "observations": [
                            {"label": "疲れ", "intensity": 0.6, "at": "2026-10-09T08:59:46", "surface": "x", "origin": "verification"}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert recall_user_affect("u", now=now, path=path).to_payload().get("recurrent", "") != "疲れ"
    assert not recall_user_affect("u", now=now, path=path).to_payload().get("count")


def test_mind_is_untouched_by_verification_turns(tmp_path):
    from lease_intelligence_mind import (
        MIND_FILE_NAME,
        load_lease_intelligence_mind,
        mind_directory,
        record_dialogue_memory,
        record_lease_knowledge,
        register_dialogue_event,
        save_conversation_keypoints,
    )

    load_lease_intelligence_mind(tmp_path)
    mind_file = mind_directory(tmp_path) / MIND_FILE_NAME
    before = mind_file.read_text(encoding="utf-8") if mind_file.exists() else ""
    _in_verification_turn(register_dialogue_event, tmp_path, "お疲れさま、少し休憩しよう", "お疲れさま")
    _in_verification_turn(record_dialogue_memory, tmp_path, "お疲れさま、少し休憩しよう", "お疲れさま")
    _in_verification_turn(save_conversation_keypoints, tmp_path, "s", ["運送・建設業は価格転嫁の可否を最重要視する"], "2026-10-09")
    written = _in_verification_turn(record_lease_knowledge, tmp_path, "運送業", "価格転嫁を見る", "2026-10-09")
    assert written["path"] == "" and written["skipped"] == "verification"
    assert (mind_file.read_text(encoding="utf-8") if mind_file.exists() else "") == before


def test_reflection_material_skips_verification(tmp_path, monkeypatch):
    import lease_intelligence_reflection as reflection

    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {"ts": "2026-10-08T23:59:47+00:00", "user_id": "lease-intelligence-dialogue", "user_message": "お疲れさま、少し休憩しよう", "assistant_reply": "x", "origin": "verification"},
        {"ts": "2026-10-08T23:36:46+00:00", "user_id": "rev540_verify", "user_message": "運送業や建設業の倒産が増えてるけど", "assistant_reply": "x"},
        {"ts": "2026-10-09T08:10:11+00:00", "user_id": "lease-intelligence-dialogue", "user_message": "フォークリフトの見積書作成している", "assistant_reply": "x"},
    ]
    (data / "cloudrun_chat_log.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    material = reflection._load_cloudrun_chat_jsonl("2026-10-09")
    assert "フォークリフト" in material
    assert "休憩" not in material and "運送業" not in material

    note = (
        "# リース知性体との対話 — 2026-10-09\n\n"
        f"## 08:59:47 {VERIFICATION_NOTE_MARK}\n\n**ユーザー**\n\nお疲れさま、少し休憩しよう\n\n**リース知性体**\n\nお疲れさま\n\n"
        "## 17:10:11\n\n**ユーザー**\n\nフォークリフトの見積書作成している\n\n**リース知性体**\n\nなるほど\n"
    )
    compact = reflection._compact_dialogue_note(note)
    assert "フォークリフト" in compact and "休憩" not in compact


def test_recent_reflection_uses_correction(tmp_path):
    import lease_intelligence_reflection as reflection

    rdir = reflection._reflection_dir(tmp_path)
    rdir.mkdir(parents=True)
    (rdir / "2026-10-10.md").write_text(
        "---\ndate: 2026-10-10\nmaterial_origin: verification\n---\n# 非公開の内省\n\n"
        "## 訂正（検証の会話）\n\n同じ質問の繰り返しは検証テストで、ユーザーの実際の発言ではない\n\n"
        "## 今日の対話について\n\nユーザーは思考停止していた\n",
        encoding="utf-8",
    )
    recent = reflection._load_recent_reflections(tmp_path, base_date=datetime(2026, 10, 11).date())
    assert "検証テスト" in recent and "思考停止" not in recent
    assert "検証テスト" in reflection._load_reflection_section(tmp_path, "2026-10-10")


def test_growth_weekly_ignores_verification_rows(tmp_path, monkeypatch):
    from scripts import shion_growth_weekly as growth

    rows = [
        {"ts": "2026-10-08T23:21:47+00:00", "user_id": "lease-intelligence-dialogue", "user_message": "運送業や建設業の倒産が増えてるけど？", "assistant_reply": "a", "origin": "verification"},
        {"ts": "2026-10-08T23:35:51+00:00", "user_id": "lease-intelligence-dialogue", "user_message": "運送業や建設業の倒産が増えてるけど？", "assistant_reply": "a", "origin": "verification"},
    ]
    (tmp_path / "cloudrun_chat_log.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    start, end = datetime(2026, 10, 5).date(), datetime(2026, 10, 11).date()
    result = growth.collect(tmp_path, tmp_path, start, end)
    assert result["repeat"]["repeats"] == 0
    assert growth.prediction_metrics([{"at": "2026-10-09T08:59:46", "origin": "verification", "affect_hit": False}], start, end)["n"] == 0


def test_module_has_single_header_name():
    assert origin.VERIFICATION_HEADER == "X-Shion-Verification"
    assert Path(origin.__file__).name == "shion_verification_origin.py"


def test_mark_script_marks_copies_without_deleting(tmp_path, monkeypatch):
    import importlib.util
    import sqlite3

    spec = importlib.util.spec_from_file_location(
        "mark_rev591", Path(__file__).resolve().parents[1] / "scripts" / "mark_verification_conversations_20261009.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    data, vault = tmp_path / "data", tmp_path / "vault"
    data.mkdir()
    li = vault / "Projects" / "tune_lease_55" / "Lease Intelligence"
    (li / "Dialogue").mkdir(parents=True)
    (li / "Private Reflection").mkdir()
    chat = [
        {"event_id": "a", "ts": "2026-10-08T23:59:47+00:00", "user_id": "lease-intelligence-dialogue", "user_message": "お疲れさま、少し休憩しよう"},
        {"event_id": "b", "ts": "2026-10-09T08:40:49+00:00", "user_id": "lease-intelligence-dialogue", "user_message": "よくわかったね なぜ？疲れてるとわかった？"},
    ]
    (data / "cloudrun_chat_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chat), encoding="utf-8")
    conn = sqlite3.connect(data / "lease_data.db")
    conn.execute("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, user_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP)")
    conn.executemany(
        "INSERT INTO chat_messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        [
            ("lease-intelligence-dialogue", "user", "お疲れさま、少し休憩しよう", "2026-10-08 23:59:47"),
            ("lease-intelligence-dialogue", "user", "おはよう。", "2026-10-08 22:07:01"),  # 音声の文字起こし（本人）
        ],
    )
    conn.commit()
    conn.close()
    (li / "Dialogue" / "2026-10-09.md").write_text(
        "# 対話\n\n## 07:19:27\n\n**ユーザー**\n\n判断資産に入れて\n\n## 08:59:47\n\n**ユーザー**\n\nお疲れさま、少し休憩しよう\n",
        encoding="utf-8",
    )
    reflection = li / "Private Reflection" / "2026-10-10.md"
    reflection.write_text("---\ndate: 2026-10-10\n---\n# 非公開の内省\n\n## 今日の対話について\n\nユーザーは思考停止\n", encoding="utf-8")

    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.delenv("DB_PATH", raising=False)
    monkeypatch.setattr(script, "resolve_obsidian_vault", lambda: vault)
    marker = script.run(apply=True)

    rows = [json.loads(line) for line in (data / "cloudrun_chat_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["origin"] == "verification" and "origin" not in rows[1]
    conn = sqlite3.connect(data / "lease_data.db")
    assert [r[0] for r in conn.execute("SELECT user_id FROM chat_messages ORDER BY id")] == [
        "lease-intelligence-dialogue:verification",
        "lease-intelligence-dialogue",
    ]
    conn.close()
    note = (li / "Dialogue" / "2026-10-09.md").read_text(encoding="utf-8")
    assert f"## 08:59:47 {VERIFICATION_NOTE_MARK}" in note and "## 07:19:27\n" in note and "休憩しよう" in note
    text = reflection.read_text(encoding="utf-8")
    assert "## 訂正（検証の会話）" in text and "ユーザーは思考停止" in text and "material_origin: verification" in text
    assert (marker.backup_dir / "cloudrun_chat_log.jsonl").exists() and (marker.backup_dir / "lease_data.db").exists()
    again = script.run(apply=True)  # 二度目は何も変えない
    assert again.inventory == []
