from __future__ import annotations

import datetime as dt
import json
import sqlite3

from api import shion_illustration_gallery as gallery
from scripts import shion_weekly_illustration as weekly


def _db(tmp_path, rows):
    path = tmp_path / "chat.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, user_id TEXT, role TEXT, content TEXT, created_at TEXT)")
    for user_id, role, content, age_days in rows:
        conn.execute(
            "INSERT INTO chat_messages (user_id, role, content, created_at) VALUES (?, ?, ?, datetime('now', ?))",
            (user_id, role, content, f"-{age_days} days"),
        )
    conn.commit()
    conn.close()
    return path


def test_collect_snippets_drops_business_old_and_masks_numbers(tmp_path):
    db = _db(tmp_path, [
        ("lease-intelligence-dialogue", "user", "今日はタムと川沿いを散歩したよ", 1),
        ("lease-intelligence-dialogue", "user", "株式会社ABCの審査スコアどう？", 1),
        ("shion-default", "user", "餃子を12個食べた", 2),
        ("lease-intelligence-dialogue", "assistant", "紫苑の返答は使わない", 1),
        ("default", "user", "先月の話は古い", 10),
        ("compare-shion", "user", "対象外のユーザー", 1),
    ])
    snippets = weekly.collect_snippets(db)
    assert any("散歩" in s for s in snippets)
    assert any("餃子" in s and "12" not in s for s in snippets)
    assert not any("審査" in s or "ABC" in s or "古い" in s or "対象外" in s or "返答" in s for s in snippets)


def test_validate_topic_rejects_business_numbers_and_missing_keys():
    ok = {"topic": "タムとの散歩", "caption": "川沿いでひと休み", "scene": "riverside", "composition": "wide", "expression": "smile"}
    assert weekly.validate_topic(ok)["caption"] == "川沿いでひと休み"
    assert weekly.validate_topic({**ok, "caption": "審査おつかれ"}) is None
    assert weekly.validate_topic({**ok, "topic": "餃子12個"}) is None
    assert weekly.validate_topic({**ok, "scene": ""}) is None


def test_prompts_avoid_desk_and_keep_history():
    history = [{"topic": "ラーメン", "composition": "close-up"}]
    prompt = weekly.topic_prompt(["散歩した"], history)
    assert "ラーメン" in prompt and "机の前で唸る" in prompt and "散歩した" in prompt
    image = weekly.image_prompt({"scene": "s", "composition": "c", "expression": "e", "topic": "t", "caption": "x"})
    assert "NOT at an office desk" in image and "Exactly ONE instance" in image


def test_run_saves_image_caption_history_and_skips_when_exists(tmp_path):
    db = _db(tmp_path, [("lease-intelligence-dialogue", "user", "ラーメン屋で大盛りを頼んだ", 1)])
    out = tmp_path / "gallery"
    history = tmp_path / "history.jsonl"
    prompts = []

    def topic_fn(prompt):
        prompts.append(prompt)
        return {"topic": "ラーメン", "caption": "大盛りに挑戦", "scene": "ramen shop", "composition": "low angle", "expression": "excited"}

    def image_fn(prompt, target):
        target.write_bytes(b"RIFFxxxxWEBP")
        return True

    day = dt.date(2026, 10, 11)
    result = weekly.run(db_path=db, day=day, gallery_dir=out, history_path=history, topic_fn=topic_fn, image_fn=image_fn, archive=False)
    assert result["status"] == "ok" and (out / "2026-10-11.webp").exists()
    assert gallery.payload("2026-10-11.webp", out)["caption"] == "大盛りに挑戦"
    assert json.loads(history.read_text(encoding="utf-8"))["topic"] == "ラーメン"
    again = weekly.run(db_path=db, day=day, gallery_dir=out, history_path=history, topic_fn=topic_fn, image_fn=image_fn, archive=False)
    assert again["reason"] == "already_exists" and len(prompts) == 1


def test_run_skips_without_usable_chat_or_with_rejected_topic(tmp_path):
    empty = _db(tmp_path, [("default", "user", "案件の財務を見て", 1)])
    called = []
    result = weekly.run(db_path=empty, day=dt.date(2026, 10, 11), gallery_dir=tmp_path / "g", history_path=tmp_path / "h",
                        topic_fn=lambda p: called.append(p) or {}, image_fn=lambda p, t: True, archive=False)
    assert result["reason"] == "no_usable_chat" and not called
