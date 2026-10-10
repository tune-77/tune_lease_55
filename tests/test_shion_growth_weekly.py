import json
import plistlib
from datetime import date, datetime
from pathlib import Path

from scripts import shion_growth_weekly as growth

TOC = """# 目次

| ノート | 中身 |
|---|---|
| [[13_答えが固まる原因調査]] | 調査 |

## 関連
"""


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "data"
    source.mkdir()
    _jsonl(source / "shion_prediction_log.jsonl", [
        {"at": "2026-10-05T10:00:00", "affect_hit": True, "reaction_hit": True, "topic_hit": False, "surprise": 0.2, "actual_affect": "通常"},
        {"at": "2026-10-06T10:00:00", "affect_hit": False, "reaction_hit": True, "topic_hit": True, "surprise": 0.6, "actual_affect": "疲れ"},
    ])
    _jsonl(source / "shion_emotion_grounding_log.jsonl", [
        {"ts": "2026-10-06T11:00:00", "kind": "serious_emotion", "status": "applied",
         "counts": {"verified": 3, "unsupported_fact": 1, "contradicted": 0}},
    ])
    _jsonl(source / "cloudrun_chat_log.jsonl", [
        {"ts": "2026-10-05T01:00:00+00:00", "user_id": "u", "user_message": "運送業の倒産で気をつけることは？", "assistant_reply": "資金繰りを見ます。どう思いますか？"},
        {"ts": "2026-10-05T01:10:00+00:00", "user_id": "u", "user_message": "そうだね", "assistant_reply": "了解です。"},
        {"ts": "2026-10-05T01:20:00+00:00", "user_id": "u", "user_message": "運送業の倒産で気をつけることは？", "assistant_reply": "資金繰りを見ます。どう思いますか？"},
    ])
    _jsonl(source / "chat_prompt_budget_log.jsonl", [
        {"ts": "2026-10-05T10:00:00+09:00", "surface": "dialogue",
         "blocks": {"news_zettel_context": {"kept": 300}, "emotion_grounding_context": {"kept": 500}}},
    ])
    _jsonl(source / "ai_usage.jsonl", [
        {"timestamp": "2026-10-05T01:00:00+00:00", "provider": "google", "feature": "chat_memory",
         "model": "gemini-3.1-flash-lite", "input_tokens": 10000, "output_tokens": 500, "total_tokens": 10500, "call_class": "memory"},
        {"timestamp": "2026-10-05T02:00:00+00:00", "provider": "google", "feature": "recorded_reply_verify",
         "model": "gemini-3.1-flash-lite", "input_tokens": 10000, "output_tokens": 500, "total_tokens": 10500, "call_class": "verification"},
    ])
    _jsonl(source / "judgment_asset_growth_history.jsonl", [
        {"date": "2026-09-30", "counts": {"active_rules": 5}},
        {"date": "2026-10-06", "counts": {"active_rules": 7}},
    ])
    (source / "shion_relationship_state.json").write_text(json.dumps({"score": 10.0, "understanding": 0.2}), encoding="utf-8")
    (source / "user_affect_state.json").write_text(json.dumps({"users": {"u": {"observations": [{"label": "疲れ", "at": "2026-10-06T10:00:00"}]}}}), encoding="utf-8")
    (source / "canonical_judgment_rules.json").write_text(json.dumps({"rules": [
        {"id": "a", "status": "active", "created_at": "2026-10-05T00:00:00", "knowledge_kind": "insight"},
        {"id": "b", "status": "active", "created_at": "2026-09-01T00:00:00"},
        {"id": "c", "status": "excluded_shion_reply", "content_source": "shion"},
    ]}), encoding="utf-8")
    (source / "policy_likeness_queue.json").write_text(json.dumps({"items": {"x": {"judgment_id": "b", "tier": "guideline"}}}), encoding="utf-8")
    (source / "news_zettel_state.json").write_text(json.dumps({
        "m1": {"status": "written", "hubs": ["h"], "processed_at": "2026-10-05T06:00:00"},
        "m2": {"status": "written", "hubs": [], "processed_at": "2026-10-06T06:00:00"},
        "m3": {"status": "skipped", "processed_at": "2026-10-06T06:00:00"},
    }), encoding="utf-8")

    vault = tmp_path / "vault"
    li = vault / growth.REFLECTION_REL
    li.mkdir(parents=True)
    (li / "2026-10-04.md").write_text("---\ndate: x\n---\n# 内省\n- 今日の観察: ユーザーは審査の話をしていた\n- 仮説: 料率負けが多い\n", encoding="utf-8")
    (li / "2026-10-05.md").write_text("---\ndate: x\n---\n# 内省\n- 今日の観察: ユーザーは審査の話をしていた\n- 仮説: 疲れている時は短く答える\n", encoding="utf-8")
    (vault / growth.MIND_REL).write_text(json.dumps({
        "mood": {"curiosity": 60, "loneliness": 0},
        "dialogue_mood": {"curiosity": 15},
        "mood_change_log": [{"ts": "2026-10-05T10:00:00", "changes": [{"axis": "curiosity", "before": 58, "after": 60}]}],
        "long_term_memories": [{"date": "2026-10-05", "type": "emotion_snapshot", "content": "好奇心=60, 警戒=70"}],
    }, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / growth.TOC_NAME).write_text(TOC, encoding="utf-8")
    return source, vault, out


def test_collect_counts_each_metric_from_logs_only(tmp_path):
    source, vault, _ = _fixture(tmp_path)
    m = growth.collect(source, vault, date(2026, 10, 4), date(2026, 10, 10))

    assert m["prediction"]["n"] == 2 and m["prediction"]["affect_hit"] == 50.0
    assert m["grounding"]["grounded_rate"] == 75.0 and m["grounding"]["unsupported_fact"] == 1
    assert m["grounding"]["evidence_injected"] == 1
    assert m["repeat"]["repeats"] == 1 and m["repeat"]["near_copies"] == 1
    assert m["repeat"]["guard_instructions_est"] == 0  # REV-544 マージ前
    assert m["reflection"]["daily"] == {"2026-10-05": 50.0}  # 2行中、仮説の行だけが新しい
    assert m["curiosity"]["questions"] == 2 and m["curiosity"]["answered_rate"] == 50.0
    assert m["mood"]["relationship_pinned"] is True
    assert "孤独" in m["mood"]["pinned"] and "好奇心（対話の揺れ）" in m["mood"]["pinned"]
    assert m["mood"]["snapshots"] == {"2026-10-05": {"好奇心": 60, "警戒": 70}}
    assert m["judgment"]["kinds"] == {"知見": 1, "目安": 1} and m["judgment"]["excluded_shion"] == 1
    assert m["judgment"]["active_as_of_end"] == 7 and m["judgment"]["new_in_period"] == 1
    assert m["news"]["total_written"] == 2 and m["news"]["hub_rate_total"] == 50.0 and m["news"]["attached"] == 1
    assert m["cost"]["yen"] > 0
    assert set(m["cost"]["by_class"]) == {"memory", "verification"}
    assert m["cost"]["by_class"]["verification"] > 0
    assert m["cost"]["yen"] > m["cost"]["by_class"]["memory"]


def test_week_buckets_split_on_sunday():
    assert growth.week_buckets(date(2026, 10, 1), date(2026, 10, 9)) == [
        (date(2026, 10, 1), date(2026, 10, 3)), (date(2026, 10, 4), date(2026, 10, 9)),
    ]


def test_asks_question_looks_at_last_sentences():
    assert growth.asks_question("なるほど。どの業種が気になりますか？")
    assert not growth.asks_question("なぜでしょう？と思いました。結論は資金繰りです。続けます。以上です。")


def test_backfill_then_weekly_writes_notes_history_and_links_toc_once(tmp_path, monkeypatch):
    source, vault, out = _fixture(tmp_path)
    history = tmp_path / "hist" / "history.jsonl"
    common = ["--source-dir", str(source), "--vault", str(vault), "--out-dir", str(out), "--history", str(history)]

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 11, 8, 30, tzinfo=growth.JST)

    monkeypatch.setattr(growth, "datetime", _Now)
    assert growth.main(["--from", "2026-10-01", "--to", "2026-10-09", *common]) == 0
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert [(r["start"], r["end"]) for r in rows] == [("2026-10-01", "2026-10-03"), ("2026-10-04", "2026-10-09")]
    assert rows[0]["relationship_score"] is None  # 現在値しか無い指標は過去の週に入れない
    assert rows[0]["judgment_active"] == 5

    assert growth.main(common) == 0  # 日曜の定期実行: 2026-10-04〜10-10
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert [(r["start"], r["end"]) for r in rows] == [("2026-10-01", "2026-10-03"), ("2026-10-04", "2026-10-10")]
    assert rows[-1]["relationship_score"] == 10.0 and rows[-1]["partial"] is False

    note = (out / growth.NOTE_DIR / "週次_2026-10-04〜2026-10-10.md").read_text(encoding="utf-8")
    assert "xychart-beta" in note and "上限に張り付き" in note and "rag_exclude: true" in note
    assert (out / growth.NOTE_DIR / "遡り集計_2026-10-01〜2026-10-09.md").exists()
    index = (out / f"{growth.INDEX_NAME}.md").read_text(encoding="utf-8")
    assert "週次_2026-10-04〜2026-10-10" in index and "遡り集計_2026-10-01〜2026-10-09" in index
    toc = (out / growth.TOC_NAME).read_text(encoding="utf-8")
    assert toc.count(f"[[{growth.INDEX_NAME}]]") == 1
    assert toc.index("[[13_") < toc.index(f"[[{growth.INDEX_NAME}]]") < toc.index("## 関連")


def test_dry_run_writes_nothing(tmp_path, capsys):
    source, vault, out = _fixture(tmp_path)
    history = tmp_path / "history.jsonl"
    assert growth.main(["--from", "2026-10-04", "--to", "2026-10-06", "--source-dir", str(source), "--vault", str(vault),
                        "--out-dir", str(out), "--history", str(history), "--dry-run"]) == 0
    assert "紫苑の成長記録" in capsys.readouterr().out
    assert not history.exists() and not (out / growth.NOTE_DIR).exists()


def test_launchd_job_runs_sunday_morning_with_vault_env():
    plist = plistlib.loads(Path("launchd/com.tunelease.shion-growth-weekly.plist").read_bytes())
    assert plist["StartCalendarInterval"] == {"Weekday": 0, "Hour": 8, "Minute": 30}
    assert plist["ProgramArguments"][-1].endswith("scripts/shion_growth_weekly.py")
    env = plist["EnvironmentVariables"]
    assert env["OBSIDIAN_VAULT_PATH"] == env["OBSIDIAN_VAULT"]
