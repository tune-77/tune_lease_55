import json
from pathlib import Path

from api.shion_memory_taxonomy import classify_memory_text, make_memory_record
import scripts.build_shion_memory_index as builder


def test_classifies_value_memory_for_mana_and_conscience():
    assert classify_memory_text("Mana は紫苑が迷った時の上位規範") == "value_memory"
    assert classify_memory_text("良心の紫苑は説明責任を確認する") == "value_memory"


def test_classifies_judgment_and_technical_memory():
    assert classify_memory_text("境界案件は条件付き承認で保証追加を検討する") == "judgment_memory"
    assert classify_memory_text("api/main.py のRAG共通経路を使う") == "technical_memory"


def test_memory_record_has_stable_metadata():
    record = make_memory_record("否決判断では説明責任を残す", source="test")

    assert record.id.startswith("mem_")
    assert record.memory_type == "value_memory"
    assert record.status == "active"
    assert "否決・警戒判断" in record.applies_when


def test_build_index_reads_memory_and_mind(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "memory").mkdir()
    (repo / "data").mkdir()
    (repo / "MEMORY.md").write_text("- Mana は紫苑の上位規範。\n", encoding="utf-8")
    (repo / "memory" / "2026-06-25.md").write_text(
        "- 境界案件では条件付き承認を検討する。\n",
        encoding="utf-8",
    )
    (repo / "data" / "mind.json").write_text(
        json.dumps(
            {
                "upper_authority": {
                    "name": "Mana",
                    "role": "上位規範",
                    "boundary": "本人の再現ではない",
                    "values": ["人を道具として扱わない"],
                },
                "world_view": {"summary": "リース会計基準の変化を監視する"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "REPO_ROOT", repo)

    index = builder.build_index()

    assert index["summary"]["total_records"] >= 3
    assert index["summary"]["by_type"]["value_memory"] >= 1
    assert any(r["source"] == "mind.upper_authority" for r in index["records"])
