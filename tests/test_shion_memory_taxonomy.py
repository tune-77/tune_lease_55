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


def test_judgment_memory_not_hijacked_by_single_tech_word():
    # 「テスト」1語で判断記憶が技術記憶に化けない（ヒット数比較）
    assert classify_memory_text("境界案件の審査判断はテスト前に承認条件を確認する") == "judgment_memory"


def test_applies_when_ignores_numbers_inside_amounts():
    from api.shion_memory_taxonomy import infer_applies_when

    # 「1400万円」の 40 が境界案件タグを誤発火させない
    assert "境界案件" not in infer_applies_when("リース料は1400万円で契約した")
    assert "境界案件" in infer_applies_when("スコア60前後の境界案件")


def test_markdown_snippets_skip_yaml_frontmatter():
    text = "\n".join(
        [
            "---",
            "type: lease_rule",
            "tags: [条件付き承認, 追加資料, 前受金]",
            "confidence: medium",
            "---",
            "# 条件付き承認",
            "- 条件付き承認は審査部の不安を先回りして解く設計として扱う。",
        ]
    )
    snippets = builder._markdown_snippets(text)
    assert any("条件付き承認は審査部の不安" in s for s in snippets)
    assert not any("tags:" in s or "confidence:" in s for s in snippets)


def test_memory_record_has_stable_metadata():
    record = make_memory_record("否決判断では説明責任を残す", source="test")

    assert record.id.startswith("mem_")
    assert record.memory_type == "value_memory"
    assert record.status == "active"
    assert "否決・警戒判断" in record.applies_when


def test_build_index_carries_over_created_at(tmp_path, monkeypatch):
    """再生成で created_at（初出日）と last_used_at がリセットされない。"""
    repo = tmp_path
    (repo / "data").mkdir()
    (repo / "MEMORY.md").write_text("- 境界案件では条件付き承認を検討する。\n", encoding="utf-8")
    monkeypatch.setattr(builder, "REPO_ROOT", repo)

    first = builder.build_index()
    rid = first["records"][0]["id"]
    first["records"][0]["created_at"] = "2026-01-01"
    first["records"][0]["last_used_at"] = "2026-06-01"
    previous_path = repo / "data" / "shion_memory_index.json"
    previous_path.write_text(json.dumps(first, ensure_ascii=False), encoding="utf-8")

    second = builder.build_index(previous_index_path=previous_path)

    record = next(r for r in second["records"] if r["id"] == rid)
    assert record["created_at"] == "2026-01-01"
    assert record["last_used_at"] == "2026-06-01"


def test_build_index_demo_safe_excludes_dialogue_and_private(tmp_path, monkeypatch):
    """--demo-safe では対話・内省・private の記憶が公開バンドルに載らない。"""
    repo = tmp_path
    (repo / "data").mkdir()
    (repo / "MEMORY.md").write_text(
        "\n".join(
            [
                "- 境界案件では条件付き承認を検討する。",
                "- ユーザーの好みは機能追加より知識基盤の優先。",
                "- Private Reflection: 同じ文型への違和感を覚えた。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "REPO_ROOT", repo)

    full = builder.build_index()
    safe = builder.build_index(demo_safe=True)

    full_types = {r["memory_type"] for r in full["records"]}
    assert "dialogue_memory" in full_types or "reflection_memory" in full_types
    for record in safe["records"]:
        assert record["memory_type"] not in {"dialogue_memory", "reflection_memory"}
        assert record["status"] != "private"
        assert not record.get("private")


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
