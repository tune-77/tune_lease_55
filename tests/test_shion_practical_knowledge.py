import json

from api.shion_practical_knowledge import (
    classify_practical_layer,
    extract_practical_entries_from_memory_records,
    infer_practical_scene,
)


def test_classify_practical_layers():
    assert classify_practical_layer("OCR値を売上、利益、純資産へ分けて確認する。") == "procedure_layer"
    assert classify_practical_layer("外部情報は案件判断を一般論から外すために使う。") == "meaning_layer"
    assert classify_practical_layer("返済原資が弱い場合は条件付き承認に逃げない。") == "judgment_layer"


def test_extract_practical_entries_from_memory_records():
    learned = extract_practical_entries_from_memory_records(
        [
            {
                "id": "m1",
                "content": "境界案件では承認条件、追加確認、否決理由を同時に作る。",
                "memory_type": "judgment_memory",
                "status": "active",
                "confidence": 0.8,
                "source": "test",
                "source_path": "memory/test.md",
            },
            {
                "id": "m2",
                "content": "外部調査は業界ニュースを当該顧客の資金繰りへ接続するために使う。",
                "memory_type": "judgment_memory",
                "status": "active",
                "confidence": 0.8,
                "source": "test",
                "source_path": "memory/test.md",
            },
        ]
    )

    scenes = {scene["id"]: scene for scene in learned["scenes"]}
    assert "borderline_decision" in scenes
    assert "external_research" in scenes


def test_infer_practical_scene_merges_learned_map(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "id": "borderline_decision",
                        "label": "承認・否決の境界",
                        "triggers": ["境界", "微妙", "承認"],
                        "procedure_layer": [
                            {
                                "text": "過去のレビュー済み案件から、銀行支援と物件保全を先に確認する。",
                                "source_path": "judgment_feedback:1",
                            }
                        ],
                        "meaning_layer": [],
                        "judgment_layer": [
                            {
                                "text": "弱点が限定的な場合だけ条件付き承認に寄せる。",
                                "source_path": "judgment_feedback:1",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    scene = infer_practical_scene("この承認条件は微妙。境界案件としてどう見る？", map_path=path)

    assert scene["id"] == "borderline_decision"
    assert scene["learned_entry_count"] == 2
    assert "judgment_feedback:1" in scene["learned_sources"]
    assert any("過去のレビュー済み案件" in item for item in scene["procedure_layer"])
