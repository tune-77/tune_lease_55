import json

from api.shion_memory_recall import build_recall_prompt_block, infer_recall_route, recall_memories


def test_infer_recall_route():
    assert infer_recall_route("紫苑とManaの関係は？") == "shion_identity"
    assert infer_recall_route("api/main.pyの実装を直して") == "implementation"
    assert infer_recall_route("この案件は否決かな") == "case_screening"


def test_recall_prefers_route_types(tmp_path):
    index = {
        "records": [
            {
                "id": "mem_value",
                "content": "Mana は紫苑が迷った時に立ち返る上位規範。",
                "memory_type": "value_memory",
                "status": "active",
            },
            {
                "id": "mem_tech",
                "content": "api/main.py はチャット文脈を構築する。",
                "memory_type": "technical_memory",
                "status": "active",
            },
            {
                "id": "mem_private",
                "content": "Private Reflection の私室メモ。",
                "memory_type": "reflection_memory",
                "status": "private",
            },
        ]
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    recalled = recall_memories("Manaについて紫苑はどう思う？", index_path=path)

    assert recalled["route"] == "shion_identity"
    assert recalled["refs"][0] == "mem_value"
    assert "mem_private" not in recalled["refs"]


def test_build_recall_prompt_block():
    block, recalled = build_recall_prompt_block("紫苑の記憶システム")

    assert recalled["route"] == "shion_identity"
    assert "【紫苑の想起メモ】" in block
    assert "想起ルート" in block


def test_case_recall_uses_industry_asset_score_band(tmp_path):
    index = {
        "records": [
            {
                "id": "generic_judgment",
                "content": "境界案件では追加資料を確認して条件付き承認を検討する。",
                "memory_type": "judgment_memory",
                "status": "active",
                "applies_when": ["境界案件", "条件付き承認"],
            },
            {
                "id": "medical_asset_judgment",
                "content": "医療機器は陳腐化と保守契約を確認し、境界案件では条件付き承認に寄せる。",
                "memory_type": "judgment_memory",
                "status": "active",
                "applies_when": ["境界案件", "条件付き承認"],
            },
            {
                "id": "technical_memory",
                "content": "api/main.py はチャット文脈を構築する。",
                "memory_type": "technical_memory",
                "status": "active",
            },
        ]
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    recalled = recall_memories(
        "医療機器の案件、スコア52点。条件付き承認でいいかな？",
        index_path=path,
        limit=2,
    )

    assert recalled["route"] == "case_screening"
    assert recalled["case_profile"]["score_band"] == "boundary"
    assert recalled["refs"][0] == "medical_asset_judgment"
    assert "technical_memory" not in recalled["refs"]


def test_case_recall_caps_value_memories(tmp_path):
    index = {
        "records": [
            {
                "id": "value_a",
                "content": "良心の紫苑は否決や条件付き承認で説明責任を確認する。",
                "memory_type": "value_memory",
                "status": "active",
            },
            {
                "id": "value_b",
                "content": "Mana は否決判断で人を道具として扱わないことを確認する。",
                "memory_type": "value_memory",
                "status": "active",
            },
            {
                "id": "judgment_a",
                "content": "低スコア案件では否決理由と追加確認事項を分ける。",
                "memory_type": "judgment_memory",
                "status": "active",
            },
        ]
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    recalled = recall_memories("スコア35点の案件は否決かな？", index_path=path, limit=3)

    selected_types = [m["memory_type"] for m in recalled["memories"]]
    assert selected_types.count("value_memory") == 1
    assert "judgment_a" in recalled["refs"]
