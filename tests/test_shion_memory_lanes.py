import json
from pathlib import Path

from api import shion_memory_lanes
from api import shion_tasks


def test_memory_lanes_keep_personal_tasks_and_judgment_separate(tmp_path: Path):
    personal_path = tmp_path / "user_personal_memory.md"
    personal_path.write_text(
        "\n".join(
            [
                "# User Personal Memory",
                "",
                "## Personal Facts",
                "- [confirmed] Dog name: タム",
                "- [candidate] 好み: 長文が苦手かもしれない",
                "- [sensitive] 家族に関する記憶",
            ]
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "shion_memory_index.json"
    index_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "mem_value",
                        "memory_type": "value_memory",
                        "status": "active",
                        "content": "紫苑は判断を奪わず、判断者の横に置く。",
                        "source_path": "memory/test.md",
                        "private": False,
                    },
                    {
                        "id": "mem_judgment",
                        "memory_type": "judgment_memory",
                        "status": "active",
                        "content": "業務判断に効くものだけ判断資産候補にする。",
                        "source_path": "memory/test.md",
                        "private": False,
                    },
                    {
                        "id": "mem_private",
                        "memory_type": "dialogue_memory",
                        "status": "private",
                        "content": "通常表示しない記憶",
                        "source_path": "memory/private.md",
                        "private": True,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    task_path = tmp_path / "shion_tasks.jsonl"
    shion_tasks.create_task(title="今日のタスクを確認する", due_at="2026-07-20", path=task_path)

    lanes = shion_memory_lanes.build_memory_lanes(
        personal_path=personal_path,
        index_path=index_path,
        task_path=task_path,
    )

    assert lanes["guardrail"] == "ordinary_memory_does_not_automatically_become_judgment_asset"
    assert lanes["lanes"]["personal_memory"]["count"] == 1
    assert "タム" in lanes["lanes"]["personal_memory"]["items"][0]["text"]
    assert lanes["lanes"]["task_memory"]["count"] == 1
    assert lanes["lanes"]["indexed_memory"]["by_type"] == {
        "judgment_memory": 1,
        "value_memory": 1,
    }
    assert "mem_private" not in json.dumps(lanes, ensure_ascii=False)


def test_memory_lanes_can_include_sensitive_and_private_when_explicit(tmp_path: Path):
    personal_path = tmp_path / "user_personal_memory.md"
    personal_path.write_text(
        "## Captured Personal Memories\n- [sensitive] 家族に関する記憶\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "shion_memory_index.json"
    index_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "mem_private",
                        "memory_type": "dialogue_memory",
                        "status": "private",
                        "content": "明示時だけ見る記憶",
                        "private": True,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lanes = shion_memory_lanes.build_memory_lanes(
        include_private=True,
        include_sensitive_personal=True,
        personal_path=personal_path,
        index_path=index_path,
        task_path=tmp_path / "missing_tasks.jsonl",
    )

    assert lanes["lanes"]["personal_memory"]["count"] == 1
    assert lanes["lanes"]["indexed_memory"]["by_type"] == {"dialogue_memory": 1}
