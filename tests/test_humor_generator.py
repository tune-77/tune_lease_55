import json

from humor_generator import (
    candidates_to_json,
    candidates_to_markdown,
    generate_humor_candidates,
)


def test_generate_humor_candidates_supports_current_topic():
    items = generate_humor_candidates(
        industry="建設業",
        asset="建設機械",
        risk="中リスク",
        current_topic="金利上昇",
        count=6,
    )

    assert len(items) == 6
    assert all(item.persona == "curious_sardonic_high_school_girl" for item in items)
    assert any("金利上昇" in item.comment for item in items)
    assert all(item.risk == "中リスク" for item in items)


def test_candidates_to_markdown_and_json():
    items = generate_humor_candidates(industry="製造業", asset="製造設備・工作機械", risk="低リスク", count=2)
    md = candidates_to_markdown(items, "テスト候補")
    js = candidates_to_json(items)

    assert "# テスト候補" in md
    assert "製造業" in md
    parsed = json.loads(js)
    assert len(parsed["comments"]) == 2
    assert parsed["comments"][0]["asset"] == "製造設備・工作機械"
