import json

from api.chat_human_feedback import (
    build_shion_light_tone_feedback_prompt_block,
    load_human_response_feedback,
    summarize_human_response_feedback,
)


def test_load_human_response_feedback_skips_bad_json_and_limits(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"route": "default", "rating": "good"}),
            "not-json",
            json.dumps({"route": "lease_judgment", "rating": "thin"}),
        ]),
        encoding="utf-8",
    )

    rows = load_human_response_feedback(path, limit=2)

    assert rows == [{"route": "lease_judgment", "rating": "thin"}]


def test_summarize_human_response_feedback_matches_route_default_and_recent_comments(tmp_path):
    path = tmp_path / "feedback.jsonl"
    rows = [
        {"route": "relationship_ux", "rating": "good", "response_start": "覚えています", "comment": "自然"},
        {"route": "default", "rating": "thin", "response_start": "もちろんです", "comment": "薄い"},
        {"route": "implementation", "rating": "good", "response_start": "別ルート"},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    summary = summarize_human_response_feedback(path, "relationship_ux")

    assert summary["rows"] == 2
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert summary["positive_starts"] == ["覚えています"]
    assert summary["negative_starts"] == ["もちろんです"]
    assert summary["recent_comments"] == ["薄い", "自然"]


def test_build_shion_light_tone_feedback_prompt_block_triggers_and_skips():
    block = build_shion_light_tone_feedback_prompt_block("硬いね")

    assert "紫苑の軽いトーン修正" in block
    assert "理念説明や自己陶酔に寄せない" in block
    assert build_shion_light_tone_feedback_prompt_block("工作機械リースの確認点を教えて") == ""
