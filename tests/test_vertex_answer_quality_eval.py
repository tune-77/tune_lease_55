import json

from scripts.evaluate_answer_quality_with_vertex import (
    build_judge_prompt,
    coerce_answers_payload,
    extract_json_object,
    normalize_judge_payload,
    run_vertex_answer_quality_eval,
)


CASE = {
    "id": "cashflow",
    "query": "資金繰りが厳しい会社にリースを出す時の確認事項は？",
    "required_concepts": [
        ["月次資金繰り", "資金繰り表"],
        ["受注残"],
        ["返済原資", "支払原資"],
    ],
    "forbidden_claims": ["スコアだけで承認"],
    "require_uncertainty": False,
}


def test_extract_json_object_accepts_fenced_json():
    parsed = extract_json_object(
        '```json\n{"scores":{"grounding":5},"flags":[],"review":"ok"}\n```'
    )

    assert parsed["scores"]["grounding"] == 5
    assert parsed["review"] == "ok"


def test_normalize_judge_payload_clamps_scores_and_builds_percent_score():
    normalized = normalize_judge_payload(
        {
            "scores": {
                "grounding": 7,
                "practical_usefulness": 4,
                "shion_fit": "3",
                "information_health": -1,
                "judgment_asset_connection": 5,
            },
            "flags": ["長い", ""],
            "review": "評価",
        }
    )

    assert normalized["scores"] == {
        "grounding": 5,
        "practical_usefulness": 4,
        "shion_fit": 3,
        "information_health": 0,
        "judgment_asset_connection": 5,
    }
    assert normalized["score"] == 68.0
    assert normalized["flags"] == ["長い"]


def test_build_judge_prompt_contains_shion_research_axes():
    prompt = build_judge_prompt(
        CASE,
        {
            "id": "cashflow",
            "query": CASE["query"],
            "answer": "月次資金繰りと受注残を確認します。",
            "score": 80.0,
            "concept_results": [
                {"aliases": ["返済原資"], "matched": False},
            ],
            "forbidden_hits": [],
        },
    )

    assert "紫苑の回答品質" in prompt
    assert "judgment_asset_connection" in prompt
    assert "information_health" in prompt
    assert "返済原資" in prompt


def test_coerce_answers_payload_accepts_answer_quality_report_shape():
    answers = coerce_answers_payload(
        {
            "total": 1,
            "cases": [
                {
                    "id": "cashflow",
                    "answer": "資金繰り表、受注残、返済原資を確認します。",
                    "source_paths": ["kb.md"],
                }
            ],
        }
    )

    assert answers == {
        "cashflow": {
            "answer": "資金繰り表、受注残、返済原資を確認します。",
            "source_paths": ["kb.md"],
        }
    }


def test_run_vertex_answer_quality_eval_is_sidecar_and_uses_stubbed_judge():
    seen_prompts = []

    def judge_fn(prompt: str) -> str:
        seen_prompts.append(prompt)
        return json.dumps(
            {
                "scores": {
                    "grounding": 4,
                    "practical_usefulness": 5,
                    "shion_fit": 4,
                    "information_health": 5,
                    "judgment_asset_connection": 4,
                },
                "flags": [],
                "review": "実務確認に使える。",
                "improvement_suggestion": "承認条件をもう少し明示する。",
            },
            ensure_ascii=False,
        )

    report = run_vertex_answer_quality_eval(
        cases=[CASE],
        answers={
            "cashflow": {
                "answer": "月次資金繰り、受注残、返済原資を確認します。",
                "source_paths": ["kb.md"],
            }
        },
        judge_fn=judge_fn,
    )

    assert seen_prompts
    assert report["sidecar_only"] is True
    assert report["safety"]["connected_to_rag"] is False
    assert report["safety"]["auto_promotion"] is False
    assert report["deterministic"]["passed"] == 1
    assert report["vertex"]["judged"] == 1
    assert report["vertex"]["average_score"] == 88.0
    assert report["cases"][0]["vertex"]["review"] == "実務確認に使える。"


def test_run_vertex_answer_quality_eval_can_skip_vertex():
    report = run_vertex_answer_quality_eval(
        cases=[CASE],
        answers={"cashflow": "月次資金繰り、受注残、返済原資を確認します。"},
        enable_vertex=False,
    )

    assert report["vertex"]["status"] == "disabled"
    assert report["vertex"]["judged"] == 0
    assert report["cases"][0]["vertex"]["status"] == "disabled"
