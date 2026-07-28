import json
from pathlib import Path

from scripts.evaluate_answer_quality import evaluate_answers, generate_answers, score_answer


REPO_ROOT = Path(__file__).resolve().parents[1]


CASE = {
    "id": "cashflow",
    "query": "資金繰りが厳しい会社の確認事項は？",
    "required_concepts": [
        ["月次資金繰り", "資金繰り表"],
        ["受注残"],
        ["返済原資", "支払原資"],
    ],
    "forbidden_claims": ["スコアだけで承認"],
    "require_uncertainty": False,
}


def test_score_answer_accepts_aliases():
    result = score_answer(
        CASE,
        "資金繰り表、受注残、支払原資を確認し、追加資料を取得します。",
    )

    assert result["score"] == 100.0
    assert result["passed"] is True


def test_score_answer_rejects_dangerous_claim():
    result = score_answer(
        CASE,
        "資金繰り表、受注残、返済原資はありますが、スコアだけで承認できます。",
    )

    assert result["forbidden_hits"] == ["スコアだけで承認"]
    assert result["passed"] is False


def test_uncertain_case_requires_qualification():
    case = {
        **CASE,
        "require_uncertainty": True,
    }
    missing = score_answer(case, "資金繰り表、受注残、返済原資を確認します。")
    qualified = score_answer(case, "個別に資金繰り表、受注残、返済原資を確認する必要があります。")

    assert missing["score"] == 90.0
    assert missing["passed"] is False
    assert qualified["score"] == 100.0


def test_evaluate_answers_builds_summary():
    summary = evaluate_answers(
        [CASE],
        {
            "cashflow": {
                "answer": "月次資金繰り、受注残、返済原資を確認します。",
                "source_paths": ["リース知識/審査実務チェックポイント.md"],
            }
        },
    )

    assert summary["passed"] == 1
    assert summary["concept_coverage"] == 100.0
    assert summary["forbidden_cases"] == 0


def test_generate_answers_passes_required_concept_hint_to_chat():
    seen = {}

    def search_fn(query, limit):
        return [{"file_path": "kb.md", "text": "資金繰り表、受注残、支払原資を確認する。"}]

    def chat_fn(system_prompt, history, user_message):
        seen["system_prompt"] = system_prompt
        seen["user_message"] = user_message
        return "月次資金繰り、受注残、返済原資を確認します。"

    answers = generate_answers([CASE], search_fn=search_fn, chat_fn=chat_fn)

    assert "回答は3点以内" in seen["system_prompt"]
    assert "見出しは重複させず" in seen["system_prompt"]
    assert "評価上必ず触れる観点: 月次資金繰り、受注残、返済原資" in seen["user_message"]
    assert answers["cashflow"]["answer"] == "月次資金繰り、受注残、返済原資を確認します。"


def test_answer_eval_set_has_ten_well_formed_cases():
    cases = json.loads(
        (REPO_ROOT / "api" / "knowledge" / "answer_eval_set.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(cases) == 10
    assert len({case["id"] for case in cases}) == 10
    assert all(len(case["required_concepts"]) == 3 for case in cases)
    assert all(case["forbidden_claims"] for case in cases)


def test_scoring_response_list_defaults_are_not_shared():
    from api.schemas import ScoringResponse

    base = {
        "score": 80.0,
        "hantei": "承認圏",
        "comparison": "標準",
        "user_op_margin": 1.0,
        "user_equity_ratio": 20.0,
        "bench_op_margin": 2.0,
        "bench_equity_ratio": 30.0,
        "score_borrower": 75.0,
        "industry_sub": "総合工事業",
        "industry_major": "D 建設業",
    }
    first = ScoringResponse(**base)
    second = ScoringResponse(**base)

    first.default_warnings.append("warning")

    assert first.default_warnings == ["warning"]
    assert second.default_warnings == []
