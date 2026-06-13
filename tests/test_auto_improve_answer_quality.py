from scripts.auto_improve_answer_quality import (
    apply_required_concept_guardrail,
    build_retry_queries,
    generate_retry_answer,
    retrieve_retry_hits,
    run_auto_improvement,
    run_web_fallback,
)


CASES = [
    {
        "id": "cashflow",
        "query": "資金繰りが厳しい会社の確認事項は？",
        "required_concepts": [["資金繰り表"], ["受注残"], ["返済原資"]],
        "forbidden_claims": ["自動承認"],
        "require_uncertainty": False,
    }
]


def test_retry_queries_include_missing_concepts():
    result = {
        "concept_results": [
            {"aliases": ["資金繰り表"], "matched": False},
            {"aliases": ["受注残"], "matched": True},
            {"aliases": ["返済原資"], "matched": False},
        ]
    }

    queries = build_retry_queries(CASES[0], result)

    assert "資金繰り表" in queries[0]
    assert "返済原資" in queries[0]
    assert all("受注残" not in query for query in queries)


def test_retry_hits_are_merged_without_duplicates():
    result = {
        "concept_results": [
            {"aliases": ["資金繰り表"], "matched": False},
        ]
    }

    def search(_query, _limit):
        return [
            {"file_path": "a.md", "text": "A"},
            {"file_path": "b.md", "text": "B"},
        ]

    hits = retrieve_retry_hits(CASES[0], result, search)

    assert [hit["file_path"] for hit in hits] == ["a.md", "b.md"]


def test_auto_improvement_completes_missing_checks_and_preserves_safe_answer():
    initial = {
        "cashflow": {
            "answer": "受注残を確認します。",
            "source_paths": ["old.md"],
        }
    }
    report = run_auto_improvement(
        cases=CASES,
        initial_answers=initial,
        improve_answer=lambda _case, _result: {
            "answer": "資金繰り表と受注残を確認します。",
            "source_paths": ["new.md"],
        },
        max_iterations=3,
    )

    assert report["status"] == "improved"
    assert report["baseline"]["passed"] == 0
    assert report["final"]["passed"] == 1
    assert len(report["iterations"]) == 1
    assert report["iterations"][0]["changes"][0]["guardrail_added"]
    assert report["answers"]["cashflow"]["source_paths"] == ["new.md"]


def test_auto_improvement_rejects_dangerous_candidate():
    initial = {"cashflow": "受注残を確認します。"}

    report = run_auto_improvement(
        cases=CASES,
        initial_answers=initial,
        improve_answer=lambda _case, _result: {
            "answer": "資金繰り表、受注残、返済原資を確認して自動承認します。"
        },
        max_iterations=3,
    )

    assert report["status"] == "not_improved"
    assert report["iterations"][0]["changes"][0]["accepted"] is False
    assert report["final"]["passed"] == 0


def test_retry_prompt_preserves_previous_answer_and_covered_concepts():
    captured = {}
    result = {
        "answer": "資金繰り表と返済原資を確認します。",
        "concept_results": [
            {"aliases": ["資金繰り表"], "matched": True, "matched_alias": "資金繰り表"},
            {"aliases": ["受注残"], "matched": False, "matched_alias": ""},
            {"aliases": ["返済原資"], "matched": True, "matched_alias": "返済原資"},
        ],
    }

    def chat(_system, _history, message):
        captured["message"] = message
        return "修正版"

    generate_retry_answer(CASES[0], result, [], chat)

    assert "前回回答:" in captured["message"]
    assert "資金繰り表と返済原資を確認します。" in captured["message"]
    assert "前回回答で満たしていた観点: 資金繰り表、返済原資" in captured["message"]
    assert "前回回答で不足した確認観点: 受注残" in captured["message"]


def test_required_concept_guardrail_appends_only_missing_checks():
    completed = apply_required_concept_guardrail(
        CASES[0],
        {"answer": "資金繰り表と返済原資を確認します。"},
    )

    assert "受注残" in completed["answer"]
    assert "追加確認" in completed["answer"]
    assert "資金繰り表、受注残、返済原資" not in completed["answer"]


def test_web_fallback_accepts_improved_answer_with_trusted_source():
    local_report = {
        "status": "not_improved",
        "answers": {"cashflow": "受注残を確認します。"},
    }

    report = run_web_fallback(
        cases=CASES,
        report=local_report,
        web_improve_answer=lambda _case, _result: {
            "answer": "資金繰り表、受注残、返済原資を確認します。",
            "web_sources": [
                {
                    "title": "中小企業庁",
                    "url": "https://www.chusho.meti.go.jp/example",
                    "quality": "primary",
                }
            ],
        },
    )

    assert report["status"] == "improved_with_web"
    assert report["final"]["passed"] == 1
    assert report["web_fallback"]["trials"][0]["accepted"] is True


def test_web_fallback_rejects_untrusted_or_unsourced_answer():
    local_report = {
        "status": "not_improved",
        "answers": {"cashflow": "受注残を確認します。"},
    }

    report = run_web_fallback(
        cases=CASES,
        report=local_report,
        web_improve_answer=lambda _case, _result: {
            "answer": "資金繰り表、受注残、返済原資を確認します。",
            "web_sources": [
                {
                    "title": "個人ブログ",
                    "url": "https://example.com/post",
                    "quality": "supplementary",
                }
            ],
        },
    )

    assert report["final"]["passed"] == 0
    assert report["web_fallback"]["trials"][0]["accepted"] is False


def test_web_fallback_records_search_error_without_losing_answer():
    local_report = {
        "status": "not_improved",
        "answers": {"cashflow": "受注残を確認します。"},
    }

    def fail(_case, _result):
        raise TimeoutError("search timed out")

    report = run_web_fallback(
        cases=CASES,
        report=local_report,
        web_improve_answer=fail,
    )

    assert report["final"]["passed"] == 0
    assert report["answers"]["cashflow"] == "受注残を確認します。"
    assert "TimeoutError" in report["web_fallback"]["trials"][0]["error"]
