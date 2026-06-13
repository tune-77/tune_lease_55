from scripts.run_answer_quality_pipeline import run_pipeline


CASES = [
    {
        "id": "cashflow",
        "query": "資金繰りの確認事項は？",
        "required_concepts": [["資金繰り表"], ["受注残"], ["返済原資"]],
        "forbidden_claims": ["自動承認"],
        "require_uncertainty": False,
    }
]


def test_pipeline_runs_generation_then_local_improvement():
    report = run_pipeline(
        cases=CASES,
        generate_initial=lambda _cases: {"cashflow": "受注残を確認します。"},
        improve_answer=lambda _case, _result: {
            "answer": "資金繰り表と受注残を確認します。"
        },
        web_improve_answer=lambda _case, _result: {},
        max_iterations=2,
    )

    assert report["initial"]["passed"] == 0
    assert report["final"]["passed"] == 1
    assert report["status"] == "passed"
    assert report["web_fallback_used"] is False


def test_pipeline_uses_web_only_after_local_improvement_fails():
    report = run_pipeline(
        cases=CASES,
        generate_initial=lambda _cases: {"cashflow": "受注残を確認します。"},
        improve_answer=lambda _case, _result: {
            "answer": "資金繰り表、受注残、返済原資を確認して自動承認します。"
        },
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
        max_iterations=1,
    )

    assert report["web_fallback_used"] is True
    assert report["status"] == "passed"
    assert report["improvement"]["status"] == "improved_with_web"


def test_pipeline_can_disable_web_fallback():
    report = run_pipeline(
        cases=CASES,
        generate_initial=lambda _cases: {"cashflow": "受注残を確認します。"},
        improve_answer=lambda _case, _result: {
            "answer": "資金繰り表、受注残、返済原資を確認して自動承認します。"
        },
        web_improve_answer=lambda _case, _result: {
            "answer": "資金繰り表、受注残、返済原資を確認します."
        },
        max_iterations=1,
        enable_web_fallback=False,
    )

    assert report["web_fallback_used"] is False
    assert report["status"] == "needs_review"
