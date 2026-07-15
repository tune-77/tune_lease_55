from __future__ import annotations

from scripts.screening_terms_audit import build_audit, render_markdown


def test_screening_terms_audit_flags_unsafe_pd_fallback(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text(
        "\n".join(
            [
                "pd = result.pd",
                "text = '算出済みPD: 未算出'",
                "warning = '高リスク財務パターン警告（実PDではありません）'",
            ]
        ),
        encoding="utf-8",
    )

    report = build_audit([target])

    assert report["counts"]["warn"] == 1
    assert report["counts"]["ok"] >= 2
    markdown = render_markdown(report)
    assert "result.pd" in markdown
    assert "actual_pd" in markdown


def test_screening_terms_audit_classifies_ambiguous_pd_as_review(tmp_path):
    target = tmp_path / "sample.tsx"
    target.write_text("const label = 'PDとAIスコアの関係';\n", encoding="utf-8")

    report = build_audit([target])

    assert report["counts"]["review"] == 2
    assert report["findings"][0]["severity"] == "review"
