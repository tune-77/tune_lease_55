from __future__ import annotations

from scripts import screening_terms_audit as audit_mod
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


def test_screening_terms_audit_ok_for_explicit_pd_disclaimers(tmp_path):
    target = tmp_path / "sample.tsx"
    target.write_text(
        "\n".join(
            [
                "const title = 'PD（デフォルト確率）の解説';",
                "const note = '実績デフォルトで校正したPDではない。';",
                "const hint = '候補重みはPDやスコアではなく、検討優先度です。';",
            ]
        ),
        encoding="utf-8",
    )

    report = build_audit([target])

    assert report["counts"].get("review", 0) == 0
    assert report["counts"].get("warn", 0) == 0


def test_screening_terms_audit_ignores_pd_inside_unrelated_identifiers(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text(
        "\n".join(
            [
                'conn.execute("UPDATE past_cases SET data = ? WHERE id = ?", updates)',
                "mode = 'DRY RUN' if args.dry_run else 'UPDATED'",
                "_CPD_MIN_CASES = 20",
                "manual = 'PDF形式で出力'",
                "review_cycle = 'PDCAサイクルを回す'",
            ]
        ),
        encoding="utf-8",
    )

    report = build_audit([target])

    assert report["counts"].get("review", 0) == 0
    assert report["counts"].get("warn", 0) == 0


def test_screening_terms_audit_still_flags_standalone_pd_terms(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("comment = 'PD値は50%'\n", encoding="utf-8")

    report = build_audit([target])

    assert report["counts"]["warn"] == 1


def test_main_fails_when_no_files_are_scanned(monkeypatch, capsys, tmp_path):
    """scan targetsが移動/リネームされて1件もスキャンできない場合は検知する。"""
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "screening_terms_audit.py",
            "--json", str(output_json),
            "--report", str(output_md),
            "--target", str(tmp_path / "does-not-exist"),
        ],
    )

    exit_code = audit_mod.main()

    assert exit_code == 1
    assert "1件もスキャンできませんでした" in capsys.readouterr().err


def test_main_is_ok_when_files_scanned_but_clean(monkeypatch, capsys, tmp_path):
    """スキャンはできて危険表現が0件（クリーン）なのは正常。"""
    target = tmp_path / "sample.py"
    target.write_text("x = 1\n", encoding="utf-8")
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "screening_terms_audit.py",
            "--json", str(output_json),
            "--report", str(output_md),
            "--target", str(target),
        ],
    )

    exit_code = audit_mod.main()

    assert exit_code == 0
