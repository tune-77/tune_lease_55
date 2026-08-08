from scripts.check_pr_change_risk import (
    Numstat,
    build_report,
    parse_numstat_line,
    render_report,
)


def test_parse_numstat_line_counts_text_changes():
    row = parse_numstat_line("12\t3\tscripts/example.py")

    assert row == Numstat(path="scripts/example.py", additions=12, deletions=3)


def test_parse_numstat_line_treats_binary_as_zero_text_changes():
    row = parse_numstat_line("-\t-\tfrontend/public/lease-grumble/2026-08-08.webp")

    assert row == Numstat(
        path="frontend/public/lease-grumble/2026-08-08.webp",
        additions=0,
        deletions=0,
    )


def test_generated_artifact_fails_report():
    report = build_report(
        ["scripts/fix.py", "reports/loop_engineering_latest.md"],
        [Numstat("scripts/fix.py", 10, 1), Numstat("reports/loop_engineering_latest.md", 20, 4)],
        max_files=80,
        max_text_changes=2500,
    )

    assert report.failed is True
    assert [item.path for item in report.generated_artifacts] == ["reports/loop_engineering_latest.md"]
    assert "generated/runtime artifacts" in render_report(report)


def test_size_limits_fail_report():
    too_many_files = [f"scripts/file_{idx}.py" for idx in range(4)]
    report = build_report(
        too_many_files,
        [Numstat("scripts/big.py", 51, 50)],
        max_files=3,
        max_text_changes=100,
    )

    assert report.failed is True
    text = render_report(report)
    assert "changed file count exceeds limit" in text
    assert "changed line count exceeds limit" in text


def test_normal_source_change_passes_report():
    report = build_report(
        ["scripts/fix.py", "tests/test_fix.py"],
        [Numstat("scripts/fix.py", 25, 5), Numstat("tests/test_fix.py", 30, 2)],
        max_files=80,
        max_text_changes=2500,
    )

    assert report.failed is False
    assert "OK: PR size and generated artifact checks passed." in render_report(report)
