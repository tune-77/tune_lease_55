from scripts.classify_git_ship_candidates import classify_path, parse_porcelain_line, render


def test_generated_outputs_are_avoided():
    paths = [
        "reports/loop_engineering_latest.md",
        "reports/recursive_self_improvement_20260725.md",
        "static_data/macro_context.json",
        "data/shion_pending_tasks.json",
        "frontend/public/lease-grumble/2026-07-25.webp",
    ]

    assert {classify_path(path).bucket for path in paths} == {"avoid"}


def test_source_and_tests_are_include_candidates():
    paths = [
        "scripts/run_daily_improvement_post.sh",
        "scripts/mana_obsidian_curator.py",
        "tests/test_mana_obsidian_curator.py",
        "docs/improvement_source_of_truth.md",
    ]

    assert {classify_path(path).bucket for path in paths} == {"include"}


def test_rule_and_dependency_files_require_review():
    paths = [
        "api/rule_engine/ledger_rules.json",
        "requirements-pipeline.txt",
        "package-lock.json",
    ]

    assert {classify_path(path).bucket for path in paths} == {"review"}


def test_porcelain_rename_uses_destination_path():
    assert parse_porcelain_line("R  old/path.py -> scripts/new_path.py") == "scripts/new_path.py"


def test_render_groups_buckets():
    text = render([
        classify_path("scripts/x.py"),
        classify_path("reports/x_latest.md"),
        classify_path("api/rule_engine/ledger_rules.json"),
    ])

    assert "## include (1)" in text
    assert "## review (1)" in text
    assert "## avoid (1)" in text
