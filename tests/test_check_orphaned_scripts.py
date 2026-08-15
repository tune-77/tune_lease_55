import subprocess
from pathlib import Path

from scripts.check_orphaned_scripts import (
    build_report,
    discover_entry_point_scripts,
    find_references,
    is_entry_point_script,
    read_text_corpus,
)


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)


def test_is_entry_point_script_requires_main_guard(tmp_path: Path):
    entry = tmp_path / "runnable.py"
    entry.write_text("if __name__ == '__main__':\n    pass\n", encoding="utf-8")
    helper = tmp_path / "helper.py"
    helper.write_text("def util():\n    return 1\n", encoding="utf-8")

    assert is_entry_point_script(entry) is True
    assert is_entry_point_script(helper) is False


def test_discover_entry_point_scripts_filters_non_entry_points(tmp_path: Path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "runnable.py").write_text(
        "if __name__ == '__main__':\n    pass\n", encoding="utf-8"
    )
    (scripts_dir / "lib_only.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    found = discover_entry_point_scripts(scripts_dir)

    assert [p.name for p in found] == ["runnable.py"]


def test_find_references_excludes_self_and_own_test(tmp_path: Path):
    root = tmp_path
    script = root / "scripts" / "build_thing.py"
    script.parent.mkdir(parents=True)
    script.write_text("if __name__ == '__main__':\n    pass\n", encoding="utf-8")

    own_test = root / "tests" / "test_build_thing.py"
    own_test.parent.mkdir(parents=True)
    own_test.write_text("from scripts.build_thing import main\n", encoding="utf-8")

    pipeline = root / "scripts" / "run_daily.sh"
    pipeline.write_text("python scripts/build_thing.py\n", encoding="utf-8")

    corpus = read_text_corpus([script, own_test, pipeline])

    references = find_references(script, corpus=corpus, root=root)

    assert references == ["scripts/run_daily.sh"]


def test_find_references_returns_empty_when_only_self_and_test_mention_it(tmp_path: Path):
    root = tmp_path
    script = root / "scripts" / "orphan.py"
    script.parent.mkdir(parents=True)
    script.write_text("if __name__ == '__main__':\n    pass\n", encoding="utf-8")

    own_test = root / "tests" / "test_orphan.py"
    own_test.parent.mkdir(parents=True)
    own_test.write_text("from scripts.orphan import main\n", encoding="utf-8")

    corpus = read_text_corpus([script, own_test])

    references = find_references(script, corpus=corpus, root=root)

    assert references == []


def test_build_report_flags_unwired_script_and_clears_once_wired(tmp_path: Path):
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "tests").mkdir()

    orphan = root / "scripts" / "build_thing.py"
    orphan.write_text("if __name__ == '__main__':\n    pass\n", encoding="utf-8")
    (root / "tests" / "test_build_thing.py").write_text(
        "from scripts.build_thing import main\n", encoding="utf-8"
    )

    _init_git_repo(root)

    report = build_report(root=root, generated_at="2026-01-01T00:00:00+00:00")
    assert report["summary"]["orphaned"] == 1
    assert report["orphaned_scripts"][0]["script"] == "scripts/build_thing.py"

    pipeline = root / "scripts" / "run_daily.sh"
    pipeline.write_text("python scripts/build_thing.py\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "wire it in"], cwd=root, check=True)

    report_after = build_report(root=root, generated_at="2026-01-01T00:00:00+00:00")
    assert report_after["summary"]["orphaned"] == 0
