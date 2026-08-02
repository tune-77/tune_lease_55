"""launchd ジョブの Python 実行系チェックのテスト。

実行系を anaconda から venv に統一したため、「venv に無い依存を
トップレベル import しているジョブ」を出せることが重要。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_launchd_python_env as envcheck  # noqa: E402
import check_obsidian_ops_consistency as checker  # noqa: E402


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------- import 分類 ----------

def test_top_level_imports_are_collected(tmp_path):
    script = _write(tmp_path / "a.py", "import json\nimport requests\nfrom pandas import DataFrame\n")
    assert envcheck.top_level_imports(script) == {"json", "requests", "pandas"}


def test_deferred_imports_are_separated(tmp_path):
    script = _write(
        tmp_path / "b.py",
        "import json\n\ndef run():\n    import numpy\n    from sklearn import tree\n    return numpy, tree\n",
    )
    assert envcheck.top_level_imports(script) == {"json"}
    assert envcheck.deferred_imports(script) == {"numpy", "sklearn"}


def test_module_imported_both_ways_is_not_deferred(tmp_path):
    script = _write(tmp_path / "c.py", "import numpy\n\ndef run():\n    import numpy\n    return numpy\n")
    assert envcheck.top_level_imports(script) == {"numpy"}
    assert envcheck.deferred_imports(script) == set()


def test_relative_imports_are_ignored(tmp_path):
    script = _write(tmp_path / "d.py", "from . import sibling\nfrom .. import parent\n")
    assert envcheck.top_level_imports(script) == set()


def test_syntax_error_does_not_raise(tmp_path):
    script = _write(tmp_path / "broken.py", "def (:\n")
    assert envcheck.top_level_imports(script) == set()
    assert envcheck.deferred_imports(script) == set()


def test_stdlib_and_local_modules_are_excluded():
    assert envcheck._external({"json", "pathlib"}) == set()
    # リポジトリ直下と scripts/ のモジュールはローカル扱い
    assert envcheck._external({"runtime_paths"}) == set()
    assert envcheck._external({"auto_wikilink"}) == set()
    assert envcheck._external({"requests"}) == {"requests"}


# ---------- エントリスクリプトの解決 ----------

def _job(program):
    return checker.Job(label="x", path=Path("/tmp/x.plist"), program_arguments=program)


def test_direct_python_script_is_resolved():
    script = envcheck._script_for_job(
        _job(["/Users/u/clawd/tune_lease_55/.venv/bin/python", "/Users/u/clawd/tune_lease_55/daily_knowledge_feed.py"])
    )
    assert script == REPO_ROOT / "daily_knowledge_feed.py"


def test_shell_wrapper_python_script_is_resolved():
    script = envcheck._script_for_job(
        _job(["/bin/zsh", "/Users/u/clawd/tune_lease_55/scripts/run_obsidian_backup.sh"])
    )
    assert script == REPO_ROOT / "scripts" / "backup_obsidian_vault.py"


def test_shell_wrapper_dash_m_module_is_resolved():
    script = envcheck._script_for_job(
        _job(["/bin/bash", "/Users/u/clawd/tune_lease_55/scripts/run_obsidian_reindex.sh"])
    )
    assert script == REPO_ROOT / "mobile_app" / "rag_daily_maintenance.py"


# ---------- 実 plist に対する回帰 ----------

def test_every_repo_job_resolves_to_an_entry_script():
    unresolved = []
    for job in checker.load_jobs(checker.DEFAULT_LAUNCHD_DIR):
        script = envcheck._script_for_job(job)
        if script is None or not script.exists():
            unresolved.append(job.label)
    assert unresolved == [], f"エントリスクリプトを解決できないジョブ: {unresolved}"


def test_missing_interpreter_is_skipped_not_failed(tmp_path):
    report = envcheck.run(checker.DEFAULT_LAUNCHD_DIR, interpreter_override=str(tmp_path / "nope"))
    assert report["status"] == "skipped"
    assert envcheck.main(["--interpreter", str(tmp_path / "nope")]) == 0
