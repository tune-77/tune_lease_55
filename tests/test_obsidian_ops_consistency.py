"""launchd の Obsidian 運用整合性チェックのテスト。

リポジトリ内の実 plist に対する回帰テストを含む: Vault の env 分裂や
夜間ジョブの同時発火が再び入り込んだら、ここで落ちる。
"""
from __future__ import annotations

import plistlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_obsidian_ops_consistency as checker  # noqa: E402


def _job(label, env=None, hour=3, minute=0, calendar_extra=None, program=None):
    calendar = {"Hour": hour, "Minute": minute}
    calendar.update(calendar_extra or {})
    return checker.Job(
        label=label,
        path=Path(f"/tmp/{label}.plist"),
        env=dict(env or {}),
        program_arguments=list(program or []),
        calendar=calendar,
    )


VAULT = "/Users/x/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
BOTH = {"OBSIDIAN_VAULT_PATH": VAULT, "OBSIDIAN_VAULT": VAULT}
VAULT_LABEL = "com.tunelease.obsidian-reindex"
OTHER_VAULT_LABEL = "com.tunelease.obsidian-backup"


# ---------- vault env ----------

def test_vault_job_with_both_env_vars_is_clean():
    findings = checker.check_vault_env([_job(VAULT_LABEL, BOTH)])
    assert findings == []


def test_missing_env_var_is_an_error():
    findings = checker.check_vault_env(
        [_job(VAULT_LABEL, {"OBSIDIAN_VAULT_PATH": VAULT})]
    )
    assert [f.level for f in findings] == [checker.ERROR]
    assert "OBSIDIAN_VAULT" in findings[0].message


def test_conflicting_env_vars_in_one_job_is_an_error():
    findings = checker.check_vault_env(
        [_job(VAULT_LABEL, {"OBSIDIAN_VAULT_PATH": VAULT, "OBSIDIAN_VAULT": "/other"})]
    )
    assert any(f.level == checker.ERROR for f in findings)


def test_vault_path_mismatch_across_jobs_is_an_error():
    other = {"OBSIDIAN_VAULT_PATH": "/other", "OBSIDIAN_VAULT": "/other"}
    findings = checker.check_vault_env(
        [_job(VAULT_LABEL, BOTH), _job(OTHER_VAULT_LABEL, other)]
    )
    assert any("一致していません" in f.message for f in findings)


def test_trailing_slash_is_not_a_mismatch():
    findings = checker.check_vault_env(
        [
            _job(VAULT_LABEL, BOTH),
            _job(OTHER_VAULT_LABEL, {k: v + "/" for k, v in BOTH.items()}),
        ]
    )
    assert findings == []


def test_non_vault_job_declaring_vault_env_is_a_warning():
    findings = checker.check_vault_env([_job("com.tunelease.unrelated", BOTH)])
    assert [f.level for f in findings] == [checker.WARNING]


# ---------- schedule ----------

def test_same_time_vault_jobs_collide():
    findings = checker.check_schedule_collisions(
        [_job(VAULT_LABEL, BOTH, hour=6), _job(OTHER_VAULT_LABEL, BOTH, hour=6)]
    )
    assert [f.level for f in findings] == [checker.ERROR]


def test_near_time_vault_jobs_warn():
    findings = checker.check_schedule_collisions(
        [
            _job(VAULT_LABEL, BOTH, hour=6, minute=0),
            _job(OTHER_VAULT_LABEL, BOTH, hour=6, minute=5),
        ]
    )
    assert [f.level for f in findings] == [checker.WARNING]


def test_staggered_vault_jobs_do_not_collide():
    findings = checker.check_schedule_collisions(
        [
            _job(VAULT_LABEL, BOTH, hour=6, minute=0),
            _job(OTHER_VAULT_LABEL, BOTH, hour=6, minute=20),
        ]
    )
    assert findings == []


def test_different_weekdays_never_coincide():
    findings = checker.check_schedule_collisions(
        [
            _job(VAULT_LABEL, BOTH, hour=4, calendar_extra={"Weekday": 0}),
            _job(OTHER_VAULT_LABEL, BOTH, hour=4, calendar_extra={"Weekday": 3}),
        ]
    )
    assert findings == []


def test_weekly_and_daily_jobs_can_coincide():
    findings = checker.check_schedule_collisions(
        [
            _job(VAULT_LABEL, BOTH, hour=4, calendar_extra={"Weekday": 0}),
            _job(OTHER_VAULT_LABEL, BOTH, hour=4),
        ]
    )
    assert [f.level for f in findings] == [checker.ERROR]


def test_non_vault_job_collision_is_ignored():
    findings = checker.check_schedule_collisions(
        [_job("com.tunelease.a", hour=6), _job("com.tunelease.b", hour=6)]
    )
    assert findings == []


# ---------- interpreters ----------

VENV_PY = "/repo/tune_lease_55/.venv/bin/python"


def test_single_venv_interpreter_is_clean():
    jobs = [
        _job(VAULT_LABEL, BOTH, program=[VENV_PY, "x.py"]),
        _job(OTHER_VAULT_LABEL, BOTH, program=[VENV_PY, "y.py"]),
    ]
    assert checker.check_interpreters(jobs) == []


def test_split_interpreters_is_an_error():
    jobs = [
        _job(VAULT_LABEL, BOTH, program=[VENV_PY, "x.py"]),
        _job(OTHER_VAULT_LABEL, BOTH, program=["/anaconda3/bin/python3", "y.py"]),
    ]
    findings = checker.check_interpreters(jobs)
    assert any(f.level == checker.ERROR for f in findings)


def test_non_venv_interpreter_warns_even_when_unified():
    jobs = [
        _job(VAULT_LABEL, BOTH, program=["/anaconda3/bin/python", "x.py"]),
        _job(OTHER_VAULT_LABEL, BOTH, program=["/anaconda3/bin/python", "y.py"]),
    ]
    findings = checker.check_interpreters(jobs)
    assert [f.level for f in findings] == [checker.WARNING]
    assert "venv" in findings[0].message


def test_interpreter_from_python_bin_env_is_detected():
    job = _job(VAULT_LABEL, {**BOTH, "PYTHON_BIN": "/anaconda3/bin/python"}, program=["/bin/zsh", "run.sh"])
    assert job.interpreter == "/anaconda3/bin/python"


def test_repo_launchd_uses_a_single_venv_interpreter():
    jobs = checker.load_jobs(checker.DEFAULT_LAUNCHD_DIR)
    findings = checker.check_interpreters(jobs)
    assert findings == [], "\n".join(f.message for f in findings)


# ---------- regression over the real plists ----------

def test_repo_launchd_plists_are_all_parseable():
    jobs = checker.load_jobs(checker.DEFAULT_LAUNCHD_DIR)
    assert jobs, "launchd/ に plist がありません"
    for job in jobs:
        with job.path.open("rb") as handle:
            plistlib.load(handle)


def test_repo_launchd_has_no_vault_env_errors():
    jobs = checker.load_jobs(checker.DEFAULT_LAUNCHD_DIR)
    errors = [f for f in checker.check_vault_env(jobs) if f.level == checker.ERROR]
    assert errors == [], "\n".join(f.message for f in errors)


def test_repo_launchd_has_no_schedule_collisions():
    jobs = checker.load_jobs(checker.DEFAULT_LAUNCHD_DIR)
    collisions = checker.check_schedule_collisions(jobs)
    assert collisions == [], "\n".join(f.message for f in collisions)


def test_no_hardcoded_vault_paths_remain():
    findings = checker.scan_hardcoded_vault_paths()
    assert findings == [], "\n".join(f.message for f in findings)


def test_hardcoded_scan_flags_a_new_offender(tmp_path):
    offender = tmp_path / "bad.py"
    offender.write_text(
        'VAULT = "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"\n',
        encoding="utf-8",
    )
    findings = checker.scan_hardcoded_vault_paths(tmp_path)
    assert [f.level for f in findings] == [checker.ERROR]
    assert "bad.py" in findings[0].message


def test_hardcoded_scan_ignores_tests(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        'assert "iCloud~md~obsidian" in path\n', encoding="utf-8"
    )
    assert checker.scan_hardcoded_vault_paths(tmp_path) == []


def test_every_vault_job_label_has_a_plist():
    labels = {job.label for job in checker.load_jobs(checker.DEFAULT_LAUNCHD_DIR)}
    missing = checker.VAULT_JOB_LABELS - labels
    assert missing == set(), f"VAULT_JOB_LABELS に実在しないジョブがあります: {missing}"
