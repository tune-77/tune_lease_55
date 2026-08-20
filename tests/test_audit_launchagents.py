from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_launchagents as audit  # noqa: E402


def _plist(
    path: Path,
    *,
    label: str = "com.tunelease.test",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "Label": label,
        "ProgramArguments": args or ["/repo/.venv/bin/python", "/repo/job.py"],
        "EnvironmentVariables": env or {},
        "RunAtLoad": False,
    }
    path.write_bytes(plistlib.dumps(data))
    return path


def test_repo_plists_collects_standard_locations(tmp_path):
    _plist(tmp_path / "launchd" / "com.tunelease.a.plist")
    _plist(tmp_path / "scripts" / "launchd" / "com.tunelease.b.plist")
    _plist(tmp_path / "scripts" / "com.tunelease.c.plist")
    _plist(tmp_path / "other" / "com.tunelease.d.plist")

    names = [path.name for path in audit.repo_plists(tmp_path)]

    assert names == [
        "com.tunelease.a.plist",
        "com.tunelease.b.plist",
        "com.tunelease.c.plist",
    ]


def test_installed_match_detects_diff_missing_and_extra(tmp_path):
    repo = tmp_path / "repo"
    installed = tmp_path / "installed"
    _plist(repo / "launchd" / "com.tunelease.same.plist", label="com.tunelease.same")
    _plist(repo / "launchd" / "com.tunelease.diff.plist", label="com.tunelease.diff")
    _plist(repo / "launchd" / "com.tunelease.missing.plist", label="com.tunelease.missing")
    _plist(installed / "com.tunelease.same.plist", label="com.tunelease.same")
    _plist(installed / "com.tunelease.diff.plist", label="com.tunelease.diff", env={"X": "changed"})
    _plist(installed / "com.tunelease.extra.plist", label="com.tunelease.extra")

    findings = audit.check_installed_matches_repo(installed_dir=installed, repo_root=repo)
    messages = "\n".join(f.message for f in findings)

    assert "com.tunelease.diff.plist: installed plist differs from repo" in messages
    assert "com.tunelease.missing.plist: installed plist is missing" in messages
    assert "com.tunelease.extra.plist: installed plist has no repo source of truth" in messages
    assert all(f.level == audit.ERROR for f in findings)


def test_plist_policy_flags_secret_keys_without_values(tmp_path):
    path = _plist(
        tmp_path / "com.tunelease.secret.plist",
        env={"GEMINI_API_KEY": "do-not-print"},
    )

    findings = audit.check_plist_policy([path])

    assert [f.level for f in findings] == [audit.ERROR]
    assert "GEMINI_API_KEY" in findings[0].message
    assert "do-not-print" not in findings[0].message


def test_plist_policy_flags_anaconda_python(tmp_path):
    path = _plist(
        tmp_path / "com.tunelease.anaconda.plist",
        args=["/Users/u/anaconda3/bin/python3", "/repo/job.py"],
    )

    findings = audit.check_plist_policy([path])

    assert [f.level for f in findings] == [audit.ERROR]
    assert "anaconda3" in findings[0].message


def test_clean_policy_reports_info(tmp_path):
    path = _plist(tmp_path / "com.tunelease.clean.plist")

    findings = audit.check_plist_policy([path])

    assert [f.level for f in findings] == [audit.INFO]


def test_loaded_state_uses_launchctl_result(monkeypatch, tmp_path):
    path = _plist(tmp_path / "com.tunelease.test.plist", label="com.tunelease.test")

    def fake_print(label: str, *, uid: str | None = None):
        assert label == "com.tunelease.test"
        return subprocess.CompletedProcess(["launchctl"], 113, "", "not found")

    monkeypatch.setattr(audit, "_launchctl_print", fake_print)
    findings = audit.check_loaded_state([path], uid="501")

    assert [f.level for f in findings] == [audit.WARNING]
    assert "not loaded" in findings[0].message


def test_current_repo_plists_have_no_policy_errors():
    findings = audit.check_plist_policy(audit.repo_plists())
    errors = [f for f in findings if f.level == audit.ERROR]
    assert errors == [], "\n".join(f.message for f in errors)
