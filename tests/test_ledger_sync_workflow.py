"""台帳PRが承認待ちで止まらないようにする配線のテスト。

bot（GITHUB_TOKEN）が作ったPRでは pr-checks.yml の run が action_required で止まり、
必須チェックが1件も報告されないため auto-merge が永久に待つ。PR #943 の run #1029 で、
bot のPRが一度マージされた後も承認待ちのままになることを実測している
（「マージ実績ができれば承認不要になる」という想定は誤りだった）。

ledger-sync.yml は push 後に pr-checks.yml を workflow_dispatch で起動して必須チェックを
埋める。workflow_dispatch は GITHUB_TOKEN からでも発火する唯一の入口なので、この起動が
消えると台帳PRは毎回人間の承認待ちで止まる（滞留検知が赤にするまで誰も気づけない）。
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER_SYNC = ROOT / ".github/workflows/ledger-sync.yml"
PR_CHECKS = ROOT / ".github/workflows/pr-checks.yml"

# master の branch protection が必須にしている status check。
# ここに無いジョブ名を pr-checks.yml から消すと、台帳PRは永久にマージできなくなる。
REQUIRED_CHECKS = ("frontend", "python-syntax", "pytest-core", "cloudrun-data-safety")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sync_job() -> dict:
    return _load(LEDGER_SYNC)["jobs"]["sync-ledger"]


def _step(name: str) -> dict:
    for step in _sync_job()["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"ledger-sync.yml に step '{name}' が無い")


def test_ledger_sync_dispatches_pr_checks() -> None:
    """承認不要の入口から必須チェックを起動する配線が残っていること。"""
    step = _step("Run pr-checks on the sync branch")

    assert step["env"]["TARGET_WORKFLOW"] == "pr-checks.yml"
    assert 'gh workflow run "${TARGET_WORKFLOW}" --ref "$SYNC_BRANCH"' in step["run"], (
        "台帳ブランチに対する dispatch が消えると、台帳PRは毎回承認待ちで止まる"
    )
    assert _sync_job()["permissions"]["actions"] == "write", "dispatch には actions:write が要る"


def test_dispatch_skips_only_on_previous_dispatch() -> None:
    """二重起動よけが pull_request の run まで数えると永久に起動しなくなる。

    承認待ちで残っている pull_request の run も同じ head_sha を持つため、
    event を絞らずに数えると「既に起動済み」と誤判定して dispatch しない。
    """
    run = _step("Run pr-checks on the sync branch")["run"]

    assert "event=workflow_dispatch" in run, (
        "head_sha だけで数えると承認待ちの pull_request run を拾って永久に起動しない"
    )


def test_dispatch_failure_does_not_fail_the_job() -> None:
    """起動できなくても台帳自体はPRに入っている。ここで赤にすると滞留検知と二重に鳴る。"""
    run = _step("Run pr-checks on the sync branch")["run"]

    assert "::warning::" in run
    assert "::error::" not in run, "起動失敗は警告どまりにする（滞留検知が本来の赤）"


def test_required_checks_exist_in_pr_checks() -> None:
    """dispatch で埋めたい必須チェックが pr-checks.yml 側に実在すること。"""
    jobs = _load(PR_CHECKS)["jobs"]

    missing = [name for name in REQUIRED_CHECKS if name not in jobs]
    assert not missing, f"branch protection の必須チェックが pr-checks.yml に無い: {missing}"


def test_stall_detection_still_guards_the_dispatch_path() -> None:
    """dispatch が効かなかった場合の最後の砦（REV-366）が残っていること。"""
    step = _step("Fail when the ledger PR is stalled")

    assert step.get("if") == "always()"
    assert "::error::" in step["run"], "滞留は赤にする"
