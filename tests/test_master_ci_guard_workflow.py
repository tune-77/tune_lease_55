"""master が未検査のまま進むのを防ぐ CI ガードの配線テスト。

GitHub は GITHUB_TOKEN 起因の push / pull_request では再帰防止のためワークフローを
起動しない。ledger-sync.yml は GITHUB_TOKEN で台帳PRを作り auto-merge するため、
そのマージコミットは pull_request も push も発火せず pr-checks.yml が一度も走らない
まま master に入っていた（実例: 2320ffd）。

master-ci-guard.yml がその取りこぼしを検知して pr-checks.yml を workflow_dispatch で
起動する。workflow_dispatch は GITHUB_TOKEN からでも発火する唯一の入口なので、
pr-checks.yml 側からこのトリガーが消えるとガードは黙って無力化する。
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PR_CHECKS = ROOT / ".github/workflows/pr-checks.yml"
GUARD = ROOT / ".github/workflows/master-ci-guard.yml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML は YAML 1.1 として `on:` を True と解釈する
    return workflow.get("on", workflow.get(True, {}))


def test_pr_checks_accepts_workflow_dispatch() -> None:
    triggers = _triggers(_load(PR_CHECKS))

    assert "workflow_dispatch" in triggers, (
        "workflow_dispatch が無いと master-ci-guard.yml から起動し直せない"
    )
    assert "push" in triggers and "pull_request" in triggers


def test_pr_checks_warn_only_steps_cover_dispatch() -> None:
    """warn-only 側を push 限定にすると dispatch 実行でだけ strict になって落ちる。"""
    jobs = _load(PR_CHECKS)["jobs"]

    for job_id in ("preflight-guard", "pr-change-risk"):
        conditions = [step.get("if") for step in jobs[job_id]["steps"] if step.get("if")]
        assert "github.event_name == 'pull_request'" in conditions
        assert "github.event_name != 'pull_request'" in conditions


def test_guard_dispatches_pr_checks_on_master() -> None:
    guard = _load(GUARD)
    triggers = _triggers(guard)

    assert "schedule" in triggers, "定期実行が無いと未検査コミットを誰も拾わない"
    assert guard["permissions"]["actions"] == "write", "dispatch には actions:write が要る"

    body = GUARD.read_text(encoding="utf-8")
    assert 'TARGET_WORKFLOW: pr-checks.yml' in body
    assert 'gh workflow run "${TARGET_WORKFLOW}" --ref master' in body
