"""Cloud Runの積み上がり課金（リビジョン蓄積・demo_warm.shのつけっぱなし）を
定期検知する .github/workflows/cloudrun-cost-guard.yml の配線テスト。

2026-09のCloud Runコスト急増インシデント（PR #977）の振り返りで挙げた
残り2件の改善ポイントに対応する（PR #978フォローアップ）。
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/cloudrun-cost-guard.yml"


def _load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML は YAML 1.1 として `on:` を True と解釈する
    return workflow.get("on", workflow.get(True, {}))


def _run_script(job: dict, step_name: str) -> str:
    for step in job["steps"]:
        if step.get("name") == step_name:
            return step["run"]
    raise AssertionError(f"step not found: {step_name}")


def test_workflow_runs_on_a_schedule() -> None:
    triggers = _triggers(_load())

    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers


def test_workflow_skips_when_wif_not_configured() -> None:
    workflow = _load()

    assert workflow["jobs"]["cost-guard"]["needs"] == "check-wif"
    assert workflow["jobs"]["cost-guard"]["if"] == "needs.check-wif.outputs.wif == 'true'"


def test_workflow_deletes_stale_revisions() -> None:
    workflow = _load()
    script = _run_script(workflow["jobs"]["cost-guard"], "Delete stale 0%-traffic Cloud Run revisions")

    assert "scripts/cleanup_cloud_run_revisions.py" in script
    assert "--apply" in script
    assert "tune-lease-55-api" in script
    assert "tune-lease-55-web" in script


def test_workflow_fails_when_min_instances_left_nonzero() -> None:
    workflow = _load()
    script = _run_script(workflow["jobs"]["cost-guard"], "Warn if min-instances is left non-zero")

    assert "autoscaling.knative.dev/minScale" in script
    assert "exit 1" in script
    assert "demo_warm.sh off" in script
