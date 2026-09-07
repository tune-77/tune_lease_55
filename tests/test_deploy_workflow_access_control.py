"""deploy.yml が本番のfail-closed認証設定から乖離するのを防ぐ回帰テスト。

scripts/deploy_cloud_run_api.sh / deploy_cloud_run_web.sh は「インターネット公開する
Cloud RunはAPIキー・IAM認証必須」という方針だが、GitHub Actions の自動デプロイ
(.github/workflows/deploy.yml) は別実装として重複しており、この方針が反映されない
まま api/** や frontend/** への push のたびに未認証状態へ戻していた
（2026-09のCloud Runコスト急増調査で判明。REQUIRE_API_ACCESS_KEY/API_ACCESS_KEY
secretが無く、Webも --allow-unauthenticated のままだった）。
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = ROOT / ".github/workflows/deploy.yml"


def _load() -> dict:
    return yaml.safe_load(DEPLOY_WORKFLOW.read_text(encoding="utf-8"))


def _run_script(job: dict, step_name: str) -> str:
    for step in job["steps"]:
        if step.get("name") == step_name:
            return step["run"]
    raise AssertionError(f"step not found: {step_name}")


def test_api_deploy_requires_access_key_fail_closed() -> None:
    workflow = _load()
    script = _run_script(workflow["jobs"]["deploy-api"], "Deploy to Cloud Run")

    assert "REQUIRE_API_ACCESS_KEY=1" in script
    assert "API_ACCESS_KEY=API_ACCESS_KEY:latest" in script
    assert "secrets describe API_ACCESS_KEY" in script
    assert "exit 1" in script, "API_ACCESS_KEY secret が無い時にfail-closedしていない"


def test_web_deploy_requires_iam_auth() -> None:
    workflow = _load()
    script = _run_script(workflow["jobs"]["deploy-web"], "Deploy to Cloud Run")

    assert "--allow-unauthenticated" not in script
    assert "--no-allow-unauthenticated" in script
    assert "--invoker-iam-check" in script
    assert "API_ACCESS_KEY=API_ACCESS_KEY:latest" in script
    assert "secrets describe API_ACCESS_KEY" in script
    assert "exit 1" in script, "API_ACCESS_KEY secret が無い時にfail-closedしていない"
