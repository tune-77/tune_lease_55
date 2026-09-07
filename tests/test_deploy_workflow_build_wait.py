"""deploy.yml のビルド待機方式が、ログバケット読み取りに依存する
`gcloud builds submit`(非async)の同期待ちに戻らないようにする回帰テスト。

WIF(Workload Identity Federation)認証のCIランナーからだと、対象サービス
アカウントに roles/storage.admin を持たせていても
"This tool can only stream logs if you are Viewer/Owner of the project" で
gcloud builds submit の完了待ちが失敗し続けた（2026-09、.github/workflows/
deploy.yml の自動デプロイインシデント振り返り）。
scripts/lib/submit_cloud_build_and_wait.sh が --async + gcloud builds describe
による代替実装。
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deploy.yml"
LIB_SCRIPT = (ROOT / "scripts/lib/submit_cloud_build_and_wait.sh").read_text(encoding="utf-8")


def _load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _run_script(job: dict, step_name: str) -> str:
    for step in job["steps"]:
        if step.get("name") == step_name:
            return step["run"]
    raise AssertionError(f"step not found: {step_name}")


def test_lib_submits_build_async_and_polls_describe() -> None:
    assert "--async" in LIB_SCRIPT
    assert "gcloud builds describe" in LIB_SCRIPT
    assert "gcloud builds submit" in LIB_SCRIPT
    assert '"$build_id"' in LIB_SCRIPT


def test_api_build_step_uses_shared_wait_helper() -> None:
    workflow = _load()
    script = _run_script(workflow["jobs"]["deploy-api"], "Build API image")

    assert "source scripts/lib/submit_cloud_build_and_wait.sh" in script
    assert "submit_cloud_build_and_wait" in script
    assert "--suppress-logs" not in script


def test_web_build_step_uses_shared_wait_helper() -> None:
    workflow = _load()
    script = _run_script(workflow["jobs"]["deploy-web"], "Build frontend image")

    assert "source scripts/lib/submit_cloud_build_and_wait.sh" in script
    assert "submit_cloud_build_and_wait" in script
    assert "--suppress-logs" not in script
