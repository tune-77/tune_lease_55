"""deploy.yml が本番のfail-closed認証設定から乖離するのを防ぐ回帰テスト。

scripts/deploy_cloud_run_api.sh / deploy_cloud_run_web.sh は「インターネット公開する
Cloud RunのAPIはREQUIRE_API_ACCESS_KEYによるapp層キー検証必須」という方針だが、
GitHub Actions の自動デプロイ(.github/workflows/deploy.yml) は別実装として重複して
おり、この方針が反映されないまま api/** や frontend/** への push のたびに未認証
状態へ戻していた（2026-09のCloud Runコスト急増調査で判明。REQUIRE_API_ACCESS_KEY/
API_ACCESS_KEY secretが無く、Webも --allow-unauthenticated のままだった）。

なお Web境界は常に --no-allow-unauthenticated --invoker-iam-check（IAM認証必須）で
運用する。2026-09-08、公開URLが直接ブラウザから繋がらなくなる実害を理由に一時
--allow-unauthenticated へ変更したが（PR #983）、frontend/src/proxy.ts はWeb境界の
IAM有無に関係なく全ての未認証 /api/* リクエストへ特権的な API_ACCESS_KEY を代理
付与する実装のため、Webを公開にすると誰でもそのプロキシ経由でAPIの鍵付き
エンドポイント（コスト発生するchat・案件削除等）を叩けてしまい、元のコスト急増
インシデントと同種の穴を別URLで再現していた（Codexレビューで指摘）。マージ直後に
本ファイルを含めて巻き戻した。Web単体を安全に公開したい場合は、IAMロックを外す
のではなく proxy.ts 側で未認証キャラーへのキー代理付与自体を止める実装が必要。
"""

from __future__ import annotations

from pathlib import Path

import pytest
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
    assert "source scripts/lib/require_api_access_key_secret.sh" in script
    assert 'require_api_access_key_secret "${PROJECT_ID}" "API"' in script
    assert "|| exit 1" in script, "API_ACCESS_KEY secret が無い時にfail-closedしていない"


def test_web_deploy_always_requires_iam() -> None:
    """Web境界はIAM認証必須（--no-allow-unauthenticated --invoker-iam-check）。

    2026-09-08、公開URLが直接ブラウザから繋がらなくなる実害を理由に一時
    --allow-unauthenticated へ変更したが（PR #983）、frontend/src/proxy.ts は
    Web境界のIAM有無に関係なく全ての未認証 /api/* リクエストへ特権的な
    API_ACCESS_KEY を代理付与する実装のため、Webを公開にすると誰でもそのプロキシ
    経由でAPIの鍵付きエンドポイント（コスト発生するchat・案件削除等）を叩けて
    しまい、元のコスト急増インシデントと同種の穴を別URLで再現していた
    （Codexレビューで指摘、マージ直後に巻き戻し）。
    """
    workflow = _load()
    script = _run_script(workflow["jobs"]["deploy-web"], "Deploy to Cloud Run")

    assert "--allow-unauthenticated" not in script
    assert "--no-allow-unauthenticated" in script
    assert "--invoker-iam-check" in script
    assert "source scripts/lib/require_api_access_key_secret.sh" in script
    assert 'require_api_access_key_secret "${PROJECT_ID}" "Web"' in script
    assert "|| exit 1" in script, "API_ACCESS_KEY secret が無い時にfail-closedしていない"


@pytest.mark.parametrize(
    ("job", "build_step"),
    [("deploy-api", "Build API image"), ("deploy-web", "Build frontend image")],
)
def test_secret_preflight_runs_before_the_expensive_build(job: str, build_step: str) -> None:
    """シークレット検証は17分のビルドより前に置く。

    後ろに置くと、キー未設定や権限不足がビルド完了まで分からず、切り分け1回に
    17分かかる（2026-09-07に実際に何度も空費した）。
    """
    steps = [step.get("name") for step in _load()["jobs"][job]["steps"]]

    assert "Preflight - verify API access key secret" in steps
    assert steps.index("Preflight - verify API access key secret") < steps.index(build_step)


def test_lib_require_api_access_key_secret_is_fail_closed() -> None:
    # deploy.yml と scripts/deploy_cloud_run*.sh の両方が同じ関数を経由することで、
    # 片方だけ認証強化して他方が取り残される乖離（2026-09のコスト急増の原因）を防ぐ。
    lib_script = (
        ROOT / "scripts/lib/require_api_access_key_secret.sh"
    ).read_text(encoding="utf-8")

    assert "gcloud secrets describe API_ACCESS_KEY" in lib_script
    assert "return 1" in lib_script
