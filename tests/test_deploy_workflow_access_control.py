"""deploy.yml が本番のfail-closed認証設定から乖離するのを防ぐ回帰テスト。

scripts/deploy_cloud_run_api.sh / deploy_cloud_run_web.sh は「インターネット公開する
Cloud RunのAPIはREQUIRE_API_ACCESS_KEYによるapp層キー検証必須」という方針だが、
GitHub Actions の自動デプロイ(.github/workflows/deploy.yml) は別実装として重複して
おり、この方針が反映されないまま api/** や frontend/** への push のたびに未認証
状態へ戻していた（2026-09のCloud Runコスト急増調査で判明。REQUIRE_API_ACCESS_KEY/
API_ACCESS_KEY secretが無く、Webも --allow-unauthenticated のままだった）。

なお Web境界は公開（--allow-unauthenticated）で運用するが、単純にIAMを外すだけ
では frontend/src/proxy.ts が Web境界のIAM有無に関係なく全ての未認証 /api/*
リクエストへ特権的な API_ACCESS_KEY を代理付与するため、Webを公開にすると誰でも
そのプロキシ経由でAPIの鍵付きエンドポイント（コスト発生するchat・案件削除等）を
叩けてしまい、元のコスト急増インシデントと同種の穴を別URLで再現する
（2026-09-08、PR #983 で実際に発生。Codexレビューで指摘）。そのため
PUBLIC_TUNNEL=1 と PUBLIC_TUNNEL_AUTH（Secret Manager）を必ずセットで配線する。
proxy.ts はこの2つが揃っている時、Basic認証（ID: lease）を通らない限り /api/* を
含む全パスを401で拒否するため、Webを公開にしても未認証の第三者はAPIへ到達
できない。ブラウザで直接開く場合は `https://lease:<パスワード>@<URL>` の形で
ブックマークすれば、以後パスワード入力なしで開ける。
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


def test_web_deploy_is_public_behind_tunnel_auth_gate() -> None:
    """Web境界は公開（--allow-unauthenticated）だが、PUBLIC_TUNNEL=1 と
    PUBLIC_TUNNEL_AUTH を必ずセットで配線し、proxy.tsのBasic認証ゲートで守る。

    単純にIAMを外しただけの時（2026-09-08、PR #983）、frontend/src/proxy.ts が
    Web境界のIAM有無に関係なく全ての未認証 /api/* リクエストへ特権的な
    API_ACCESS_KEY を代理付与する実装のため、Webを公開にすると誰でもそのプロキシ
    経由でAPIの鍵付きエンドポイント（コスト発生するchat・案件削除等）を叩けて
    しまい、元のコスト急増インシデントと同種の穴を別URLで再現していた
    （Codexレビューで指摘）。
    """
    workflow = _load()
    script = _run_script(workflow["jobs"]["deploy-web"], "Deploy to Cloud Run")

    assert "--allow-unauthenticated" in script
    assert "--no-allow-unauthenticated" not in script
    assert "--invoker-iam-check" not in script
    assert "PUBLIC_TUNNEL=1" in script
    assert "source scripts/lib/require_api_access_key_secret.sh" in script
    assert "source scripts/lib/require_public_tunnel_auth_secret.sh" in script
    assert 'require_api_access_key_secret "${PROJECT_ID}" "Web"' in script
    assert 'require_public_tunnel_auth_secret "${PROJECT_ID}" "Web"' in script
    assert "|| exit 1" in script, "secretが無い時にfail-closedしていない"
    assert "PUBLIC_TUNNEL_AUTH=" in script


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


def test_web_preflight_also_verifies_tunnel_auth_secret() -> None:
    """WebのPreflightはAPI_ACCESS_KEYだけでなくPUBLIC_TUNNEL_AUTHも検証する。

    ここが無いと、Basic認証ゲートのパスワードが無いままWebが公開されてしまい
    （proxy.tsは全リクエストを401で拒否するため使い物にならない）、17分のビルド
    後に気づくことになる。
    """
    workflow = _load()
    script = _run_script(workflow["jobs"]["deploy-web"], "Preflight - verify API access key secret")

    assert "source scripts/lib/require_public_tunnel_auth_secret.sh" in script
    assert 'require_public_tunnel_auth_secret "${PROJECT_ID}" "Web"' in script


def test_lib_require_api_access_key_secret_is_fail_closed() -> None:
    # deploy.yml と scripts/deploy_cloud_run*.sh の両方が同じ関数を経由することで、
    # 片方だけ認証強化して他方が取り残される乖離（2026-09のコスト急増の原因）を防ぐ。
    lib_script = (
        ROOT / "scripts/lib/require_api_access_key_secret.sh"
    ).read_text(encoding="utf-8")

    assert "gcloud secrets describe API_ACCESS_KEY" in lib_script
    assert "return 1" in lib_script


def test_lib_require_public_tunnel_auth_secret_is_fail_closed() -> None:
    lib_script = (
        ROOT / "scripts/lib/require_public_tunnel_auth_secret.sh"
    ).read_text(encoding="utf-8")

    assert "gcloud secrets describe PUBLIC_TUNNEL_AUTH" in lib_script
    assert "return 1" in lib_script
