"""Execute deploy scripts with fake gcloud; never contact or mutate Cloud Run."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def deploy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("deploy_cloud_run.sh", "deploy_cloud_run_web.sh"):
        (scripts / name).write_text((root / "scripts" / name).read_text())
    lib_dir = scripts / "lib"
    lib_dir.mkdir()
    for lib_name in ("require_api_access_key_secret.sh", "require_public_tunnel_auth_secret.sh"):
        (lib_dir / lib_name).write_text((root / "scripts" / "lib" / lib_name).read_text())
    package = scripts / "package_cloud_run_bundle.sh"
    package.write_text("#!/bin/sh\nexit 0\n")
    package.chmod(0o755)
    (scripts / "check_cloudrun_demo_readiness.py").write_text("")
    gcloud = tmp_path / "gcloud"
    # SECRET_STATE は describe の実挙動を再現する:
    #   ok      … シークレットあり
    #   missing … 本当に存在しない (NOT_FOUND)
    #   denied  … 存在するが describe 権限が無い (PERMISSION_DENIED)
    # 旧実装は stderr を捨てて missing と denied を同じ「not found」に潰しており、
    # それが2026-09-07のIAM当て推量ループの原因だった。
    gcloud.write_text('''#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
def describe_result(secret_name, state):
    if state == "ok":
        sys.exit(0)
    if state == "missing":
        sys.stderr.write(f"ERROR: (gcloud.secrets.describe) NOT_FOUND: Secret [projects/test-project/secrets/{secret_name}] not found.\\n")
        sys.exit(1)
    sys.stderr.write("ERROR: (gcloud.secrets.describe) PERMISSION_DENIED: Permission 'secretmanager.secrets.get' denied for resource.\\n")
    sys.exit(1)
if a[:3] == ["secrets", "describe", "API_ACCESS_KEY"]:
    describe_result("API_ACCESS_KEY", os.environ["SECRET_STATE"])
if a[:3] == ["secrets", "describe", "PUBLIC_TUNNEL_AUTH"]:
    describe_result("PUBLIC_TUNNEL_AUTH", os.environ.get("TUNNEL_SECRET_STATE", "ok"))
if a[:3] == ["run", "services", "describe"]:
    if "--format=json" in a:
        mode = os.environ["API_MODE"]
        env = [{"name":"CLOUDRUN_DATA_MODE", "value":mode}] if mode else []
        print(json.dumps({"spec":{"template":{"spec":{"containers":[{"env":env}]}}}}))
    else:
        print("https://test.invalid")
if a[:2] == ["run", "deploy"]:
    open(os.environ["DEPLOY_ARGS"], "w").write(json.dumps(a))
''')
    gcloud.chmod(0o755)
    log = tmp_path / "args.json"
    def run(script, mode, secret="ok", tunnel_secret="ok"):
        log.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(scripts / script)], env={**os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}", "PROJECT_ID": "test-project",
            "SHORT_SHA": "test", "CLOUDRUN_DATA_MODE": mode, "API_MODE": mode,
            "SECRET_STATE": secret, "TUNNEL_SECRET_STATE": tunnel_secret, "DEPLOY_ARGS": str(log),
        }, capture_output=True, text=True, timeout=20)
        return result, json.loads(log.read_text()) if log.exists() else []
    return run


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
@pytest.mark.parametrize("mode", ["production", "demo"])
def test_web_boundary_is_public_behind_tunnel_auth_gate(deploy, script, mode):
    """Web境界は公開（--allow-unauthenticated）だが、PUBLIC_TUNNEL=1 と
    PUBLIC_TUNNEL_AUTH を必ずセットで配線する。

    2026-09-08、単純にIAMを外しただけの時（PR #983）、frontend/src/proxy.ts が
    Web境界のIAM有無に関係なく全ての未認証 /api/* リクエストへ特権的な
    API_ACCESS_KEY を代理付与する実装のため、Webを公開にすると誰でもそのプロキシ
    経由でAPIの鍵付きエンドポイント（コスト発生するchat・案件削除等）を叩けてしまい、
    元のコスト急増インシデントと同種の穴を別URLで再現していた（Codexレビューで指摘）。
    PUBLIC_TUNNEL=1 と PUBLIC_TUNNEL_AUTH がセットで揃っている時、proxy.ts は
    Basic認証（ID: lease）を通らない限り /api/* を含む全パスを401で拒否するため、
    Webを公開にしても未認証の第三者はAPIへ到達できない。
    """
    result, args = deploy(script, mode)
    assert result.returncode == 0, result.stderr
    assert "--allow-unauthenticated" in args
    assert "--no-allow-unauthenticated" not in args
    assert "--invoker-iam-check" not in args

    secrets_indices = [i for i, v in enumerate(args) if v == "--set-secrets"]
    last_secrets_value = args[secrets_indices[-1] + 1]
    assert "API_ACCESS_KEY=API_ACCESS_KEY:latest" in last_secrets_value
    assert "PUBLIC_TUNNEL_AUTH=PUBLIC_TUNNEL_AUTH:latest" in last_secrets_value

    env = args[args.index("--set-env-vars") + 1]
    assert "PUBLIC_TUNNEL=1" in env
    if script == "deploy_cloud_run.sh":
        assert "REQUIRE_API_ACCESS_KEY=1" in env


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
@pytest.mark.parametrize("mode", ["production", "demo"])
def test_missing_tunnel_auth_secret_never_deploys(deploy, script, mode):
    """PUBLIC_TUNNEL_AUTHが無いままの公開デプロイは、Basic認証ゲートが機能せず
    危険なので許さない（API_ACCESS_KEYと同じfail-closed方針）。"""
    result, args = deploy(script, mode, tunnel_secret="missing")
    assert result.returncode != 0
    assert not args
    assert "NOT_FOUND" in result.stderr


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
@pytest.mark.parametrize("mode", ["production", "demo"])
def test_missing_key_never_deploys(deploy, script, mode):
    result, args = deploy(script, mode, secret="missing")
    assert result.returncode != 0
    assert not args


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
def test_unverifiable_key_still_deploys_with_the_key_wired(deploy, script):
    """事前チェックの権限(secretmanager.secrets.get)が無いだけの時は止めない。

    ここで止めると「本来デプロイできるのに事前チェックだけが落ちて永久に進めない」
    状態になる（2026-09-07にCIが実際にこれで停止した）。キー無しデプロイを防ぐ実体は
    --set-secrets 側にあるので、配線が残っていれば fail-closed は維持される。
    """
    result, args = deploy(script, "production", secret="denied")

    assert result.returncode == 0, result.stderr
    assert any("API_ACCESS_KEY=API_ACCESS_KEY:latest" in a for a in args)
    assert "PERMISSION_DENIED" in result.stderr, "本当のgcloudエラーが出ていない"


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
def test_unverifiable_tunnel_auth_secret_still_deploys_with_it_wired(deploy, script):
    """PUBLIC_TUNNEL_AUTHも同じ理由(secretmanager.secrets.get権限が無いだけ)では止めない。"""
    result, args = deploy(script, "production", tunnel_secret="denied")

    assert result.returncode == 0, result.stderr
    assert any("PUBLIC_TUNNEL_AUTH=PUBLIC_TUNNEL_AUTH:latest" in a for a in args)
    assert "PERMISSION_DENIED" in result.stderr, "本当のgcloudエラーが出ていない"


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
def test_real_gcloud_error_is_surfaced_when_secret_is_missing(deploy, script):
    """原因が「潰れた」メッセージにならず、gcloudの実エラーが残ること。"""
    result, _ = deploy(script, "production", secret="missing")

    assert "NOT_FOUND" in result.stderr


def test_public_tunnel_requires_web_auth_and_same_origin_api_proxy():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "run_next_stable.sh").read_text()
    proxy = (root / "frontend/src/proxy.ts").read_text()
    api_client = (root / "frontend/src/lib/api.ts").read_text()

    installer = (root / "scripts/install_next_launchagent.sh").read_text()
    judgment_drill = (root / "frontend/src/app/api/judgment-drill/route.ts").read_text()

    assert "PUBLIC_TUNNEL_AUTH_FILE" in launcher
    assert 'chmod 600 "$AUTH_FILE"' in installer
    assert "EnvironmentVariables.PUBLIC_TUNNEL_AUTH_FILE" in installer
    assert "process.env.PUBLIC_TUNNEL_AUTH" in proxy
    assert 'matcher: "/:path*"' in proxy
    assert '=== "/api/system/knowledge-sync-health"' in proxy
    assert "&& !isPublicKnowledgeSyncProbe" in proxy
    assert 'return "http://127.0.0.1:8000"' not in api_client
    assert judgment_drill.count("internalApiAuthHeaders()") == 3


def test_smart_web_check_retries_with_identity_without_printing_token(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/smart_deploy.sh").read_text()
    function = source[source.index("check_web() {"):source.index("DEPLOYED_API_SHA=")]
    gcloud = tmp_path / "gcloud"
    gcloud.write_text("#!/bin/sh\nprintf '%s' 'test-identity-token'\n")
    gcloud.chmod(0o755)
    curl = tmp_path / "curl"
    curl.write_text('''#!/usr/bin/env python3
import sys
a=sys.argv[1:]
if "--header" not in a:
    print("403", end="")
else:
    value=open(a[a.index("--header")+1][1:]).read()
    assert value == "Authorization: Bearer test-identity-token\\n"
    print("200", end="")
''')
    curl.chmod(0o755)
    result = subprocess.run(["bash", "-c", function + '\ncheck_web "https://test.invalid"'],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "CHECK_TIMEOUT": "1"},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "HTTP 200" in result.stdout
    assert "test-identity-token" not in result.stdout + result.stderr
