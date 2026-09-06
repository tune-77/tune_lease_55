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
    package = scripts / "package_cloud_run_bundle.sh"
    package.write_text("#!/bin/sh\nexit 0\n")
    package.chmod(0o755)
    (scripts / "check_cloudrun_demo_readiness.py").write_text("")
    gcloud = tmp_path / "gcloud"
    gcloud.write_text('''#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
if a[:3] == ["secrets", "describe", "API_ACCESS_KEY"]:
    sys.exit(0 if os.environ["HAS_KEY"] == "1" else 1)
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
    def run(script, mode, key=True):
        log.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(scripts / script)], env={**os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}", "PROJECT_ID": "test-project",
            "SHORT_SHA": "test", "CLOUDRUN_DATA_MODE": mode, "API_MODE": mode,
            "HAS_KEY": "1" if key else "0", "DEPLOY_ARGS": str(log),
        }, capture_output=True, text=True, timeout=20)
        return result, json.loads(log.read_text()) if log.exists() else []
    return run


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
@pytest.mark.parametrize("mode", ["production", "demo"])
def test_web_boundary_always_requires_iam(deploy, script, mode):
    result, args = deploy(script, mode)
    assert result.returncode == 0, result.stderr
    assert "--allow-unauthenticated" not in args
    assert "--no-allow-unauthenticated" in args
    assert "--invoker-iam-check" in args
    assert "API_ACCESS_KEY=API_ACCESS_KEY:latest" in args
    if script == "deploy_cloud_run.sh":
        env = args[args.index("--set-env-vars") + 1]
        assert "REQUIRE_API_ACCESS_KEY=1" in env


@pytest.mark.parametrize("script", ["deploy_cloud_run.sh", "deploy_cloud_run_web.sh"])
@pytest.mark.parametrize("mode", ["production", "demo"])
def test_missing_key_never_deploys(deploy, script, mode):
    result, args = deploy(script, mode, key=False)
    assert result.returncode != 0
    assert not args


def test_unknown_api_mode_keeps_web_private(deploy):
    result, args = deploy("deploy_cloud_run_web.sh", "")
    assert result.returncode == 0, result.stderr
    assert "--no-allow-unauthenticated" in args


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
