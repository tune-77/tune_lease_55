"""scripts/cloudflare_edge_setup.py を偽の Cloudflare API で検証する（外部通信なし）。"""
import shutil
import subprocess
from pathlib import Path

import pytest

import scripts.cloudflare_edge_setup as mod

ACC = "acc1"
ZONE = "zone1"


class FakeApi:
    """Cloudflare API の最小限の状態を持つ偽物。GET は状態から返し、書き込みは記録して反映する。"""

    def __init__(self, *, org=True):
        self.calls = []
        self.org = {"auth_domain": "team.cloudflareaccess.com"} if org else None
        self.idps, self.tokens, self.policies, self.apps, self.routes = [], [], [], [], []
        self.rules = None  # None = entrypoint 未作成(404)
        self.worker = None
        self.worker_subdomain = {"enabled": False, "previews_enabled": False}
        self._seq = 0

    def _new_id(self, prefix):
        self._seq += 1
        return f"{prefix}{self._seq}"

    def get(self, path):
        if path.startswith("/zones?"):
            return [{"id": ZONE, "account": {"id": ACC}, "plan": {"name": "Free Website"}}]
        if self.org is None and "/access/" in path:
            raise mod.CloudflareError(403, path, [{"message": "access.api.error.not_enabled: Access is not enabled."}])
        if path.endswith("/access/organizations"):
            return self.org
        table = {
            "/access/identity_providers": self.idps,
            "/access/service_tokens": self.tokens,
            "/access/policies": self.policies,
            "/access/apps": self.apps,
            "/workers/routes": self.routes,
        }
        for suffix, value in table.items():
            if path.endswith(suffix):
                return value
        if path.endswith("/http_ratelimit/entrypoint"):
            if self.rules is None:
                raise mod.CloudflareError(404, path, "")
            return {"rules": self.rules}
        if path.endswith(f"/workers/scripts/{mod.WORKER_NAME}/subdomain"):
            return dict(self.worker_subdomain)
        raise AssertionError(f"unexpected GET {path}")

    def get_text(self, path):
        # 実APIと同じく multipart で包んで返す
        if self.worker is None:
            return None
        return f"--b\r\nContent-Disposition: form-data; name=\"worker.mjs\"\r\n\r\n{self.worker}\r\n--b--\r\n"

    def call(self, method, path, body=None, *, data=None, content_type="application/json"):
        self.calls.append((method, path, body))
        if method == "POST" and path.endswith("/access/identity_providers"):
            item = {**body, "id": self._new_id("idp")}
            self.idps.append(item)
            return item
        if method == "POST" and path.endswith("/access/service_tokens"):
            item = {"name": body["name"], "id": self._new_id("tok"), "client_id": "cid.access"}
            self.tokens.append(item)
            return {**item, "client_secret": "s3cret-value"}
        if method == "POST" and path.endswith("/access/policies"):
            item = {**body, "id": self._new_id("pol")}
            self.policies.append(item)
            return item
        if method == "POST" and path.endswith("/access/apps"):
            item = {**body, "id": self._new_id("app")}
            self.apps.append(item)
            return item
        if method == "PUT" and path.endswith("/http_ratelimit/entrypoint"):
            self.rules = body["rules"]
            return {"rules": self.rules}
        if method == "PUT" and "/workers/scripts/" in path:
            self.worker = mod.worker_source()
            return {}
        if method == "POST" and path.endswith("/subdomain"):
            self.worker_subdomain = dict(body)
            return {}
        if method == "POST" and path.endswith("/workers/routes"):
            self.routes.append({**body, "id": self._new_id("route")})
            return {}
        raise AssertionError(f"unexpected {method} {path}")


@pytest.fixture
def keychain(monkeypatch):
    store = {}
    monkeypatch.setattr(mod, "keychain_has", lambda service: service in store)
    monkeypatch.setattr(mod, "keychain_store", lambda service, secret: store.__setitem__(service, secret))
    return store


def _apply(plan):
    for change in plan.changes:
        if change.kind in ("create", "update"):
            change.apply()


def test_dry_run_plans_everything_without_writing(keychain):
    api = FakeApi()
    plan = mod.build_plan(api)

    kinds = {c.resource: c.kind for c in plan.changes}
    assert kinds["ログイン方法: メールのワンタイムPIN"] == "create"
    assert kinds[f"Access アプリ {mod.HOSTNAME}"] == "create"
    assert kinds["rate limiting ルール shion_costly_paths"] == "create"
    assert kinds[f"Worker {mod.WORKER_NAME}"] == "create"
    assert kinds[f"Worker ルート {mod.WORKER_ROUTE}"] == "create"
    assert api.calls == []  # ドライランでは何も書き込まない
    rendered = "\n".join(c.render() for c in plan.changes)
    assert mod.OWNER_EMAIL in rendered and "730h" in rendered


def test_apply_then_rerun_is_idempotent(keychain):
    api = FakeApi()
    api.rules = [{"ref": "other", "expression": "true", "action": "block", "id": "r0"}]
    _apply(mod.build_plan(api))

    app = api.apps[0]
    assert app["allowed_idps"] == [api.idps[0]["id"]]
    assert [p["id"] for p in app["policies"]] == [p["id"] for p in api.policies]
    owner, monitor = api.policies
    assert owner["include"] == [{"email": {"email": mod.OWNER_EMAIL}}] and owner["decision"] == "allow"
    assert monitor["include"] == [{"service_token": {"token_id": api.tokens[0]["id"]}}]
    assert monitor["decision"] == "non_identity"
    assert keychain == {mod.MONITOR_ID_SERVICE: "cid.access", mod.MONITOR_SECRET_SERVICE: "s3cret-value"}
    assert [r["ref"] for r in api.rules] == ["other", "shion_costly_paths"]  # 既存ルールは維持
    assert api.rules[1]["ratelimit"]["period"] == 10

    # 現行の API は app の policies に precedence を付けて返す
    rerun = mod.build_plan(api)
    assert {c.kind for c in rerun.changes} == {"ok"}, [c.render() for c in rerun.changes if c.kind != "ok"]


def test_worker_module_comparison_rejects_wrapped_source():
    api = FakeApi()
    api.worker = "// unexpected prefix\n" + mod.worker_source() + "\n// unexpected suffix"
    plan = mod.Plan()

    mod.plan_worker(api, ACC, plan)

    [change] = plan.changes
    assert change.kind == "update"
    assert change.resource == f"Worker {mod.WORKER_NAME}"


def test_worker_subdomain_is_reconciled_when_source_matches():
    api = FakeApi()
    api.worker = mod.worker_source()
    api.worker_subdomain = {"enabled": True, "previews_enabled": True}
    plan = mod.Plan()

    mod.plan_worker(api, ACC, plan)

    [change] = plan.changes
    assert change.kind == "update"
    assert change.resource == f"Worker {mod.WORKER_NAME} workers.dev"
    change.apply()
    assert api.worker_subdomain == {"enabled": False, "previews_enabled": False}
    assert not any(method == "PUT" for method, _path, _body in api.calls)


def test_worker_multipart_parser_returns_exact_module():
    source = "export default { fetch() { return new Response('ok'); } };\n"
    payload = (
        "--boundary\r\n"
        'Content-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n'
        "Content-Type: application/json\r\n\r\n{}\r\n"
        "--boundary\r\n"
        'Content-Disposition: form-data; name="worker.mjs"; filename="worker.mjs"\r\n'
        "Content-Type: application/javascript+module\r\n\r\n"
        f"{source}\r\n--boundary--\r\n"
    )

    assert mod.worker_module_from_multipart(payload) == source


def test_missing_service_token_secret_is_rotated(keychain, monkeypatch):
    api = FakeApi()
    _apply(mod.build_plan(api))
    keychain.clear()
    rotated = []

    original_call = api.call

    def call(method, path, body=None, **kw):
        if path.endswith("/rotate"):
            rotated.append(path)
            return {"client_secret": "rotated-secret"}
        return original_call(method, path, body, **kw)

    api.call = call
    plan = mod.build_plan(api)
    token_change = next(c for c in plan.changes if c.resource.startswith("service token"))
    assert token_change.kind == "update"
    _apply(plan)
    assert rotated and keychain[mod.MONITOR_SECRET_SERVICE] == "rotated-secret"


def test_uninitialized_zero_trust_is_reported_as_manual_step(keychain):
    api = FakeApi(org=False)
    plan = mod.build_plan(api)
    manual = [c for c in plan.changes if c.kind == "manual"]
    assert len(manual) == 1 and "未有効化" in manual[0].note
    assert not any(c.resource.startswith("Access アプリ") for c in plan.changes)
    assert any(c.resource.startswith("rate limiting") for c in plan.changes)  # 他は進める


def test_main_never_prints_token(keychain, monkeypatch, capsys):
    secret_token = "cf-token-must-not-leak-123"
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", secret_token)
    monkeypatch.setattr(mod, "Api", lambda token: FakeApi())
    assert mod.main([]) == 0
    out = capsys.readouterr()
    assert secret_token not in out.out + out.err
    assert "ドライラン" in out.out


def test_main_without_token_exits(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    assert mod.main([]) == 2


def test_real_api_repr_hides_token():
    assert "abc" not in repr(mod.Api("abc"))


def test_rate_limit_rule_fits_free_plan():
    rule = mod.rate_limit_rule()
    assert rule["ratelimit"]["period"] == 10 and rule["ratelimit"]["mitigation_timeout"] == 10
    assert rule["ratelimit"]["characteristics"] == ["cf.colo.id", "ip.src"]
    assert '"/api/shion/voice/session"' in rule["expression"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_sleep_worker_node_tests():
    worker_test = Path(__file__).resolve().parents[1] / "cloudflare" / "shion-sleep-worker" / "worker.test.mjs"
    result = subprocess.run(["node", "--test", str(worker_test)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_access_permission_error_is_distinguished_from_not_enabled(keychain):
    api = FakeApi()

    def denied(path):
        if "/access/" in path:
            raise mod.CloudflareError(403, path, [{"code": 10000, "message": "Authentication error"}])
        return FakeApi.get(api, path)

    api.get = denied
    manual = [c for c in mod.build_plan(api).changes if c.kind == "manual"]
    assert "権限不足" in manual[0].note
