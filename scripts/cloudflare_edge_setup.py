#!/usr/bin/env python3
"""shion.tune77.com の Cloudflare エッジ保護を冪等に設定する。

1. Cloudflare Access: ホスト全体を保護。ログインはメールのワンタイムPIN、許可は所有者メールのみ、
   セッション約1か月。朝報の死活チェック用に service token も許可する（キーチェーンに保存）。
2. WAF rate limiting: 費用のかかるパス（音声トークン発行・チャット等）への短時間の大量アクセスを遮断。
3. Worker: Mac（オリジン）不達時だけ「紫苑は今お休み中です」を返す（cloudflare/shion-sleep-worker）。

既定はドライラン（差分表示のみ）。--apply で反映、--verify で反映後の確認。
API トークンは環境変数 CLOUDFLARE_API_TOKEN からのみ読み、表示・保存しない。
通常は scripts/cloudflare_edge_setup.sh（キーチェーンから読んで渡す）経由で実行する。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://api.cloudflare.com/client/v4"

ZONE_NAME = "tune77.com"
HOSTNAME = "shion.tune77.com"
OWNER_EMAIL = "tune7575@gmail.com"
SESSION_DURATION = "730h"  # 約1か月

ACCESS_APP_NAME = "shion"
OTP_IDP_NAME = "One-time PIN"
OWNER_POLICY_NAME = "shion-owner-email-otp"
MONITOR_POLICY_NAME = "shion-morning-monitor-service-token"
SERVICE_TOKEN_NAME = "tunelease-morning-monitor"
SERVICE_TOKEN_DURATION = "8760h"

# 朝報（scripts/send_daily_improvement_slack.py）がこの2つを読んで Access を通過する。
KEYCHAIN_ACCOUNT = "tune-lease-55"
MONITOR_ID_SERVICE = "cloudflare-access-monitor-id"
MONITOR_SECRET_SERVICE = "cloudflare-access-monitor-secret"

# 無料プランの rate limiting は 1ルール・期間10秒・遮断10秒・IP+colo 単位。
# アプリ側の音声は1日20回・20秒間隔なので、人の操作では届かず、スクリプト連打だけを止める値にする。
RATE_LIMIT_REF = "shion_costly_paths"
RATE_LIMIT_PATHS = (
    "/api/shion/voice/session",
    "/api/chat",
    "/api/gunshi/chat",
    "/api/gunshi/stream",
    "/api/gunshi/advise",
    "/api/multi-agent-screening",
    "/api/multi-agent-screening/stream",
)
RATE_LIMIT_REQUESTS = 5
RATE_LIMIT_PERIOD = 10
RATE_LIMIT_TIMEOUT = 10

WORKER_NAME = "shion-sleep-page"
WORKER_FILE = REPO_ROOT / "cloudflare" / "shion-sleep-worker" / "worker.mjs"
WORKER_COMPAT_DATE = "2026-09-01"
WORKER_ROUTE = f"{HOSTNAME}/*"


class CloudflareError(RuntimeError):
    def __init__(self, status: int, path: str, errors: Any):
        # トークンやリクエスト本文は含めない。
        super().__init__(f"Cloudflare API {status} {path}: {json.dumps(errors, ensure_ascii=False)[:500]}")
        self.status = status


class Api:
    def __init__(self, token: str, opener: Callable[..., Any] = urllib.request.urlopen):
        self._token = token
        self._opener = opener

    def __repr__(self) -> str:  # トークンを repr に出さない
        return "Api(<token hidden>)"

    def call(self, method: str, path: str, body: Any = None, *, data: bytes | None = None,
             content_type: str = "application/json") -> Any:
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(API_BASE + path, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        if data is not None:
            request.add_header("Content-Type", content_type)
        try:
            with self._opener(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8") or "{}")
            except ValueError:
                payload = {}
            raise CloudflareError(exc.code, path, payload.get("errors") or payload) from None
        if not payload.get("success", True):
            raise CloudflareError(200, path, payload.get("errors"))
        return payload.get("result")

    def get(self, path: str) -> Any:
        return self.call("GET", path)

    def get_text(self, path: str) -> str | None:
        """JSON でない応答（Worker のソース）を返す。404 なら None。"""
        request = urllib.request.Request(API_BASE + path, method="GET")
        request.add_header("Authorization", f"Bearer {self._token}")
        try:
            with self._opener(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise CloudflareError(exc.code, path, "") from None


@dataclass
class Change:
    kind: str  # "create" | "update" | "ok" | "manual"
    resource: str
    before: Any = None
    after: Any = None
    apply: Callable[[], Any] | None = None
    note: str = ""

    def render(self) -> str:
        mark = {"create": "+", "update": "~", "ok": "=", "manual": "!"}[self.kind]
        head = f"{mark} {self.resource}" + (f"  ({self.note})" if self.note else "")
        if self.kind not in ("create", "update"):
            return head
        before = json.dumps(self.before, ensure_ascii=False, indent=2, sort_keys=True).splitlines() if self.before else []
        after = json.dumps(self.after, ensure_ascii=False, indent=2, sort_keys=True).splitlines()
        diff = difflib.unified_diff(before, after, "current", "desired", lineterm="", n=1)
        return head + "\n" + "\n".join("    " + line for line in diff)


@dataclass
class Plan:
    changes: list[Change] = field(default_factory=list)

    def add(self, change: Change) -> Change:
        self.changes.append(change)
        return change


# ── キーチェーン（値は標準出力・引数に出さない） ─────────────────────────


def keychain_has(service: str) -> bool:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", KEYCHAIN_ACCOUNT],
        capture_output=True,
    )
    return result.returncode == 0


def keychain_store(service: str, secret: str) -> None:
    # `security -i` は標準入力からコマンドを読むので、値がプロセス一覧（argv）に出ない。
    if any(ch in secret for ch in "\"\n\r"):
        raise ValueError("unexpected characters in credential")
    command = f'add-generic-password -U -s "{service}" -a "{KEYCHAIN_ACCOUNT}" -w "{secret}"\n'
    result = subprocess.run(["security", "-i"], input=command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"keychain store failed for {service}")


# ── 各リソース ─────────────────────────────────────────────────────────


def _pick(obj: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: obj.get(key) for key in keys} if obj else {}


def _id(resolve: Callable[[], str], placeholder: str) -> str:
    """これから作るリソースの id はドライランでは未確定なので、表示用の仮名を返す。"""
    try:
        return resolve()
    except KeyError:
        return placeholder


def find_zone(api: Api) -> dict[str, Any]:
    zones = api.get("/zones?" + urllib.parse.urlencode({"name": ZONE_NAME}))
    if not zones:
        raise RuntimeError(f"zone {ZONE_NAME} not found (token needs Zone:Read)")
    return zones[0]


def access_ready(api: Api, account_id: str, plan: Plan) -> bool:
    reason = ""
    try:
        org = api.get(f"/accounts/{account_id}/access/organizations")
    except CloudflareError as exc:
        org = None
        if exc.status not in (400, 403, 404):
            raise
        reason = str(exc)
    if not org or not org.get("auth_domain"):
        # 未有効化と権限不足はどちらも 403 なので、Access 側の別APIの応答で見分ける。
        if not reason or "not_enabled" not in reason:
            try:
                api.get(f"/accounts/{account_id}/access/apps")
            except CloudflareError as exc:
                reason = str(exc)
        if "not_enabled" in reason:
            note = "Access 未有効化。ダッシュボードの Zero Trust でチーム名と Free プランを選んでから再実行"
        else:
            note = f"取得できません（トークンの Access 権限不足の可能性）: {reason[:200]}"
        plan.add(Change("manual", "Zero Trust 組織（チーム名）", note=note))
        return False
    plan.add(Change("ok", f"Zero Trust 組織 {org.get('auth_domain')}"))
    return True


def plan_otp_idp(api: Api, account_id: str, plan: Plan) -> Callable[[], str]:
    idps = api.get(f"/accounts/{account_id}/access/identity_providers") or []
    existing = next((idp for idp in idps if idp.get("type") == "onetimepin"), None)
    if existing:
        plan.add(Change("ok", "ログイン方法: メールのワンタイムPIN"))
        return lambda: existing["id"]
    created: dict[str, str] = {}

    def apply() -> None:
        created["id"] = api.call("POST", f"/accounts/{account_id}/access/identity_providers",
                                 {"name": OTP_IDP_NAME, "type": "onetimepin", "config": {}})["id"]

    plan.add(Change("create", "ログイン方法: メールのワンタイムPIN",
                    after={"name": OTP_IDP_NAME, "type": "onetimepin"}, apply=apply))
    return lambda: created["id"]


def plan_service_token(api: Api, account_id: str, plan: Plan) -> Callable[[], str]:
    tokens = api.get(f"/accounts/{account_id}/access/service_tokens") or []
    existing = next((t for t in tokens if t.get("name") == SERVICE_TOKEN_NAME), None)
    stored = keychain_has(MONITOR_ID_SERVICE) and keychain_has(MONITOR_SECRET_SERVICE)
    if existing and stored:
        plan.add(Change("ok", f"service token {SERVICE_TOKEN_NAME}（キーチェーン保存済み）"))
        return lambda: existing["id"]

    result: dict[str, str] = {}

    def store(token: dict[str, Any]) -> None:
        keychain_store(MONITOR_ID_SERVICE, token["client_id"])
        keychain_store(MONITOR_SECRET_SERVICE, token["client_secret"])

    if existing:
        # secret は作成時にしか取得できないため、キーチェーンに無ければローテーションして保存し直す。
        def rotate() -> None:
            token = api.call("POST", f"/accounts/{account_id}/access/service_tokens/{existing['id']}/rotate")
            store({"client_id": existing["client_id"], "client_secret": token["client_secret"]})
            result["id"] = existing["id"]

        plan.add(Change("update", f"service token {SERVICE_TOKEN_NAME}",
                        before={"keychain": "missing"}, after={"keychain": "stored (rotated)"}, apply=rotate))
    else:
        def create() -> None:
            token = api.call("POST", f"/accounts/{account_id}/access/service_tokens",
                             {"name": SERVICE_TOKEN_NAME, "duration": SERVICE_TOKEN_DURATION})
            store(token)
            result["id"] = token["id"]

        plan.add(Change("create", f"service token {SERVICE_TOKEN_NAME}",
                        after={"name": SERVICE_TOKEN_NAME, "duration": SERVICE_TOKEN_DURATION,
                               "keychain": f"{MONITOR_ID_SERVICE} / {MONITOR_SECRET_SERVICE}"},
                        apply=create))
    return lambda: result.get("id") or (existing or {})["id"]


_POLICY_KEYS = ("name", "decision", "include", "session_duration")


def plan_policy(api: Api, account_id: str, plan: Plan, policies: list[dict[str, Any]],
                desired_fn: Callable[[], dict[str, Any]]) -> Callable[[], str]:
    desired = desired_fn()
    name = desired["name"]
    existing = next((p for p in policies if p.get("name") == name), None)
    result: dict[str, str] = {}
    if existing:
        current = _pick(existing, tuple(desired))
        if current == desired:
            plan.add(Change("ok", f"Access ポリシー {name}"))
        else:
            plan.add(Change("update", f"Access ポリシー {name}", before=current, after=desired,
                            apply=lambda: api.call("PUT", f"/accounts/{account_id}/access/policies/{existing['id']}",
                                                   desired_fn())))
        return lambda: existing["id"]

    def create() -> None:
        result["id"] = api.call("POST", f"/accounts/{account_id}/access/policies", desired_fn())["id"]

    plan.add(Change("create", f"Access ポリシー {name}", after=desired, apply=create))
    return lambda: result["id"]


def owner_policy() -> dict[str, Any]:
    return {"name": OWNER_POLICY_NAME, "decision": "allow",
            "include": [{"email": {"email": OWNER_EMAIL}}], "session_duration": SESSION_DURATION}


def monitor_policy(token_id: str) -> dict[str, Any]:
    return {"name": MONITOR_POLICY_NAME, "decision": "non_identity",
            "include": [{"service_token": {"token_id": token_id}}]}


def plan_access_app(api: Api, account_id: str, plan: Plan, idp_id: Callable[[], str],
                    policy_ids: list[Callable[[], str]]) -> None:
    apps = api.get(f"/accounts/{account_id}/access/apps") or []
    existing = next((a for a in apps if a.get("domain") == HOSTNAME), None)

    def desired(*, preview: bool = False) -> dict[str, Any]:
        def resolve(fn: Callable[[], str], placeholder: str) -> str:
            return _id(fn, placeholder) if preview else fn()

        return {
            "name": ACCESS_APP_NAME,
            "domain": HOSTNAME,
            "type": "self_hosted",
            "session_duration": SESSION_DURATION,
            "allowed_idps": [resolve(idp_id, "<new one-time PIN>")],
            "auto_redirect_to_identity": True,
            "app_launcher_visible": False,
            "policies": [{"id": resolve(pid, f"<new policy {i + 1}>"), "precedence": i + 1}
                         for i, pid in enumerate(policy_ids)],
        }

    preview = desired(preview=True)
    if not existing:
        plan.add(Change("create", f"Access アプリ {HOSTNAME}", after=preview,
                        apply=lambda: api.call("POST", f"/accounts/{account_id}/access/apps", desired())))
        return

    current = _pick(existing, ("name", "domain", "type", "session_duration", "allowed_idps",
                               "auto_redirect_to_identity", "app_launcher_visible"))
    current["policies"] = [{"id": p.get("id"), "precedence": p.get("precedence")}
                           for p in sorted(existing.get("policies") or [], key=lambda p: p.get("precedence") or 0)]
    if current == preview:
        plan.add(Change("ok", f"Access アプリ {HOSTNAME}"))
        return
    plan.add(Change("update", f"Access アプリ {HOSTNAME}", before=current, after=preview,
                    apply=lambda: api.call("PUT", f"/accounts/{account_id}/access/apps/{existing['id']}", desired())))


def rate_limit_rule() -> dict[str, Any]:
    paths = " ".join(f'"{p}"' for p in RATE_LIMIT_PATHS)
    return {
        "ref": RATE_LIMIT_REF,
        "description": "shion: 費用のかかるパス（音声トークン・チャット）の連打を止める",
        "expression": f"(http.request.uri.path in {{{paths}}})",
        "action": "block",
        "ratelimit": {
            "characteristics": ["cf.colo.id", "ip.src"],
            "period": RATE_LIMIT_PERIOD,
            "requests_per_period": RATE_LIMIT_REQUESTS,
            "mitigation_timeout": RATE_LIMIT_TIMEOUT,
        },
        "enabled": True,
    }


_RULE_KEYS = ("ref", "description", "expression", "action", "action_parameters", "ratelimit", "enabled")


def plan_rate_limit(api: Api, zone_id: str, plan: Plan) -> None:
    path = f"/zones/{zone_id}/rulesets/phases/http_ratelimit/entrypoint"
    try:
        rules = (api.get(path) or {}).get("rules") or []
    except CloudflareError as exc:
        if exc.status != 404:
            raise
        rules = []
    desired = rate_limit_rule()
    others = [{k: v for k, v in _pick(r, _RULE_KEYS).items() if v is not None}
              for r in rules if r.get("ref") != RATE_LIMIT_REF]
    current = next((r for r in rules if r.get("ref") == RATE_LIMIT_REF), None)
    current_view = {k: v for k, v in _pick(current, _RULE_KEYS).items() if v is not None} if current else None
    if current_view == desired:
        plan.add(Change("ok", "rate limiting ルール " + RATE_LIMIT_REF))
        return
    note = f"他のルール {len(others)} 件は維持" if others else ""
    plan.add(Change("update" if current else "create", "rate limiting ルール " + RATE_LIMIT_REF,
                    before=current_view, after=desired, note=note,
                    apply=lambda: api.call("PUT", path, {"rules": [*others, desired]})))


def worker_source() -> str:
    return WORKER_FILE.read_text(encoding="utf-8")


def upload_worker(api: Api, account_id: str) -> None:
    boundary = uuid.uuid4().hex
    metadata = {"main_module": "worker.mjs", "compatibility_date": WORKER_COMPAT_DATE,
                "observability": {"enabled": False}}
    parts = [
        ("metadata", "metadata.json", "application/json", json.dumps(metadata)),
        ("worker.mjs", "worker.mjs", "application/javascript+module", worker_source()),
    ]
    body = b""
    for name, filename, ctype, content in parts:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n").encode() + content.encode("utf-8") + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    api.call("PUT", f"/accounts/{account_id}/workers/scripts/{WORKER_NAME}", data=body,
             content_type=f"multipart/form-data; boundary={boundary}")
    # *.workers.dev からは呼ばせない（ルート経由だけで動かす）。
    api.call("POST", f"/accounts/{account_id}/workers/scripts/{WORKER_NAME}/subdomain", {"enabled": False})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def plan_worker(api: Api, account_id: str, plan: Plan) -> None:
    local = worker_source()
    remote = api.get_text(f"/accounts/{account_id}/workers/scripts/{WORKER_NAME}/content/v2")
    # Cloudflare はソースを multipart で返すので、手元のソースがそのまま含まれていれば同一とみなす。
    if remote is not None and local in remote:
        plan.add(Change("ok", f"Worker {WORKER_NAME}"))
        return
    remote_hash = _sha(remote) if remote is not None else None
    plan.add(Change("update" if remote_hash else "create", f"Worker {WORKER_NAME}",
                    before={"sha256": remote_hash} if remote_hash else None,
                    after={"sha256": _sha(local), "source": str(WORKER_FILE.relative_to(REPO_ROOT)),
                           "observability": False, "workers_dev": False},
                    apply=lambda: upload_worker(api, account_id)))


def plan_worker_route(api: Api, zone_id: str, plan: Plan) -> None:
    routes = api.get(f"/zones/{zone_id}/workers/routes") or []
    existing = next((r for r in routes if r.get("pattern") == WORKER_ROUTE), None)
    desired = {"pattern": WORKER_ROUTE, "script": WORKER_NAME}
    if existing and existing.get("script") == WORKER_NAME:
        plan.add(Change("ok", f"Worker ルート {WORKER_ROUTE}"))
        return
    if existing:
        plan.add(Change("update", f"Worker ルート {WORKER_ROUTE}", before=_pick(existing, ("pattern", "script")),
                        after=desired,
                        apply=lambda: api.call("PUT", f"/zones/{zone_id}/workers/routes/{existing['id']}", desired)))
        return
    plan.add(Change("create", f"Worker ルート {WORKER_ROUTE}", after=desired,
                    apply=lambda: api.call("POST", f"/zones/{zone_id}/workers/routes", desired)))


def build_plan(api: Api) -> Plan:
    plan = Plan()
    zone = find_zone(api)
    zone_id, account_id = zone["id"], zone["account"]["id"]
    plan.add(Change("ok", f"zone {ZONE_NAME}（plan: {(zone.get('plan') or {}).get('name', '?')}）"))

    if access_ready(api, account_id, plan):
        idp_id = plan_otp_idp(api, account_id, plan)
        token_id = plan_service_token(api, account_id, plan)
        policies = api.get(f"/accounts/{account_id}/access/policies") or []
        owner_id = plan_policy(api, account_id, plan, policies, owner_policy)
        monitor_id = plan_policy(api, account_id, plan, policies,
                                 lambda: monitor_policy(_id(token_id, "<new service token>")))
        plan_access_app(api, account_id, plan, idp_id, [owner_id, monitor_id])

    plan_rate_limit(api, zone_id, plan)
    plan_worker(api, account_id, plan)
    plan_worker_route(api, zone_id, plan)
    return plan


# ── 反映後の確認 ───────────────────────────────────────────────────────


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _probe(headers: dict[str, str]) -> tuple[int, str]:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(f"https://{HOSTNAME}/chat",
                                     headers={"User-Agent": "tunelease-edge-verify/1.0", "Accept": "text/html",
                                              **headers})
    try:
        with opener.open(request, timeout=20) as response:
            return response.getcode(), response.headers.get("Location", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Location", "")


def _keychain_read(service: str) -> str:
    result = subprocess.run(["security", "find-generic-password", "-s", service, "-a", KEYCHAIN_ACCOUNT, "-w"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def verify(api: Api) -> bool:
    ok = True
    status, location = _probe({})
    login = status in (301, 302, 303) and "cloudflareaccess.com" in location
    print(f"(a) 未ログイン → Access ログイン画面: {'OK' if login else 'NG'}（{status}）")
    ok &= login

    zone = find_zone(api)
    rules = (api.get(f"/zones/{zone['id']}/rulesets/phases/http_ratelimit/entrypoint") or {}).get("rules") or []
    rule = next((r for r in rules if r.get("ref") == RATE_LIMIT_REF), None)
    active = bool(rule and rule.get("enabled"))
    print(f"(b) 回数制限ルール有効: {'OK' if active else 'NG'}")
    ok &= active

    client_id, secret = _keychain_read(MONITOR_ID_SERVICE), _keychain_read(MONITOR_SECRET_SERVICE)
    if not (client_id and secret):
        print("(c) 休止ページ: NG（service token がキーチェーンに無い）")
        return False
    auth = {"CF-Access-Client-Id": client_id, "CF-Access-Client-Secret": secret}
    normal, _ = _probe(auth)
    sleeping, _ = _probe({**auth, "x-shion-sleep-test": "1"})
    print(f"(c) service token で通常表示: {'OK' if normal == 200 else 'NG'}（{normal}）／"
          f"オリジン不達の模擬で休止ページ: {'OK' if sleeping == 503 else 'NG'}（{sleeping}）")
    return ok and normal == 200 and sleeping == 503


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="差分を反映する（既定はドライラン）")
    parser.add_argument("--verify", action="store_true", help="反映後の動作確認だけ行う")
    args = parser.parse_args(argv)

    token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    if not token:
        print("CLOUDFLARE_API_TOKEN が未設定です（scripts/cloudflare_edge_setup.sh 経由で実行）", file=sys.stderr)
        return 2
    api = Api(token)
    del token

    if args.verify:
        return 0 if verify(api) else 1

    plan = build_plan(api)
    for change in plan.changes:
        print(change.render())
    pending = [c for c in plan.changes if c.kind in ("create", "update")]
    manual = [c for c in plan.changes if c.kind == "manual"]
    if not args.apply:
        print(f"\nドライラン: 変更 {len(pending)} 件 / 手作業 {len(manual)} 件。反映は --apply")
        return 0
    for change in pending:
        print(f"applying: {change.resource}")
        assert change.apply is not None
        change.apply()
    print(f"\n反映完了: {len(pending)} 件" + (f"（手作業 {len(manual)} 件が残っています）" if manual else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
