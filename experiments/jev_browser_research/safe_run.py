"""Read-only-by-policy runner for the isolated Jev Ultrafast pilot.

The upstream agent owns browser freshness and DOM-node validation. This wrapper
adds project-level boundaries: host allowlisting, risky-label blocking, a small
step budget, and an explicit ``--auto`` switch before any model-selected action.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_ALLOWED_HOSTS = ("e-stat.go.jp", "go.jp", "boj.or.jp")
SAFE_OPERATIONS = {
    "CLICK",
    "TYPE_TEXT",
    "SELECT",
    "SCROLL_UP",
    "SCROLL_DOWN",
    "WAIT",
    "DONE",
    "BLOCKED",
}
BLOCKED_LABEL_TERMS = (
    "ログイン",
    "新規登録",
    "会員登録",
    "申し込",
    "申請",
    "送信",
    "購入",
    "注文",
    "予約",
    "支払",
    "削除",
    "アップロード",
    "login",
    "log in",
    "sign in",
    "sign up",
    "register",
    "submit",
    "apply",
    "purchase",
    "buy",
    "checkout",
    "delete",
    "upload",
)
TEXT_ROLES = {"textbox", "searchbox", "combobox"}


def normalized_host(url: str) -> str:
    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.scheme not in {"http", "https"}:
        return ""
    return (parsed.hostname or "").lower().rstrip(".")


def host_is_allowed(url: str, allowed_hosts: Sequence[str]) -> bool:
    host = normalized_host(url)
    for raw_suffix in allowed_hosts:
        suffix = str(raw_suffix).lower().strip().lstrip(".").rstrip(".")
        if suffix and (host == suffix or host.endswith("." + suffix)):
            return True
    return False


def selected_action(
    decision: Mapping[str, Any], page: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    choice = str(decision.get("choice") or "")
    if choice in {"DONE", "BLOCKED"}:
        return None
    for action in page.get("actions") or []:
        if isinstance(action, Mapping) and str(action.get("id") or "") == choice:
            return action
    return None


def action_destination(
    action: Mapping[str, Any], page: Mapping[str, Any]
) -> str:
    """Resolve an observed link target without activating it."""
    attributes = action.get("attributes")
    nested = attributes if isinstance(attributes, Mapping) else {}
    raw_target = action.get("href") or action.get("url") or nested.get("href")
    if not raw_target:
        return ""
    return urllib.parse.urljoin(str(page.get("url") or ""), str(raw_target))


def action_safety_reason(
    decision: Mapping[str, Any],
    page: Mapping[str, Any],
    *,
    min_confidence: float,
    allowed_hosts: Sequence[str] = (),
) -> tuple[bool, str]:
    operation = str(decision.get("operation") or "")
    if operation not in SAFE_OPERATIONS:
        return False, f"unsupported operation: {operation or 'missing'}"

    # These choices do not submit data or activate a DOM target. Low confidence
    # may mean several harmless options are similarly plausible, so the bounded
    # step budget is the appropriate guard rather than a confidence threshold.
    if operation in {"DONE", "BLOCKED", "SCROLL_UP", "SCROLL_DOWN", "WAIT"}:
        return True, "safe control operation"

    try:
        confidence = float(decision.get("confidence"))
    except (TypeError, ValueError):
        return False, "invalid operation confidence"
    if confidence < min_confidence:
        return False, f"operation confidence {confidence:.3f} is below {min_confidence:.3f}"

    action = selected_action(decision, page)
    if action is None:
        return False, "selected DOM action was not observed"
    label = str(action.get("label") or "").lower()
    blocked = next((term for term in BLOCKED_LABEL_TERMS if term in label), None)
    if blocked:
        return False, f"blocked label term: {blocked}"

    if operation == "CLICK" and allowed_hosts:
        role = str(action.get("role") or "").lower()
        destination = action_destination(action, page)
        if role == "link" and not destination:
            return False, "link destination was not observed before navigation"
        if destination and not host_is_allowed(destination, allowed_hosts):
            return False, f"link destination is outside the allowlist: {destination}"

    if operation == "TYPE_TEXT":
        role = str(action.get("role") or "").lower()
        if role not in TEXT_ROLES:
            return False, f"text entry role is not allowed: {role or 'missing'}"

    target_confidence = decision.get("target_confidence")
    if target_confidence is not None:
        try:
            target_value = float(target_confidence)
        except (TypeError, ValueError):
            return False, "invalid target confidence"
        if target_value < min_confidence:
            return False, f"target confidence {target_value:.3f} is below {min_confidence:.3f}"
    return True, "allowed observed action"


def configure_cdp(cdp_http: str) -> None:
    if os.environ.get("BU_CDP_WS") or not cdp_http:
        return
    endpoint = cdp_http.rstrip("/") + "/json/version"
    with urllib.request.urlopen(endpoint, timeout=3) as response:
        body = json.load(response)
    websocket_url = str(body.get("webSocketDebuggerUrl") or "")
    if not websocket_url.startswith(("ws://", "wss://")):
        raise RuntimeError("Chrome did not publish a DevTools WebSocket URL")
    os.environ["BU_CDP_WS"] = websocket_url


def load_api_key_from_keychain(key_env: str, service_env: str) -> bool:
    """Load one API key without printing it or placing it in command history."""
    if os.environ.get(key_env):
        return True
    service = str(os.environ.get(service_env) or "").strip()
    if not service or sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    key = result.stdout.strip()
    if key:
        os.environ[key_env] = key
        return True
    return False


def public_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    page = state.get("page") or {}
    elements = []
    for element in state.get("elements") or []:
        elements.append(
            {
                "index": element.get("index"),
                "operations": element.get("operations") or [],
                "label": str(element.get("label") or "")[:160],
            }
        )
    return {
        "status": state.get("status"),
        "url": page.get("url"),
        "title": page.get("title"),
        "element_count": len(elements),
        "elements": elements,
    }


def run(args: argparse.Namespace) -> int:
    allowed_hosts = tuple(args.allow_host or DEFAULT_ALLOWED_HOSTS)
    if not host_is_allowed(args.url, allowed_hosts):
        raise ValueError(f"start URL is outside the allowlist: {args.url}")
    configure_cdp(args.cdp_http)
    os.environ.setdefault("BH_TELEMETRY", "0")

    from jev_ultrafast import Agent
    from jev_ultrafast.browser import StalePage

    with Agent(args.url, args.goal) as agent:
        initial = agent.snapshot()
        print(json.dumps(public_snapshot(initial), ensure_ascii=False, indent=2))
        initial_url = str((initial.get("page") or {}).get("url") or "")
        if not host_is_allowed(initial_url, allowed_hosts):
            print(f"STOPPED: initial navigation left the allowlist: {initial_url}")
            return 2
        if not args.auto:
            print("DRY RUN: no model call or browser action was executed")
            return 0
        if not load_api_key_from_keychain(
            "TYPESAFE_API_KEY", "TYPESAFE_API_KEYCHAIN_SERVICE"
        ):
            raise RuntimeError(
                "TYPESAFE_API_KEY is unavailable; set it server-side or configure "
                "TYPESAFE_API_KEYCHAIN_SERVICE"
            )
        load_api_key_from_keychain(
            "TEXT_MODEL_API_KEY", "TEXT_MODEL_API_KEYCHAIN_SERVICE"
        )

        for _ in range(args.max_steps):
            state = agent.command("predict")
            decision = state.get("decision") or {}
            safe, reason = action_safety_reason(
                decision,
                state.get("page") or {},
                min_confidence=args.min_confidence,
                allowed_hosts=allowed_hosts,
            )
            summary = {
                "operation": decision.get("operation"),
                "target": decision.get("target"),
                "confidence": decision.get("confidence"),
                "target_confidence": decision.get("target_confidence"),
                "safe": safe,
                "reason": reason,
            }
            print(json.dumps(summary, ensure_ascii=False))
            if not safe:
                print(json.dumps(public_snapshot(state), ensure_ascii=False, indent=2))
                print("STOPPED: policy rejected the predicted action")
                return 2
            try:
                state = agent.command(
                    "act", {"fingerprint": state["page"]["fingerprint"]}
                )
            except StalePage:
                # Match the upstream tick loop: discard the stale decision,
                # observe the current DOM, and let Jev choose again.
                agent.state["decision"] = None
                agent.state["status"] = "ready"
                agent.state["page"] = agent.state["browser"].observe(
                    screenshot=agent.screenshots
                )
                print("RETRY: page changed after prediction; observed the current DOM")
                continue
            current_url = str((state.get("page") or {}).get("url") or "")
            if not host_is_allowed(current_url, allowed_hosts):
                print(f"STOPPED: navigation left the allowlist: {current_url}")
                return 2
            if state.get("status") in {"done", "blocked"}:
                print(json.dumps(public_snapshot(state), ensure_ascii=False, indent=2))
                return 0 if state.get("status") == "done" else 2
        print(f"STOPPED: reached local step budget ({args.max_steps})")
        return 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--url", required=True)
    value.add_argument("--goal", required=True)
    value.add_argument("--auto", action="store_true", help="Allow model-selected browser actions")
    value.add_argument("--allow-host", action="append", default=[])
    value.add_argument("--max-steps", type=int, choices=range(1, 13), default=8)
    value.add_argument("--min-confidence", type=float, default=0.55)
    value.add_argument("--cdp-http", default="http://127.0.0.1:9222")
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
