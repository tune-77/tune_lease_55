#!/usr/bin/env python3
"""Diagnose the local Next/FastAPI launcher and recommend the next command."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_NEXT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_NEXT_PORT = 3000
SERVICE_LABEL = "com.tunelease.next"
TUNNEL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


@dataclass(frozen=True)
class EndpointState:
    label: str
    url: str
    http_ok: bool
    listening: bool

    @property
    def status(self) -> str:
        if self.http_ok:
            return "OK"
        if self.listening:
            return "LISTENING"
        return "DOWN"


@dataclass(frozen=True)
class DeployState:
    api: EndpointState
    next: EndpointState
    launchagent_installed: bool
    cloudflared_available: bool
    tunnel_url: str | None
    public_tunnel: bool


@dataclass(frozen=True)
class Recommendation:
    action: str
    reason: str
    command: str


def run_probe(args: list[str], timeout: float = 2.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def http_ok(url: str) -> bool:
    result = run_probe(["curl", "-fsS", "--max-time", "2", url])
    return bool(result and result.returncode == 0)


def port_listening(port: int) -> bool:
    result = run_probe(["lsof", "-tiTCP:%s" % port, "-sTCP:LISTEN"])
    return bool(result and result.returncode == 0 and result.stdout.strip())


def launchagent_installed(service_label: str = SERVICE_LABEL) -> bool:
    result = run_probe(["launchctl", "print", "gui/%s/%s" % (os.getuid(), service_label)])
    return bool(result and result.returncode == 0)


def latest_tunnel_url(log_dir: Path, named_hostname: str | None = None) -> str | None:
    if named_hostname:
        return "https://%s" % named_hostname
    if not log_dir.exists():
        return None
    for path in sorted(log_dir.glob("tunnel_*.log"), key=lambda item: item.stat().st_mtime, reverse=True):
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = TUNNEL_RE.findall(text)
        if matches:
            return matches[-1]
    return None


def build_state(
    *,
    root: Path,
    api_host: str,
    api_port: int,
    next_host: str,
    next_port: int,
    public_tunnel: bool,
) -> DeployState:
    api_url = "http://%s:%s/docs" % (api_host, api_port)
    next_url = "http://%s:%s/" % (next_host, next_port)
    return DeployState(
        api=EndpointState("API", api_url, http_ok(api_url), port_listening(api_port)),
        next=EndpointState("Next", next_url, http_ok(next_url), port_listening(next_port)),
        launchagent_installed=launchagent_installed(),
        cloudflared_available=shutil.which("cloudflared") is not None,
        tunnel_url=latest_tunnel_url(
            root / "logs" / "next",
            os.environ.get("CLOUDFLARE_TUNNEL_HOSTNAME") or None,
        ),
        public_tunnel=public_tunnel,
    )


def launcher_env(public_tunnel: bool, api_host: str, next_host: str) -> str:
    public = "1" if public_tunnel else "0"
    return (
        "PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH "
        "PUBLIC_TUNNEL=%s API_HOST=%s NEXT_HOST=%s" % (public, api_host, next_host)
    )


def scoped_command(scope: str, public_tunnel: bool, api_host: str, next_host: str) -> str:
    return "%s RESTART_SCOPE=%s bash run_next_stable.sh" % (
        launcher_env(public_tunnel, api_host, next_host),
        scope,
    )


def recommend(state: DeployState, *, api_host: str, next_host: str) -> Recommendation:
    if not state.launchagent_installed:
        return Recommendation(
            "install_launchagent",
            "persistent LaunchAgent is not installed",
            "bash scripts/install_next_launchagent.sh",
        )
    if state.api.http_ok and state.next.http_ok:
        if state.public_tunnel and not state.tunnel_url:
            if not state.cloudflared_available:
                return Recommendation(
                    "install_cloudflared",
                    "public tunnel was requested but cloudflared is not available",
                    "brew install cloudflare/cloudflare/cloudflared",
                )
            return Recommendation(
                "restart_tunnel",
                "local API and Next are healthy but no tunnel URL was found",
                scoped_command("tunnel", True, api_host, next_host),
            )
        return Recommendation(
            "status_only",
            "API and Next are already healthy",
            scoped_command("status", state.public_tunnel, api_host, next_host),
        )
    if not state.api.http_ok and state.next.http_ok:
        return Recommendation(
            "restart_api",
            "Next is healthy but API is not responding",
            scoped_command("api", state.public_tunnel, api_host, next_host),
        )
    if state.api.http_ok and not state.next.http_ok:
        return Recommendation(
            "restart_next",
            "API is healthy but Next is not responding",
            scoped_command("next", state.public_tunnel, api_host, next_host),
        )
    if state.api.listening or state.next.listening:
        return Recommendation(
            "kickstart_launchagent",
            "one or more ports are occupied but health checks failed",
            "launchctl kickstart -k gui/$(id -u)/%s" % SERVICE_LABEL,
        )
    return Recommendation(
        "start_all",
        "API and Next are both down",
        scoped_command("all", state.public_tunnel, api_host, next_host),
    )


def render(state: DeployState, recommendation: Recommendation) -> str:
    lines = [
        "# Local Deploy Doctor",
        "",
        "Status:",
        "- API: %s %s" % (state.api.status, state.api.url),
        "- Next: %s %s" % (state.next.status, state.next.url),
        "- LaunchAgent: %s" % ("installed" if state.launchagent_installed else "missing"),
        "- cloudflared: %s" % ("available" if state.cloudflared_available else "missing"),
        "- Tunnel: %s" % (state.tunnel_url or "not found"),
        "",
        "Recommendation:",
        "- action: %s" % recommendation.action,
        "- reason: %s" % recommendation.reason,
        "- command: %s" % recommendation.command,
    ]
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-host", default=DEFAULT_API_HOST)
    parser.add_argument("--next-host", default=DEFAULT_NEXT_HOST)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--next-port", type=int, default=DEFAULT_NEXT_PORT)
    parser.add_argument("--public-tunnel", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable state")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    state = build_state(
        root=root,
        api_host=args.api_host,
        api_port=args.api_port,
        next_host=args.next_host,
        next_port=args.next_port,
        public_tunnel=args.public_tunnel,
    )
    recommendation = recommend(state, api_host=args.api_host, next_host=args.next_host)
    if args.json:
        print(json.dumps({"state": asdict(state), "recommendation": asdict(recommendation)}, ensure_ascii=False, indent=2))
    else:
        print(render(state, recommendation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
