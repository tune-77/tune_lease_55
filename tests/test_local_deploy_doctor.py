from scripts.local_deploy_doctor import (
    DeployState,
    EndpointState,
    Recommendation,
    latest_tunnel_url,
    recommend,
)


def state(
    *,
    api_ok=False,
    api_listening=False,
    next_ok=False,
    next_listening=False,
    launchagent=True,
    cloudflared=True,
    tunnel_url=None,
    public_tunnel=True,
):
    return DeployState(
        api=EndpointState("API", "http://127.0.0.1:8000/docs", api_ok, api_listening),
        next=EndpointState("Next", "http://127.0.0.1:3000/", next_ok, next_listening),
        launchagent_installed=launchagent,
        cloudflared_available=cloudflared,
        tunnel_url=tunnel_url,
        public_tunnel=public_tunnel,
    )


def action_for(deploy_state):
    return recommend(deploy_state, api_host="127.0.0.1", next_host="127.0.0.1").action


def test_missing_launchagent_installs_first():
    assert action_for(state(launchagent=False)) == "install_launchagent"


def test_healthy_local_without_tunnel_restarts_only_tunnel():
    assert action_for(state(api_ok=True, next_ok=True, tunnel_url=None)) == "restart_tunnel"


def test_healthy_local_with_tunnel_returns_status_only():
    assert action_for(state(api_ok=True, next_ok=True, tunnel_url="https://demo.trycloudflare.com")) == "status_only"


def test_api_down_restarts_api_only():
    assert action_for(state(api_ok=False, next_ok=True)) == "restart_api"


def test_next_down_restarts_next_only():
    assert action_for(state(api_ok=True, next_ok=False)) == "restart_next"


def test_occupied_unhealthy_ports_kickstart_launchagent():
    assert action_for(state(api_ok=False, api_listening=True, next_ok=False)) == "kickstart_launchagent"


def test_both_down_starts_all():
    assert action_for(state(api_ok=False, next_ok=False)) == "start_all"


def test_missing_cloudflared_requires_install_when_public_tunnel_requested():
    recommendation = recommend(
        state(api_ok=True, next_ok=True, cloudflared=False, tunnel_url=None),
        api_host="127.0.0.1",
        next_host="127.0.0.1",
    )

    assert recommendation == Recommendation(
        action="install_cloudflared",
        reason="public tunnel was requested but cloudflared is not available",
        command="brew install cloudflare/cloudflare/cloudflared",
    )


def test_latest_tunnel_url_uses_newest_log(tmp_path):
    old = tmp_path / "tunnel_20260101.log"
    new = tmp_path / "tunnel_20260102.log"
    old.write_text("https://old.trycloudflare.com\n", encoding="utf-8")
    new.write_text("https://new.trycloudflare.com\n", encoding="utf-8")

    assert latest_tunnel_url(tmp_path) == "https://new.trycloudflare.com"
