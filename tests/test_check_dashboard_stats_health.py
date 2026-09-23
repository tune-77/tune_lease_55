from __future__ import annotations

import re
from pathlib import Path

from api.dashboard_data_health import evaluate_dashboard_data_health
import scripts.check_dashboard_stats_health as checker


ROOT = Path(__file__).resolve().parents[1]


def _dashboard_payload(**analysis_overrides):
    analysis = {
        "closed_count": 12,
        "avg_financials": {"sales": 100},
        "top3_drivers": [{"name": "financial"}],
    }
    analysis.update(analysis_overrides)
    return {"analysis": analysis, "recent_cases": []}


def test_healthy_dashboard_payload_passes() -> None:
    assert evaluate_dashboard_data_health(_dashboard_payload()) == (True, "ok")


def test_null_closed_count_fails_without_exposing_data() -> None:
    assert evaluate_dashboard_data_health(_dashboard_payload(closed_count=None)) == (
        False,
        "invalid_closed_count",
    )


def test_closed_cases_with_null_graph_data_fails() -> None:
    assert evaluate_dashboard_data_health(_dashboard_payload(top3_drivers=None)) == (
        False,
        "missing_top3_drivers",
    )


def test_zero_closed_cases_allows_empty_aggregates() -> None:
    payload = _dashboard_payload(closed_count=0, avg_financials=None, top3_drivers=None)
    assert evaluate_dashboard_data_health(payload) == (True, "ok")


def test_missing_recent_cases_fails() -> None:
    payload = _dashboard_payload()
    payload.pop("recent_cases")
    assert evaluate_dashboard_data_health(payload) == (False, "missing_recent_cases")


def test_checker_accepts_only_minimal_healthy_response() -> None:
    assert checker.validate_health_response({"healthy": True, "reason": "ok"}) == (
        True,
        "ok",
    )
    healthy, reason = checker.validate_health_response(
        {"healthy": False, "reason": "missing_analysis"}
    )
    assert healthy is False
    assert reason == "dashboard data unhealthy: missing_analysis"


def test_scheduled_workflow_uses_privacy_preserving_health_endpoint() -> None:
    workflow = (ROOT / ".github/workflows/dashboard-data-health.yml").read_text(
        encoding="utf-8"
    )
    checker_source = (ROOT / "scripts/check_dashboard_stats_health.py").read_text(
        encoding="utf-8"
    )

    # 2026-09、プローブ自体がmin-instances=0のCloud Runをコールドスタートさせ
    # 課金増の一因になっていたため、知識同期チェックを統合し平日JST 9/13/17時の
    # 1日3回に絞った（分オフセット17はデプロイ直後と重ならせない狙いのまま維持）。
    assert 'cron: "17 0,4,8 * * 1-5"' in workflow
    assert "python scripts/check_cloudrun_knowledge_sync.py" in workflow
    assert "KNOWLEDGE_SYNC_PROBE_TOKEN: ${{ secrets.KNOWLEDGE_SYNC_PROBE_TOKEN }}" in workflow
    assert not (ROOT / ".github/workflows/knowledge-sync-health.yml").exists()
    # actions/checkout・actions/setup-pythonの配線がワークフローから消えていない
    # ことだけを保証する。バージョン番号は他ワークフローとの追従対象であり
    # Dependabotの更新のたびに変わるため、特定のaction versionはここで固定しない
    # （2026-09にv5→v7更新でこの行がハードコード起因で壊れた事故の再発防止）。
    assert re.search(r"actions/checkout@v\d+", workflow)
    assert re.search(r"actions/setup-python@v\d+", workflow)
    assert "python scripts/check_dashboard_stats_health.py" in workflow
    assert "tune-lease-55-web-6mijhyebkq-an.a.run.app" in workflow
    assert 'DEFAULT_PATH = "/api/dashboard/data-health"' in checker_source
    assert "/api/dashboard/stats" not in checker_source


def test_workflow_passes_the_dashboard_health_probe_token() -> None:
    """PUBLIC_TUNNEL=1 導入後、frontend/src/proxy.ts は /api/dashboard/data-health も
    Basic認証(または専用トークン)で守る。ワークフローが専用シークレットを渡さないと
    毎回401で落ちる（2026-09-08以降の実障害）ため、配線が消えないことを固定する。
    """
    workflow = (ROOT / ".github/workflows/dashboard-data-health.yml").read_text(
        encoding="utf-8"
    )

    assert "DASHBOARD_HEALTH_PROBE_TOKEN: ${{ secrets.DASHBOARD_HEALTH_PROBE_TOKEN }}" in workflow


def test_checker_fails_closed_without_probe_token(monkeypatch) -> None:
    monkeypatch.delenv(checker.PROBE_TOKEN_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "sys.argv", ["check_dashboard_stats_health.py", "--base-url", "https://example.invalid"]
    )

    assert checker.main() == 1


def test_checker_sends_probe_token_header(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return b'{"healthy": true, "reason": "ok"}'

    def _fake_urlopen(request, timeout):
        captured["header"] = request.get_header("X-dashboard-health-probe-key")
        return _FakeResponse()

    monkeypatch.setattr(checker.urllib.request, "urlopen", _fake_urlopen)

    payload = checker.fetch_health_response(
        "https://example.invalid", checker.DEFAULT_PATH, 5.0, "secret-token"
    )

    assert payload == {"healthy": True, "reason": "ok"}
    assert captured["header"] == "secret-token"
