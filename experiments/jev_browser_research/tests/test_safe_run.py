import sys
from types import ModuleType, SimpleNamespace

import safe_run
from safe_run import action_safety_reason, host_is_allowed


def _page(label="統計データを検索", *, role="button", kind="click", href=None):
    action = {"id": "e1", "kind": kind, "label": label, "role": role}
    if href is not None:
        action["href"] = href
    return {
        "url": "https://www.e-stat.go.jp/",
        "actions": [action],
    }


def _decision(operation="CLICK", **updates):
    value = {
        "choice": "e1",
        "operation": operation,
        "confidence": 0.9,
        "target_confidence": 0.8,
    }
    value.update(updates)
    return value


def test_host_allowlist_accepts_only_domain_boundaries():
    assert host_is_allowed("https://www.e-stat.go.jp/", ("e-stat.go.jp",))
    assert host_is_allowed("https://www.meti.go.jp/", ("go.jp",))
    assert not host_is_allowed("https://go.jp.example.com/", ("go.jp",))
    assert not host_is_allowed("file:///tmp/private", ("go.jp",))
    assert not host_is_allowed(
        "https://evil.example\\@www.e-stat.go.jp/", ("e-stat.go.jp",)
    )
    assert not host_is_allowed(
        "https://evil.example%5C@www.e-stat.go.jp/", ("e-stat.go.jp",)
    )


def test_safe_search_click_is_allowed():
    assert action_safety_reason(_decision(), _page(), min_confidence=0.55)[0]


def test_link_destination_is_checked_before_navigation():
    safe, reason = action_safety_reason(
        _decision(),
        _page(role="link", href="https://example.com/collect"),
        min_confidence=0.55,
        allowed_hosts=("e-stat.go.jp",),
    )
    assert not safe
    assert "outside the allowlist" in reason

    safe, _ = action_safety_reason(
        _decision(),
        _page(role="link", href="/stat-search"),
        min_confidence=0.55,
        allowed_hosts=("e-stat.go.jp",),
    )
    assert safe


def test_link_without_observed_destination_is_blocked():
    safe, reason = action_safety_reason(
        _decision(),
        _page(role="link"),
        min_confidence=0.55,
        allowed_hosts=("e-stat.go.jp",),
    )
    assert not safe
    assert "not observed" in reason


def test_login_and_submission_are_blocked():
    for label in ("ログイン", "申請を送信", "Purchase now", "ファイルをアップロード"):
        safe, _ = action_safety_reason(_decision(), _page(label), min_confidence=0.55)
        assert not safe


def test_text_entry_is_limited_to_text_controls():
    safe, _ = action_safety_reason(
        _decision("TYPE_TEXT"),
        _page("キーワード検索", role="searchbox", kind="fill"),
        min_confidence=0.55,
    )
    assert safe
    safe, _ = action_safety_reason(
        _decision("TYPE_TEXT"),
        _page("任意入力", role="button", kind="fill"),
        min_confidence=0.55,
    )
    assert not safe


def test_low_confidence_stops_before_action():
    safe, reason = action_safety_reason(
        _decision(confidence=0.2), _page(), min_confidence=0.55
    )
    assert not safe
    assert "below" in reason


def test_low_confidence_wait_is_safe_but_bounded_by_runner():
    safe, reason = action_safety_reason(
        _decision(operation="WAIT", choice="wait", confidence=0.2),
        _page(),
        min_confidence=0.55,
    )
    assert safe
    assert reason == "safe control operation"


def test_text_model_key_can_be_loaded_from_keychain(monkeypatch):
    monkeypatch.setattr(safe_run.sys, "platform", "darwin")
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("TEXT_MODEL_API_KEYCHAIN_SERVICE", "test-text-service")
    monkeypatch.setattr(
        safe_run.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="secret-from-keychain\n"),
    )

    assert safe_run.load_api_key_from_keychain(
        "TEXT_MODEL_API_KEY", "TEXT_MODEL_API_KEYCHAIN_SERVICE"
    )
    assert safe_run.os.environ["TEXT_MODEL_API_KEY"] == "secret-from-keychain"


def test_stale_page_retry_rechecks_observed_host(monkeypatch):
    class StalePage(Exception):
        pass

    class Browser:
        def observe(self, **_kwargs):
            return {"url": "https://evil.example/", "fingerprint": "changed"}

    class Agent:
        def __init__(self, *_args):
            self.screenshots = False
            self.state = {"browser": Browser()}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def snapshot(self):
            return {"status": "ready", "page": {"url": "https://www.e-stat.go.jp/"}}

        def command(self, operation, *_args):
            if operation == "predict":
                return {
                    "decision": {"operation": "WAIT", "choice": "wait", "confidence": 0.9},
                    "page": {"url": "https://www.e-stat.go.jp/", "fingerprint": "old"},
                }
            raise StalePage

    package = ModuleType("jev_ultrafast")
    package.Agent = Agent
    browser = ModuleType("jev_ultrafast.browser")
    browser.StalePage = StalePage
    monkeypatch.setitem(sys.modules, "jev_ultrafast", package)
    monkeypatch.setitem(sys.modules, "jev_ultrafast.browser", browser)
    monkeypatch.setattr(safe_run, "configure_cdp", lambda _url: None)
    monkeypatch.setattr(safe_run, "load_api_key_from_keychain", lambda *_args: True)
    args = SimpleNamespace(
        url="https://www.e-stat.go.jp/",
        goal="統計を見る",
        allow_host=["e-stat.go.jp"],
        cdp_http="",
        auto=True,
        max_steps=1,
        min_confidence=0.55,
    )

    assert safe_run.run(args) == 2
