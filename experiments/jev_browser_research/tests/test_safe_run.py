from types import SimpleNamespace

import safe_run
from safe_run import action_safety_reason, host_is_allowed


def _page(label="統計データを検索", *, role="button", kind="click"):
    return {
        "actions": [
            {"id": "e1", "kind": kind, "label": label, "role": role}
        ]
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


def test_safe_search_click_is_allowed():
    assert action_safety_reason(_decision(), _page(), min_confidence=0.55)[0]


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
