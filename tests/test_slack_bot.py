# -*- coding: utf-8 -*-
"""
tests/test_slack_bot.py
========================
slack_bot.py のコマンド解析・アクション実行（採用/修正/保留/却下/承認）・
チャンネルメンションのスレッド返信ラッパーのテスト。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def mock_heavy_deps(monkeypatch):
    """slack_sdk・ai_chat（重い間接依存を持つ）をモックし、
    Slackトークン系のシークレット参照が streamlit の未設定エラーで
    落ちないようダミー環境変数を設定してからインポートする。"""
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/test")
    with patch.dict("sys.modules", {
        "slack_sdk": MagicMock(),
        "slack_sdk.errors": MagicMock(),
        "ai_chat": MagicMock(),
    }):
        yield


@pytest.fixture
def slack_bot(mock_heavy_deps):
    import importlib
    import slack_bot as module
    importlib.reload(module)
    return module


def test_parse_command_action_keywords(slack_bot):
    assert slack_bot._parse_command("採用 skill-042") == ("adopt", "skill-042")
    assert slack_bot._parse_command("修正 skill-042 ここを直して") == ("revise", "skill-042 ここを直して")
    assert slack_bot._parse_command("保留 skill-042") == ("hold", "skill-042")
    assert slack_bot._parse_command("却下 skill-042 根拠不足") == ("reject", "skill-042 根拠不足")
    assert slack_bot._parse_command("承認 add-xxx-yyy") == ("approve_triage", "add-xxx-yyy")


def test_parse_command_does_not_steal_normal_chat_prefixes(slack_bot):
    assert slack_bot._parse_command("承認条件を整理して") == ("chat", "承認条件を整理して")
    assert slack_bot._parse_command("修正案を出して") == ("chat", "修正案を出して")
    assert slack_bot._parse_command("採用すべき判断軸は？") == ("chat", "採用すべき判断軸は？")


def test_parse_command_falls_back_to_chat(slack_bot):
    assert slack_bot._parse_command("今月の否決率が高い業種は？") == ("chat", "今月の否決率が高い業種は？")


def test_action_command_rejected_without_whitelist(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", set())
    review_mock = MagicMock()
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 skill-042", "U999")

    review_mock.assert_not_called()
    posted_text = client.chat_postMessage.call_args.kwargs["text"]
    assert "権限がありません" in posted_text


def test_action_command_rejected_for_non_whitelisted_user(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock()
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 skill-042", "U_OTHER")

    review_mock.assert_not_called()


def test_action_commands_are_disabled_by_default(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", False)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 skill-042", "U_ALLOWED")

    review_mock.assert_not_called()
    posted_text = client.chat_postMessage.call_args.kwargs["text"]
    assert "無効" in posted_text and "Cloud Run" in posted_text


def test_adopt_command_waits_for_confirmation_when_enabled(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 skill-042", "U_ALLOWED")

    review_mock.assert_not_called()
    posted_text = client.chat_postMessage.call_args.kwargs["text"]
    assert "確認" in posted_text and "はい" in posted_text and "skill-042" in posted_text


def test_adopt_command_executes_after_confirmation(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 skill-042", "U_ALLOWED", event_key="cmd-1")
    slack_bot.handle_message(client, "C123", "はい", "U_ALLOWED", event_key="yes-1")

    review_mock.assert_called_once_with("skill-042", decision="adopted", note="", edited_claim="")
    posted_text = client.chat_postMessage.call_args.kwargs["text"]
    assert "skill-042" in posted_text and "adopted" in posted_text


def test_revise_command_requires_note(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock()
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "修正 skill-042", "U_ALLOWED")

    review_mock.assert_not_called()
    posted_text = client.chat_postMessage.call_args.kwargs["text"]
    assert "メモ" in posted_text


def test_revise_command_keeps_note_out_of_edited_claim_after_confirmation(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "修正 skill-042 ここを直して", "U_ALLOWED")
    review_mock.assert_not_called()
    slack_bot.handle_message(client, "C123", "はい", "U_ALLOWED", event_key="yes-revise-1")

    review_mock.assert_called_once_with(
        "skill-042",
        decision="revised",
        note="ここを直して",
        edited_claim="",
    )


def test_approve_triage_calls_endpoint_for_allowed_user(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    approve_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_triage_approve", approve_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "承認 add-xxx-yyy", "U_ALLOWED")
    approve_mock.assert_not_called()
    slack_bot.handle_message(client, "C123", "はい", "U_ALLOWED", event_key="yes-approve-1")

    approve_mock.assert_called_once_with("add-xxx-yyy")
    posted_text = client.chat_postMessage.call_args.kwargs["text"]
    assert "add-xxx-yyy" in posted_text


def test_action_confirmation_can_be_cancelled(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 skill-042", "U_ALLOWED")
    slack_bot.handle_message(client, "C123", "キャンセル", "U_ALLOWED")

    review_mock.assert_not_called()
    assert "キャンセル" in client.chat_postMessage.call_args.kwargs["text"]


def test_thread_confirmation_reply_can_be_detected_for_channel_message(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(
        client,
        "C123",
        "<@BOT> 採用 skill-042",
        "U_ALLOWED",
        thread_ts="1700000000.000100",
    )

    assert slack_bot._has_pending_action_confirmation("C123", "U_ALLOWED", "1700000000.000100", "はい")
    assert not slack_bot._has_pending_action_confirmation("C123", "U_ALLOWED", "", "はい")

    slack_bot.handle_message(
        client,
        "C123",
        "はい",
        "U_ALLOWED",
        thread_ts="1700000000.000100",
        event_key="thread-yes-1",
    )

    review_mock.assert_called_once_with("skill-042", decision="adopted", note="", edited_claim="")


def test_duplicate_confirmation_event_is_ignored(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 skill-042", "U_ALLOWED")
    slack_bot.handle_message(client, "C123", "はい", "U_ALLOWED", event_key="dup-yes")
    slack_bot._PENDING_ACTION_CONFIRMATIONS[("C123", "U_ALLOWED", "")] = {
        "payload": {
            "kind": "agentic_skill_review",
            "command": "adopt",
            "label": "採用",
            "target": "skill-042",
            "decision": "adopted",
            "note": "",
            "edited_claim": "",
        },
        "expires_at": 999999999999.0,
    }
    slack_bot.handle_message(client, "C123", "はい", "U_ALLOWED", event_key="dup-yes")

    review_mock.assert_called_once()


def test_invalid_action_id_is_rejected_before_confirmation(slack_bot, monkeypatch):
    monkeypatch.setattr(slack_bot, "_ENABLE_ACTION_COMMANDS", True)
    monkeypatch.setattr(slack_bot, "_ALLOWED_ACTION_USERS", {"U_ALLOWED"})
    review_mock = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(slack_bot, "_post_agentic_skill_review", review_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "採用 ../bad", "U_ALLOWED")

    review_mock.assert_not_called()
    assert "対象ID" in client.chat_postMessage.call_args.kwargs["text"]


def test_chat_fallback_uses_shion_reply(slack_bot, monkeypatch):
    reply_mock = MagicMock(return_value="紫苑からの返信です")
    monkeypatch.setattr(slack_bot, "_get_shion_reply", reply_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "リース期間について教えて", "U1")

    reply_mock.assert_called_once_with("リース期間について教えて", user_id="U1")
    last_text = client.chat_postMessage.call_args.kwargs["text"]
    assert "紫苑からの返信です" in last_text


def test_thread_reply_client_injects_thread_ts(slack_bot):
    inner_client = MagicMock()
    wrapped = slack_bot._ThreadReplyClient(inner_client, "1700000000.000100")

    wrapped.chat_postMessage(channel="C123", text="hello")

    inner_client.chat_postMessage.assert_called_once_with(
        channel="C123", text="hello", thread_ts="1700000000.000100",
    )


def test_direct_message_event_requires_explicit_im_channel_type(slack_bot):
    assert slack_bot._is_direct_message_event({"channel_type": "im"})
    assert not slack_bot._is_direct_message_event({"channel_type": "channel"})
    assert not slack_bot._is_direct_message_event({})


def test_handle_message_with_thread_ts_threads_all_replies(slack_bot, monkeypatch):
    reply_mock = MagicMock(return_value="スレッド内の返信")
    monkeypatch.setattr(slack_bot, "_get_shion_reply", reply_mock)

    client = MagicMock()
    slack_bot.handle_message(client, "C123", "こんにちは", "U1", thread_ts="1700000000.000100")

    for call in client.chat_postMessage.call_args_list:
        assert call.kwargs.get("thread_ts") == "1700000000.000100"
