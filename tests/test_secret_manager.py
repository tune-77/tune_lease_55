"""
テスト: secret_manager.py
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from secret_manager import get_secret_value, get_gemini_api_key


@patch('secret_manager.st')
def test_get_secret_value_priority(mock_st):
    """優先順位テスト: 環境変数 > st.secrets > secrets.toml > fallback"""
    mock_st.secrets.get.return_value = None

    # 環境変数を設定
    os.environ["TEST_KEY"] = "env_value"
    assert get_secret_value("TEST_KEY") == "env_value"

    # 環境変数を削除
    del os.environ["TEST_KEY"]

    # st.secretsがNoneを返す
    assert get_secret_value("NON_EXISTENT_KEY", "default") == "default"


@patch('secret_manager.st')
def test_get_gemini_api_key(mock_st):
    """Gemini APIキー取得テスト"""
    mock_st.secrets.get.return_value = None

    # 環境変数でテスト
    os.environ["GEMINI_API_KEY"] = "test_key"
    assert get_gemini_api_key() == "test_key"
    del os.environ["GEMINI_API_KEY"]



class _FakeSecretNotFound(FileNotFoundError):
    """StreamlitSecretNotFoundError と同じ継承を再現した偽例外。

    conftest.py が streamlit を MagicMock へ差し替える（CI で未インストールでも
    動かすため）ので、本物のクラスは import できない。2026-09-26 / streamlit
    1.54.0 で実際のMROを確認した:
        StreamlitSecretNotFoundError → LocalizableStreamlitException
        → StreamlitAPIException → ... → FileNotFoundError → OSError → Exception
    AttributeError でも KeyError でもない点が、この回帰テストの要。
    """


@patch('secret_manager.st')
def test_streamlit_secret_error_falls_through_to_toml(mock_st, tmp_path):
    """st.secrets が例外を投げても secrets.toml へ到達する。

    以前の `except (AttributeError, KeyError)` では捕捉できず、リポジトリ外を
    cwd として起動した launchd ジョブでは例外が呼び出し元へ抜けて手順3に
    到達しなかった。その結果 GEMINI_API_KEY が取得できず、ニュース収集の
    AI分類が無言でスキップされていた。
    """
    mock_st.secrets.get.side_effect = _FakeSecretNotFound()

    toml_path = tmp_path / "secrets.toml"
    toml_path.write_text('TEST_TOML_KEY = "from_toml"\n', encoding="utf-8")

    with patch('secret_manager.SECRETS_TOML_PATH', toml_path):
        assert get_secret_value("TEST_TOML_KEY") == "from_toml"


@patch('secret_manager.st')
def test_streamlit_secret_error_still_returns_fallback(mock_st, tmp_path):
    """secrets.toml にも無い場合は、例外を投げずに fallback を返す。"""
    mock_st.secrets.get.side_effect = _FakeSecretNotFound()

    with patch('secret_manager.SECRETS_TOML_PATH', tmp_path / "absent.toml"):
        assert get_secret_value("MISSING_KEY", "default") == "default"
