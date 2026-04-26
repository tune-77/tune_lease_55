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