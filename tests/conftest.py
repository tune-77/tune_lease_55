"""
pytest 共通設定・フィクスチャ
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# tune_lease_55/ をパスに追加
PKG_DIR = Path(__file__).parent.parent
REPO_DIR = PKG_DIR.parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(REPO_DIR))

# Streamlit や重いライブラリをモック（CI環境でインストール不要に）
for _mod in [
    "streamlit", "streamlit.components", "streamlit.components.v1",
    "plotly", "plotly.express", "plotly.graph_objects",
    "reportlab", "reportlab.lib", "reportlab.platypus",
    "ollama", "pgmpy", "pgmpy.models", "pgmpy.factors",
    "pgmpy.factors.discrete", "pgmpy.inference",
    "lightgbm", "shap",
]:
    sys.modules.setdefault(_mod, MagicMock())


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_jev_judgment_log(tmp_path, monkeypatch):
    """Jev判定ログ（REV-424）をテストがリポジトリの data/ へ書かないようにする。"""
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", str(tmp_path / "jev_judgment_log.jsonl"))


@pytest.fixture(autouse=True)
def _isolate_silent_failure_log(tmp_path, monkeypatch):
    """黙った失敗の記録をテストがリポジトリの data/ へ書かないようにする。"""
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(tmp_path / "silent_failures.jsonl"))


@pytest.fixture(autouse=True)
def _cloudrun_not_paused(monkeypatch):
    """config/cloudrun_pause.json が停止中でも、同期処理のテストは稼働中として動かす。"""
    monkeypatch.setenv("CLOUDRUN_PAUSED", "0")


@pytest.fixture(autouse=True)
def _isolate_shion_vault_memory_sources(tmp_path, monkeypatch):
    """記憶索引が実 Vault を読まず、教示ファネルも data/ へ書かないようにする。"""
    monkeypatch.setenv("SHION_MEMORY_INDEX_VAULT", "off")
    monkeypatch.setenv("SHION_TEACHING_FUNNEL_PATH", str(tmp_path / "shion_teaching_funnel.jsonl"))
    monkeypatch.setenv("SHION_KNOWLEDGE_TITLE_LLM", "off")  # Knowledge ノートの題名づけで Gemini を呼ばない
