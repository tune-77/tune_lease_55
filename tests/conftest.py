"""
pytest 共通設定・フィクスチャ
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# lease_logic_sumaho12/ をパスに追加
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
