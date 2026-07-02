"""Gemini APIキー取得の共通実装（Streamlit非依存）。

api/main.py・multi_agent_screening.py・shion_self_analysis.py・chat_memory.py に
同等の実装が4重複していたため集約した（2026-07 紫苑システムレビュー指摘）。
FastAPI プロセスから使うため streamlit / secret_manager.py には依存しない。

優先順位: 環境変数 GEMINI_API_KEY → .streamlit/secrets.toml
（api/ から最大5階層遡って直接パース。worktree / 本体リポジトリ両対応）
"""
from __future__ import annotations

import os
import re

_KEY_LINE = re.compile(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']')


def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    here = os.path.dirname(os.path.abspath(__file__))
    cur = os.path.dirname(here)
    for _ in range(5):
        sec_path = os.path.join(cur, ".streamlit", "secrets.toml")
        if os.path.exists(sec_path):
            try:
                with open(sec_path, encoding="utf-8") as f:
                    for line in f:
                        m = _KEY_LINE.match(line.strip())
                        if m:
                            return m.group(1)
            except OSError:
                pass
        cur = os.path.dirname(cur)
    return ""
