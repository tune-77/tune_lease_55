"""api/main.py のモジュール直下に同名の関数/クラスの二重定義がないことを保証する。

背景: append 系スクリプト（scripts/append_*_api.py）によるコード追記で、
同名関数の二重定義が実際に発生していた:
  - _latest_improvement_report_path: 同一実装が2箇所（後勝ちで動作は同じだが死体コード）
  - get_app_logs: 別ルート（/api/logs/app と /api/analysis/app_logs）に同名関数。
    FastAPI はデコレータ時点でルート登録するため両方動くが、Python 名は後勝ちで
    上書きされ、monkeypatch やリファクタが誤った関数を掴む地雷になっていた。

このテストは AST で機械的に検査し、再発をCIで止める。
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DAILY_GREETING_SYMBOLS = {
    "_daily_greeting_read_json",
    "_daily_greeting_git_summary",
    "_daily_greeting_yesterday_note",
    "_daily_greeting_anniversary",
    "_daily_greeting_news_thought",
    "_daily_greeting_opening",
}
EXTRACTED_LEASE_NEWS_SYMBOLS = {
    "_news_vault_root",
    "_lease_news_dir",
    "_fetch_url_text",
    "_validate_public_http_url",
    "_summarize_news_with_gemini",
    "_save_news_to_obsidian",
    "summarize_lease_news",
    "get_recent_lease_news",
}
EXTRACTED_LEASE_INTELLIGENCE_ACTIVITY_SYMBOLS = {
    "LeaseIntelligenceActivityRequest",
    "record_lease_intelligence_activity_api",
    "get_lease_intelligence_related_suggestion_api",
}


def test_api_main_has_no_duplicate_toplevel_definitions():
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    locations: dict[str, list[int]] = collections.defaultdict(list)
    for node in tree.body:  # モジュール直下のみ（クラス内メソッドは対象外）
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            locations[node.name].append(node.lineno)

    duplicates = {name: lines for name, lines in locations.items() if len(lines) > 1}

    assert duplicates == {}, (
        "api/main.py のモジュール直下に同名の二重定義があります（後勝ちで前の定義が"
        f"死ぬか、ルート付きなら名前が乗っ取られます）: {duplicates}"
    )


def test_daily_greeting_helpers_stay_out_of_api_main():
    """shion_tasks routerへ移した挨拶実装をmain.pyへ再コピーしない。"""
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    assert names.isdisjoint(EXTRACTED_DAILY_GREETING_SYMBOLS)


def test_lease_news_implementation_stays_out_of_api_main():
    """lease_news routerへ移した実装をmain.pyへ再コピーしない。"""
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    assert names.isdisjoint(EXTRACTED_LEASE_NEWS_SYMBOLS)


def test_lease_intelligence_activity_stays_out_of_api_main():
    """activity routerへ移した実装をmain.pyへ再コピーしない。"""
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    assert names.isdisjoint(EXTRACTED_LEASE_INTELLIGENCE_ACTIVITY_SYMBOLS)
