"""紫苑の音声通話（Gemini Live）で使う読み取り専用ツール（REV-531）。

文字チャット・ADK エージェントと同じツール関数（lease_intelligence_tools / READ_ONLY_DB_TOOLS）を、
Live の function calling から /api/shion/voice/tool 経由で呼ぶ。書き込み・外部課金のあるツールは載せない。
宣言は毎ターン Live の入力に乗るので、説明は短くする。
"""
from __future__ import annotations

import json
from typing import Any, Callable

MAX_RESULT_CHARS = 4000
RECALL_LIMIT = 5


def _recall_memory(query: str) -> Any:
    from api.shion_memory_recall import build_recall_prompt_block

    text, _ = build_recall_prompt_block(query, limit=RECALL_LIMIT)
    return text or "該当する記憶はありません"


def _search_obsidian(query: str) -> Any:
    from lease_intelligence_tools import search_obsidian_context

    return search_obsidian_context(query, limit=4)


def _recall_judgment(question: str) -> Any:
    from lease_intelligence_tools import recall_judgment_memory

    return recall_judgment_memory(question, limit=5)


def _score_case(**kwargs: Any) -> Any:
    from lease_intelligence_tools import score_full_case

    return score_full_case(**kwargs)


def _score_detail(company_name: str) -> Any:
    from lease_intelligence_tools import get_score_detail

    return get_score_detail(company_name)


def _search_cases(query: str, limit: int = 5) -> Any:
    from lease_intelligence_tools import search_cases

    return search_cases(query, limit=max(1, min(int(limit or 5), 10)))


def _portfolio_stats() -> Any:
    from lease_intelligence_tools import get_portfolio_stats

    return get_portfolio_stats()


_S = {"STRING": {"type": "STRING"}, "NUMBER": {"type": "NUMBER"}, "INTEGER": {"type": "INTEGER"}}

# name -> (関数, 説明, 引数スキーマ, 必須引数)
VOICE_TOOLS: dict[str, tuple[Callable[..., Any], str, dict[str, dict], list[str]]] = {
    "recall_memory": (_recall_memory, "紫苑の過去の記憶・会話を検索する", {"query": _S["STRING"]}, ["query"]),
    "search_obsidian_context": (
        _search_obsidian,
        "Obsidian の知識ノート（業界・制度・審査の知見）を検索する。業種や論点の相談で根拠を取る",
        {"query": _S["STRING"]},
        ["query"],
    ),
    "recall_judgment_memory": (
        _recall_judgment,
        "判断資産（社内方針・審査の判断基準）と紫苑の判断の記憶を想起する",
        {"question": _S["STRING"]},
        ["question"],
    ),
    "score_full_case": (
        _score_case,
        "案件条件から審査スコアを試算する（保存しない）。金額はすべて千円単位（5億円=500000、2000万円=20000）。lease_term は月",
        {
            "industry_major": {"type": "STRING", "description": "大分類業種 例: 'D 建設業'"},
            "nenshu": _S["NUMBER"],
            "op_profit": _S["NUMBER"],
            "net_assets": _S["NUMBER"],
            "total_assets": _S["NUMBER"],
            "acquisition_cost": _S["NUMBER"],
            "lease_term": _S["INTEGER"],
            "customer_type": {"type": "STRING", "description": "'既存先' か '新規先'"},
            "asset_name": _S["STRING"],
        },
        ["industry_major", "nenshu", "op_profit", "net_assets", "total_assets", "acquisition_cost"],
    ),
    "get_score_detail": (_score_detail, "企業名で過去の審査スコアの内訳を取得する", {"company_name": _S["STRING"]}, ["company_name"]),
    "search_cases": (
        _search_cases,
        "過去・類似の審査案件を検索する",
        {"query": _S["STRING"], "limit": _S["INTEGER"]},
        ["query"],
    ),
    "get_portfolio_stats": (_portfolio_stats, "審査DB全体の統計（件数・成約率・業種構成）", {}, []),
}


def declarations() -> list[Any]:
    from google.genai import types

    out = []
    for name, (_fn, description, props, required) in VOICE_TOOLS.items():
        schema = None
        if props:
            schema = types.Schema(
                type="OBJECT",
                properties={key: types.Schema(**value) for key, value in props.items()},
                required=required,
            )
        out.append(types.FunctionDeclaration(name=name, description=description, parameters=schema))
    return out


def run_tool(name: str, args: dict[str, Any] | None) -> str:
    """Live のツール呼び出しを実行し、読み上げ前提の短い文字列で返す（失敗しても通話は止めない）。"""
    entry = VOICE_TOOLS.get(name)
    if entry is None:
        return f"{name} は通話では使えません"
    fn, _description, props, _required = entry
    kwargs = {key: value for key, value in (args or {}).items() if key in props and value not in (None, "")}
    try:
        result = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - ツールの失敗は結果として紫苑に伝える
        return f"{name} を実行できませんでした（{type(exc).__name__}）"
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    return text[:MAX_RESULT_CHARS]
