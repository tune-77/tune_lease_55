"""紫苑 ADK エージェント用の Vertex AI Search ツール。

`shion_agent_tools.READ_ONLY_DB_TOOLS` はローカル読み取りのみ（追加課金ゼロ）という
線引きで運用されている。Vertex AI Search は課金される外部 API なので、その線引きを
崩さないようここに分離し、環境変数で明示的に opt-in させる。

有効化:
    SHION_ENABLE_VERTEX_TOOLS=1

無効時は `VERTEX_AGENT_TOOLS` が空リストになり、エージェントの構成は一切変わらない。
"""

from __future__ import annotations

import os
from typing import Any

_MAX_REFS = 5
_EXCERPT_LIMIT = 400


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def vertex_tools_enabled() -> bool:
    """Vertex ツールを登録するか（既定は無効 = 追加課金なし）。"""
    return _env_truthy("SHION_ENABLE_VERTEX_TOOLS")


def _trim(value: Any, limit: int = _EXCERPT_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _compact_refs(refs: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for ref in (refs or [])[:_MAX_REFS]:
        if not isinstance(ref, dict):
            continue
        out.append(
            {
                "title": _trim(ref.get("title"), 120),
                "source": _trim(ref.get("source") or ref.get("uri"), 200),
                "excerpt": _trim(ref.get("excerpt") or ref.get("snippet")),
            }
        )
    return out


def search_lease_knowledge_vertex(query: str, limit: int = 5) -> dict[str, Any]:
    """Vertex AI Search でリース審査ナレッジ（Obsidian 由来）を検索する。

    ローカルの search_obsidian_context で裏が取れないときの補助。
    Google 管理の検索インデックスを引くため、ローカル検索より表記ゆれに強い一方、
    同期タイミング次第では Vault の最新版を反映していないことがある。

    Args:
        query: 検索クエリ（日本語可）
        limit: 返す件数の上限（1〜10）

    Returns:
        used / status / refs（title・source・excerpt）を含む辞書。
        未設定・無効時は used=False を返し、エラーにはしない。
    """
    from api.vertex_agent_search import search_vertex_agent

    try:
        size = max(1, min(int(limit), 10))
    except (TypeError, ValueError):
        size = 5

    try:
        result = search_vertex_agent(query, page_size=size)
    except Exception as exc:  # noqa: BLE001 - ツールは落とさず理由を返す
        return {"used": False, "status": f"error: {type(exc).__name__}", "refs": []}

    return {
        "used": bool(result.get("used")),
        "status": result.get("status", ""),
        "refs": _compact_refs(result.get("refs")),
    }


def answer_lease_question_vertex(query: str) -> dict[str, Any]:
    """Vertex AI Search の Answer API で、根拠付きの要約回答を得る。

    検索結果の羅列ではなく引用付きの回答が欲しいときに使う。
    返る answer_text は **審査ナレッジの要約であって審査判断ではない**。
    個別案件の判定は score_full_case / assess_risk_level の結果を根拠にすること。

    Args:
        query: 質問文（日本語可）

    Returns:
        used / status / answer_text / refs / grounding を含む辞書。
        未設定・無効時は used=False を返し、エラーにはしない。
    """
    from api.vertex_agent_search import answer_vertex_agent

    try:
        result = answer_vertex_agent(query)
    except Exception as exc:  # noqa: BLE001 - ツールは落とさず理由を返す
        return {"used": False, "status": f"error: {type(exc).__name__}", "answer_text": "", "refs": []}

    grounding = result.get("grounding") or {}
    return {
        "used": bool(result.get("used")),
        "status": result.get("status", ""),
        "answer_text": _trim(result.get("answer_text"), 2000),
        "refs": _compact_refs(result.get("refs")),
        "grounding_score": grounding.get("citation_coverage_score"),
    }


def build_vertex_agent_tools() -> list:
    """opt-in 時だけツールを返す。

    env をモジュール読み込み時に一度だけ見る作りにすると、import 順に依存して
    有効/無効が決まってしまう。判定は呼び出しのたびに行う。
    """
    if not vertex_tools_enabled():
        return []
    return [search_lease_knowledge_vertex, answer_lease_question_vertex]


# エージェント側はこのリストを素通しで連結する（import 時点の env で確定）。
VERTEX_AGENT_TOOLS = build_vertex_agent_tools()
