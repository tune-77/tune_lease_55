"""紫苑記憶想起のLLMリランカー（opt-in・fail-open）。

キーワード＋ベクトル加点で選んだ想起候補の上位プールを、Gemini に
「質問に本当に関連する順」で並べ直させる。言い換え・文脈依存の質問で
キーワード一致だけでは順位が崩れるケースを救う二段目。

設計原則:
- SHION_MEMORY_RERANK=1 のときだけ動く（既定OFF。まず環境比較で効果測定する）
- 失敗（APIキー無し・タイムアウト・不正応答）は必ず元の順序へフォールバック
  し、チャット応答を止めない
- LLMが返さなかった候補は元の相対順序のままリランク済みの後ろに置く
  （記憶を落とさない。最終選抜は従来の _select_records が行う）
"""

from __future__ import annotations

import os
from typing import Any

_POOL_SIZE = 12
_CONTENT_CHARS = 160


def rerank_enabled() -> bool:
    return os.environ.get("SHION_MEMORY_RERANK", "").strip().lower() in {"1", "true", "on"}


def _build_prompt(question: str, pool: list[tuple[float, dict[str, Any]]]) -> str:
    lines = [
        "あなたはリース審査AIの記憶検索の再ランク付け器です。",
        "以下の質問に回答するために本当に役立つ記憶だけを、関連が強い順に並べてください。",
        "",
        f"質問: {question}",
        "",
        "記憶候補:",
    ]
    for _, record in pool:
        rid = str(record.get("id") or "")
        content = " ".join(str(record.get("content") or "").split())[:_CONTENT_CHARS]
        lines.append(f"- {rid}: {content}")
    lines += [
        "",
        '出力は JSON 配列のみ: ["関連が強い順のid", ...]',
        "無関係な候補は配列から除外してよい。説明文は書かない。",
    ]
    return "\n".join(lines)


def maybe_rerank_scored(
    question: str,
    scored: list[tuple[float, dict[str, Any]]],
    *,
    pool_size: int = _POOL_SIZE,
) -> tuple[list[tuple[float, dict[str, Any]]], bool]:
    """スコア降順の候補リストを（可能なら）LLMで並べ直す。

    戻り値: (並べ直し後のリスト, リランクを実際に適用したか)
    """
    if not rerank_enabled() or len(scored) <= 1:
        return scored, False
    pool = scored[:pool_size]
    rest = scored[pool_size:]
    try:
        from api.loop_engineering_common import call_gemini_json

        result = call_gemini_json(
            _build_prompt(question, pool), temperature=0.0, max_output_tokens=512
        )
    except Exception:
        return scored, False
    if not isinstance(result, list):
        return scored, False
    ordered_ids = [str(item) for item in result if isinstance(item, (str, int))]
    if not ordered_ids:
        return scored, False

    by_id = {str(record.get("id") or ""): (score, record) for score, record in pool}
    reranked: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for rid in ordered_ids:
        if rid in by_id and rid not in seen:
            reranked.append(by_id[rid])
            seen.add(rid)
    # LLMが言及しなかった候補は元の相対順序で後ろへ（記憶を落とさない）
    for score, record in pool:
        rid = str(record.get("id") or "")
        if rid not in seen:
            reranked.append((score, record))
    return reranked + rest, True
