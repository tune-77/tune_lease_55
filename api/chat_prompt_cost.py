"""`/api/chat` のシステムプロンプト構成コストを計測する。

`_cap_system_prompt` は合計文字数の上限（既定24000字）を超えた時に末尾ブロックを
落とすが、落とした「数」しかログに残らず、どのブロックがどれだけ食っていたかは
記録されていなかった。そのため削減を検討しても対象を数字で選べない。

このモジュールは組み立て済みブロックのサイズを `data/chat_prompt_composition.jsonl`
へ1行1レコードで追記する。`memory_layers/README.md` の Retrieval Boundary が求める
`token_budget` 相当の記録を、チャット経路側で満たすためのもの。

計測専用。プロンプト内容・ルーティング・スコアリングには一切触れない。
記録に失敗しても呼び出し元は止めない。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl

_LOG_PATH = DATA_DIR / "chat_prompt_composition.jsonl"

# 概算トークン。scripts/build_obsidian_retrieval_graph.py の _token_estimate と
# 同じ概算則（3文字≒1トークン）を使い、他レポートと数値の意味を揃える。
_CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    """日本語混在テキストの概算トークン数。空文字は0を返す。"""
    length = len(str(text or ""))
    return length // _CHARS_PER_TOKEN if length else 0


def build_composition(
    *,
    surface: str,
    blocks: dict[str, str],
    final_prompt: str,
    base_prompt: str = "",
) -> dict[str, Any]:
    """ブロック別サイズと合計・未計上分をまとめる（IOなし・テスト用に分離）。

    unaccounted_chars は「組み立て結果 − 計測したブロックの合計」。名前を付けて
    いないブロックが太った場合でも、この差分に出るので取りこぼしに気づける。
    """
    measured: dict[str, dict[str, int]] = {}
    measured_chars = 0
    for name, block in blocks.items():
        chars = len(str(block or ""))
        if not chars:
            continue
        measured[name] = {"chars": chars, "tokens": estimate_tokens(block)}
        measured_chars += chars

    base_chars = len(str(base_prompt or ""))
    final_chars = len(str(final_prompt or ""))
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "surface": surface,
        "final_chars": final_chars,
        "final_tokens": estimate_tokens(final_prompt),
        "measured_chars": measured_chars,
        "unaccounted_chars": max(0, final_chars - measured_chars),
        # base_chars > final_chars なら _cap_system_prompt が末尾ブロックを落としている
        "capped": bool(base_chars and final_chars < base_chars),
        "dropped_chars": max(0, base_chars - final_chars),
        "block_count": len(measured),
        "blocks": measured,
    }


def log_prompt_composition(
    *,
    surface: str,
    blocks: dict[str, str],
    final_prompt: str,
    base_prompt: str = "",
) -> None:
    """プロンプト構成コストを追記する。失敗しても呼び出し元を止めない。"""
    try:
        entry = build_composition(
            surface=surface,
            blocks=blocks,
            final_prompt=final_prompt,
            base_prompt=base_prompt,
        )
        append_jsonl(_LOG_PATH, entry)
    except Exception:
        pass
