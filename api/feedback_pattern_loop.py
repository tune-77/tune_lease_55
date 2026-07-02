"""
人間反応フィードバック傾向分析ループ（Feedback Pattern Loop Engineering）。

Observe   : data/human_response_feedback.jsonl（紫苑の応答への人間評価）
Aggregate : rating別の件数と、否定的評価（thin/generic/not_shion/bad）の
            質問・応答の実例をまとめる
Propose   : 否定的評価に共通する状況をGeminiに分析させ、応答スタンス・
            プロンプト調整の観点を提案させる
Persist   : data/feedback_pattern_proposals.jsonl

安全設計: このループはプロンプトやシステム指示を自動で書き換えない。
Geminiが返すのは「人間が確認すべき着眼点」であり、実際の調整は
別途人間が判断する。
"""
from __future__ import annotations

from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, call_gemini_json, load_jsonl

_FEEDBACK_PATH = DATA_DIR / "human_response_feedback.jsonl"
_PROPOSALS_PATH = DATA_DIR / "feedback_pattern_proposals.jsonl"

_NEGATIVE_RATINGS = {"thin", "generic", "not_shion", "bad"}


def aggregate_feedback(limit_examples: int = 12) -> dict[str, Any]:
    entries = load_jsonl(_FEEDBACK_PATH, newest_first=True)
    counts: dict[str, int] = {}
    negative_examples: list[dict[str, Any]] = []
    for entry in entries:
        rating = str(entry.get("rating") or "")
        counts[rating] = counts.get(rating, 0) + 1
        if rating in _NEGATIVE_RATINGS and len(negative_examples) < limit_examples:
            negative_examples.append(
                {
                    "rating": rating,
                    "route": str(entry.get("route") or ""),
                    "message": str(entry.get("message_preview") or ""),
                    "response": str(entry.get("response_start") or "")[:200],
                    "comment": str(entry.get("comment") or ""),
                }
            )
    return {
        "total_feedback": len(entries),
        "rating_counts": counts,
        "negative_examples": negative_examples,
    }


def _build_prompt(aggregate: dict[str, Any]) -> str:
    counts_lines = "\n".join(f"- {k}: {v}件" for k, v in aggregate["rating_counts"].items()) or "（データなし）"
    examples_lines = "\n".join(
        f"- [{ex['rating']}/{ex['route']}] Q: {ex['message']}\n  A: {ex['response']}"
        + (f"\n  メモ: {ex['comment']}" if ex["comment"] else "")
        for ex in aggregate["negative_examples"]
    ) or "（該当データなし）"

    return f"""あなたはリース審査AIシステム「紫苑」です。Userが紫苑の応答へ残した評価
（shion_like/good/thin/generic/not_shion/bad）を分析し、応答の質を改善する
着眼点を考えるのが役目です。

【評価件数】
{counts_lines}

【「薄い」「一般論」「紫苑らしくない」と評価された質問・応答の実例】
{examples_lines}

これらの実例に共通する状況（質問の種類、話題、routeなど）を分析し、応答スタンスや
プロンプト設計を見直す際の着眼点を2〜4件、以下のJSON配列形式のみで返してください
（前後の説明テキストは不要）:

[
  {{
    "title": "着眼点のタイトル（30字以内）",
    "pattern": "どんな状況で否定的評価が起きやすいか（100字程度）",
    "suggestion": "応答スタンス・プロンプトのどこを見直すとよいか、具体的な提案"
  }}
]

重要: あなたはシステムプロンプトを直接書き換える権限を持ちません。
提案はすべて「人間が確認・検証すべき観点」として書いてください。"""


def generate_proposals() -> dict[str, Any]:
    aggregate = aggregate_feedback()
    if aggregate["total_feedback"] == 0:
        return {"generated": False, "reason": "人間反応フィードバックデータがまだありません", "proposals": []}
    if not aggregate["negative_examples"]:
        return {
            "generated": False,
            "reason": "否定的評価が見つかりませんでした（良好）",
            "proposals": [],
        }

    prompt = _build_prompt(aggregate)
    try:
        proposals = call_gemini_json(prompt)
        if not isinstance(proposals, list):
            raise ValueError("Gemini応答がリストではありません")
    except Exception as exc:
        return {"generated": False, "reason": f"Gemini生成に失敗: {exc}", "proposals": []}

    import datetime as dt

    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    saved: list[dict[str, Any]] = []
    for item in proposals:
        if not isinstance(item, dict) or not str(item.get("title") or "").strip():
            continue
        entry = {
            "title": str(item.get("title") or "").strip(),
            "pattern": str(item.get("pattern") or "").strip(),
            "suggestion": str(item.get("suggestion") or "").strip(),
            "generated_at": generated_at,
            "status": "needs_human_review",
        }
        append_jsonl(_PROPOSALS_PATH, entry)
        saved.append(entry)

    return {"generated": True, "aggregate": aggregate, "proposals": saved}


def load_proposals(limit: int = 20) -> list[dict[str, Any]]:
    return load_jsonl(_PROPOSALS_PATH, limit=limit, newest_first=True)
