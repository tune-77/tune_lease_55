"""
審査判断乖離学習ループ（Judgment Divergence Loop Engineering）。

Observe   : data/screening_loop_feedback.jsonl（争点・稟議方針への人間フィードバック）
Aggregate : rating別の件数と、否定的評価（少し違う/違う/使えない）の
            争点・稟議方針テキストをスコア・判定とあわせてまとめる
Propose   : 否定的評価に共通する傾向をGeminiに分析させ、審査ロジックの
            レビュー観点を提案させる
Persist   : data/judgment_divergence_proposals.jsonl

安全設計: このループは scoring_core.py の係数・閾値を一切書き換えない。
Geminiが返すのは「人間が確認すべき着眼点」の自然文であり、実際に
ロジックを変えるかどうかは別途人間が scoring_core.py 側で判断する。
"""
from __future__ import annotations

from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, call_gemini_json, load_jsonl

_FEEDBACK_PATH = DATA_DIR / "screening_loop_feedback.jsonl"
_PROPOSALS_PATH = DATA_DIR / "judgment_divergence_proposals.jsonl"

_NEGATIVE_RATINGS = {"違う", "使えない"}
_MIXED_RATINGS = {"少し違う", "修正して使う"}


def aggregate_divergence(limit_examples: int = 12) -> dict[str, Any]:
    """否定的・中間的評価の傾向を集計する。"""
    entries = load_jsonl(_FEEDBACK_PATH, newest_first=True)
    counts: dict[str, int] = {}
    negative_examples: list[dict[str, Any]] = []
    for entry in entries:
        rating = str(entry.get("rating") or "")
        counts[rating] = counts.get(rating, 0) + 1
        if rating in _NEGATIVE_RATINGS or rating in _MIXED_RATINGS:
            if len(negative_examples) >= limit_examples:
                continue
            text = str(entry.get("issue_text") or entry.get("ringi_policy_text") or "").strip()
            if not text:
                continue
            negative_examples.append(
                {
                    "target": str(entry.get("target") or ""),
                    "rating": rating,
                    "text": text,
                    "comment": str(entry.get("comment") or ""),
                    "score": entry.get("score"),
                    "hantei": str(entry.get("hantei") or ""),
                }
            )
    return {
        "total_feedback": len(entries),
        "rating_counts": counts,
        "negative_examples": negative_examples,
    }


def _build_prompt(aggregate: dict[str, Any]) -> str:
    examples_lines = "\n".join(
        f"- [{ex['target']}/{ex['rating']}] score={ex['score']}, hantei={ex['hantei']}: "
        f"{ex['text']}" + (f"（メモ: {ex['comment']}）" if ex["comment"] else "")
        for ex in aggregate["negative_examples"]
    ) or "（該当データなし）"
    counts_lines = "\n".join(f"- {k}: {v}件" for k, v in aggregate["rating_counts"].items()) or "（データなし）"

    return f"""あなたはリース審査AIシステム「紫苑」です。審査担当者が「争点」「稟議方針」の
AI提案に対して残したフィードバックを分析し、審査ロジックのレビュー観点を考えるのが役目です。

【評価件数】
{counts_lines}

【否定的・修正が必要だった事例（争点/稟議方針テキストと当時のスコア・判定）】
{examples_lines}

これらの事例に共通する傾向を分析し、審査担当者が実際に scoring_core.py 側の
ロジックや基準を見直す際の着眼点を2〜4件、以下のJSON配列形式のみで返してください
（前後の説明テキストは不要）:

[
  {{
    "title": "着眼点のタイトル（30字以内）",
    "observation": "どの事例からどんな傾向が見えたか（100字程度）",
    "review_point": "審査担当者が確認すべき具体的な観点（係数の値やコード変更を断定せず、確認・検証を促す表現にする）"
  }}
]

重要: あなたはscoring_core.pyの係数や閾値を直接変更する権限を持ちません。
提案はすべて「人間が確認・検証すべき観点」として書いてください。"""


def generate_proposals() -> dict[str, Any]:
    aggregate = aggregate_divergence()
    if aggregate["total_feedback"] == 0:
        return {"generated": False, "reason": "審査フィードバックデータがまだありません", "proposals": []}
    if not aggregate["negative_examples"]:
        return {
            "generated": False,
            "reason": "否定的・修正が必要だったフィードバックが見つかりませんでした（良好）",
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
            "observation": str(item.get("observation") or "").strip(),
            "review_point": str(item.get("review_point") or "").strip(),
            "generated_at": generated_at,
            "status": "needs_human_review",
        }
        append_jsonl(_PROPOSALS_PATH, entry)
        saved.append(entry)

    return {"generated": True, "aggregate": aggregate, "proposals": saved}


def load_proposals(limit: int = 20) -> list[dict[str, Any]]:
    return load_jsonl(_PROPOSALS_PATH, limit=limit, newest_first=True)
