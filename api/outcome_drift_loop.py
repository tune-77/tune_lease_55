"""
審査実績ドリフト監視ループ（Outcome Drift Loop Engineering）。

Observe   : payment_history テーブル（成約後の実際の支払い状況。正常/延滞/
            デフォルト/完済）を読み取り専用で参照する
Aggregate : screening_score を scoring_core.APPROVAL_LINE 基準の帯に分け、
            帯ごとの延滞・デフォルト率を集計する
Propose   : 「本来低リスクなはずの帯で延滞・デフォルトが多い」等の乖離を
            Geminiに解釈させ、確認すべき観点を提案させる
Persist   : data/outcome_drift_proposals.jsonl

安全設計: このループは scoring_core.py・payment_history のいずれにも
書き込みを行わない（SELECTのみ）。統計的な精緻さより「気づきの入口」を
優先しており、厳密なモデル再学習の代わりにはならない。
"""
from __future__ import annotations

from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, call_gemini_json, load_jsonl

_PROPOSALS_PATH = DATA_DIR / "outcome_drift_proposals.jsonl"

_BAD_STATUSES = {"延滞", "デフォルト"}


def _score_bucket(score: float, approval_line: float) -> str:
    if score >= approval_line:
        return f"承認圏({approval_line:.0f}点以上)"
    if score >= approval_line - 11:
        return f"条件付き圏({approval_line - 11:.0f}〜{approval_line - 1:.0f}点)"
    return f"否決圏({approval_line - 11:.0f}点未満)"


def aggregate_outcomes() -> dict[str, Any]:
    from scoring_core import APPROVAL_LINE
    from api.db_connection import get_connection

    rows: list[tuple] = []
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT screening_score, payment_status, overdue_amount "
                "FROM payment_history WHERE screening_score IS NOT NULL"
            )
            rows = cur.fetchall()
    except Exception as exc:
        return {"available": False, "reason": f"payment_history読み取り失敗: {exc}", "buckets": []}

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        score, status, overdue = row[0], str(row[1] or ""), row[2] or 0
        if score is None:
            continue
        key = _score_bucket(float(score), APPROVAL_LINE)
        bucket = buckets.setdefault(key, {"bucket": key, "total": 0, "bad_count": 0, "overdue_amount_sum": 0})
        bucket["total"] += 1
        if status in _BAD_STATUSES:
            bucket["bad_count"] += 1
        bucket["overdue_amount_sum"] += int(overdue or 0)

    for bucket in buckets.values():
        bucket["bad_rate"] = round(bucket["bad_count"] / bucket["total"], 3) if bucket["total"] else 0.0

    return {
        "available": True,
        "approval_line": APPROVAL_LINE,
        "total_records": len(rows),
        "buckets": sorted(buckets.values(), key=lambda b: b["bucket"]),
    }


def _build_prompt(aggregate: dict[str, Any]) -> str:
    bucket_lines = "\n".join(
        f"- {b['bucket']}: 件数={b['total']}, 延滞/デフォルト率={b['bad_rate'] * 100:.1f}%, "
        f"延滞金額合計={b['overdue_amount_sum']}千円"
        for b in aggregate["buckets"]
    ) or "（データなし）"

    return f"""あなたはリース審査AIシステム「紫苑」です。成約後の実際の支払い実績
（正常/延滞/デフォルト/完済）を、審査時のスコア帯ごとに集計した結果を分析し、
審査モデルの精度ドリフトに気づく役目を持っています。

【承認ライン】{aggregate['approval_line']}点
【スコア帯ごとの実績】（承認圏=本来低リスクのはずの帯）
{bucket_lines}

この集計を見て、「本来低リスクなはずの帯で延滞・デフォルト率が高い」
「帯によって傾向が想定と逆になっている」等の乖離があれば、審査担当者が
確認すべき観点を2〜4件、以下のJSON配列形式のみで返してください
（乖離が見当たらない場合は空配列 [] を返してください。前後の説明テキストは不要）:

[
  {{
    "title": "着眼点のタイトル（30字以内）",
    "observation": "どの帯でどんな乖離が見えたか、数字を根拠に（100字程度）",
    "review_point": "審査担当者が確認すべき具体的な観点（統計的な精緻さは限定的である前提で、断定せず確認を促す表現にする）"
  }}
]"""


def generate_proposals() -> dict[str, Any]:
    aggregate = aggregate_outcomes()
    if not aggregate["available"]:
        return {"generated": False, "reason": aggregate.get("reason", "データ取得に失敗しました"), "proposals": []}
    if aggregate["total_records"] == 0:
        return {"generated": False, "reason": "支払い実績データがまだありません", "proposals": []}

    prompt = _build_prompt(aggregate)
    try:
        proposals = call_gemini_json(prompt)
        if not isinstance(proposals, list):
            raise ValueError("Gemini応答がリストではありません")
    except Exception as exc:
        return {"generated": False, "reason": f"Gemini生成に失敗: {exc}", "proposals": []}

    if not proposals:
        return {"generated": True, "reason": "明確な乖離は見つかりませんでした（良好）", "aggregate": aggregate, "proposals": []}

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
