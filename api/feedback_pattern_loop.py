"""
人間反応フィードバック傾向分析ループ（Feedback Pattern Loop Engineering）。

Observe   : data/human_response_feedback.jsonl（紫苑の応答への人間評価）
            data/shion_experience_events.jsonl（紫苑の経験イベント・不確実性シグナル）
Aggregate : rating別の件数と、否定的評価（thin/generic/not_shion/bad）の
            質問・応答の実例をまとめる。経験イベントの不確実性・低スコア事例も含める。
Propose   : 否定的評価と不確実性シグナルに共通する状況をGeminiに分析させ、応答スタンス・
            プロンプト調整の観点を提案させる
Persist   : data/feedback_pattern_proposals.jsonl

PDCA評価  : 採用済み提案の before/after フィードバック率を比較し、効果を記録する
            → data/shion_self_pdca_log.jsonl

安全設計: このループはプロンプトやシステム指示を自動で書き換えない。
Geminiが返すのは「人間が確認すべき着眼点」であり、実際の調整は
別途人間が判断する。
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, call_gemini_json, load_jsonl

_FEEDBACK_PATH = DATA_DIR / "human_response_feedback.jsonl"
_EXPERIENCE_PATH = DATA_DIR / "shion_experience_events.jsonl"
_IMPROVEMENT_LOG_PATH = DATA_DIR / "cloudrun_improvement_log.jsonl"
_PROPOSALS_PATH = DATA_DIR / "feedback_pattern_proposals.jsonl"
_PDCA_LOG_PATH = DATA_DIR / "shion_self_pdca_log.jsonl"

_NEGATIVE_RATINGS = {"thin", "generic", "not_shion", "bad"}

# 経験イベントで「不確実性が高い」または「両スコアが低い」ものを負のシグナルとみなす
_UNCERTAINTY_THRESHOLD = 1   # signals.uncertainty >= この値
_LOW_SCORE_THRESHOLD = 1     # relationship_depth と practical_depth がともに <= この値


def aggregate_feedback(limit_examples: int = 12) -> dict[str, Any]:
    """人間評価フィードバックを集計する。"""
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


def aggregate_experience_signals(limit_examples: int = 8) -> dict[str, Any]:
    """
    shion_experience_events.jsonl から不確実性シグナルを集計する。
    uncertainty >= 1 または (relationship_depth <= 1 AND practical_depth <= 1) の事例を抽出。
    """
    if not _EXPERIENCE_PATH.exists():
        return {"total_events": 0, "weak_signal_count": 0, "weak_examples": []}

    events = load_jsonl(_EXPERIENCE_PATH, newest_first=True)
    weak_examples: list[dict[str, Any]] = []
    for e in events:
        signals = e.get("signals") or {}
        uncertainty = int(signals.get("uncertainty") or 0)
        rel = int(signals.get("relationship_depth") or 0)
        prac = int(signals.get("practical_depth") or 0)
        is_weak = (
            uncertainty >= _UNCERTAINTY_THRESHOLD
            or (rel <= _LOW_SCORE_THRESHOLD and prac <= _LOW_SCORE_THRESHOLD)
        )
        if is_weak and len(weak_examples) < limit_examples:
            weak_examples.append(
                {
                    "route": str(e.get("route") or signals.get("route") or ""),
                    "message": str(e.get("message_preview") or "")[:150],
                    "response": str(e.get("response_start") or "")[:150],
                    "uncertainty": uncertainty,
                    "relationship_depth": rel,
                    "practical_depth": prac,
                    "delta": str(e.get("delta") or ""),
                }
            )
    return {
        "total_events": len(events),
        "weak_signal_count": len([
            e for e in events
            if (int((e.get("signals") or {}).get("uncertainty") or 0) >= _UNCERTAINTY_THRESHOLD
                or (int((e.get("signals") or {}).get("relationship_depth") or 0) <= _LOW_SCORE_THRESHOLD
                    and int((e.get("signals") or {}).get("practical_depth") or 0) <= _LOW_SCORE_THRESHOLD))
        ]),
        "weak_examples": weak_examples,
    }


def _build_prompt(aggregate: dict[str, Any], experience: dict[str, Any]) -> str:
    counts_lines = "\n".join(f"- {k}: {v}件" for k, v in aggregate["rating_counts"].items()) or "（データなし）"
    examples_lines = "\n".join(
        f"- [{ex['rating']}/{ex['route']}] Q: {ex['message']}\n  A: {ex['response']}"
        + (f"\n  メモ: {ex['comment']}" if ex["comment"] else "")
        for ex in aggregate["negative_examples"]
    ) or "（該当データなし）"

    exp_lines = "\n".join(
        f"- [route:{ex['route']} / uncertainty:{ex['uncertainty']} / rel:{ex['relationship_depth']} prac:{ex['practical_depth']}]"
        f"\n  Q: {ex['message']}\n  A: {ex['response']}"
        + (f"\n  変化: {ex['delta']}" if ex["delta"] else "")
        for ex in experience["weak_examples"]
    ) or "（該当なし）"

    return f"""あなたはリース審査AIシステム「紫苑」です。以下の2種類のデータを統合分析し、
応答の質を改善する着眼点を考えるのが役目です。

【①人間による評価フィードバック（件数）】
{counts_lines}

【①「薄い」「一般論」「紫苑らしくない」と評価された実例】
{examples_lines}

【②紫苑経験イベントの弱シグナル（不確実性または低スコア事例 {experience["weak_signal_count"]}/{experience["total_events"]}件）】
{exp_lines}

これら2種類のデータに共通する課題パターンを分析し、応答スタンスや
プロンプト設計を見直す際の着眼点を2〜4件、以下のJSON配列形式のみで返してください
（前後の説明テキストは不要）:

[
  {{
    "title": "着眼点のタイトル（30字以内）",
    "pattern": "どんな状況で問題が起きやすいか（100字程度）",
    "suggestion": "応答スタンス・プロンプトのどこを見直すとよいか、具体的な提案"
  }}
]

重要: あなたはシステムプロンプトを直接書き換える権限を持ちません。
提案はすべて「人間が確認・検証すべき観点」として書いてください。"""


def generate_proposals() -> dict[str, Any]:
    aggregate = aggregate_feedback()
    experience = aggregate_experience_signals()

    has_feedback = aggregate["total_feedback"] > 0 and bool(aggregate["negative_examples"])
    has_experience = experience["weak_signal_count"] > 0

    if not has_feedback and not has_experience:
        return {"generated": False, "reason": "分析に使えるデータがまだありません", "proposals": []}

    prompt = _build_prompt(aggregate, experience)
    try:
        proposals = call_gemini_json(prompt)
        if not isinstance(proposals, list):
            raise ValueError("Gemini応答がリストではありません")
    except Exception as exc:
        return {"generated": False, "reason": f"Gemini生成に失敗: {exc}", "proposals": []}

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

    return {
        "generated": True,
        "aggregate": aggregate,
        "experience": experience,
        "proposals": saved,
    }


def evaluate_proposal_impact() -> dict[str, Any]:
    """
    PDCA評価: 採用済みの紫苑自己提案について、提案前後のフィードバック否定率を比較する。

    手順:
    1. cloudrun_improvement_log.jsonl から proposed_by='shion' かつ status='approved' のエントリを取得
    2. 各提案の generated_at を基準日として、前後の human_response_feedback の negative_rate を比較
    3. 結果を shion_self_pdca_log.jsonl に追記
    """
    if not _IMPROVEMENT_LOG_PATH.exists():
        return {"evaluated": 0, "results": []}

    # 採用済み紫苑提案を収集
    adopted: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for line in _IMPROVEMENT_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            entry.get("proposed_by") == "shion"
            and entry.get("surface") == "shion_self_proposal"
        ):
            title = str(entry.get("title") or "").strip()
            ts = str(entry.get("ts") or "")
            if title and title not in seen_titles and ts:
                adopted.append({"title": title, "ts": ts})
                seen_titles.add(title)

    if not adopted:
        return {"evaluated": 0, "results": [], "reason": "採用済み紫苑提案がまだありません"}

    # feedback データを日付ごとに集計
    fb_entries = load_jsonl(_FEEDBACK_PATH, newest_first=False)

    def _negative_rate(entries: list[dict[str, Any]]) -> float:
        if not entries:
            return 0.0
        neg = sum(1 for e in entries if str(e.get("rating") or "") in _NEGATIVE_RATINGS)
        return neg / len(entries)

    results: list[dict[str, Any]] = []
    evaluated_at = dt.datetime.now().isoformat(timespec="seconds")

    for proposal in adopted:
        try:
            pivot = dt.datetime.fromisoformat(proposal["ts"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        before = [
            e for e in fb_entries
            if _parse_ts(e.get("ts")) is not None and _parse_ts(e.get("ts")) < pivot  # type: ignore[operator]
        ]
        after = [
            e for e in fb_entries
            if _parse_ts(e.get("ts")) is not None and _parse_ts(e.get("ts")) >= pivot  # type: ignore[operator]
        ]

        rate_before = _negative_rate(before)
        rate_after = _negative_rate(after)
        delta = rate_after - rate_before  # 負 = 改善、正 = 悪化

        result = {
            "title": proposal["title"],
            "proposal_ts": proposal["ts"],
            "feedback_before": len(before),
            "feedback_after": len(after),
            "negative_rate_before": round(rate_before, 3),
            "negative_rate_after": round(rate_after, 3),
            "delta": round(delta, 3),
            "verdict": "improved" if delta < -0.05 else ("degraded" if delta > 0.05 else "no_change"),
            "evaluated_at": evaluated_at,
        }
        results.append(result)
        append_jsonl(_PDCA_LOG_PATH, result)

    return {"evaluated": len(results), "results": results}


def _parse_ts(ts_str: Any) -> dt.datetime | None:
    """ISO形式のタイムスタンプをパース。失敗時は None を返す。"""
    if not ts_str:
        return None
    try:
        return dt.datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def load_proposals(limit: int = 20) -> list[dict[str, Any]]:
    return load_jsonl(_PROPOSALS_PATH, limit=limit, newest_first=True)


def load_pdca_log(limit: int = 20) -> list[dict[str, Any]]:
    return load_jsonl(_PDCA_LOG_PATH, limit=limit, newest_first=True)
